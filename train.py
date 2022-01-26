import torch.utils.data as data
import torchvision.transforms as transforms
from model.utils import Reconstruction3DDataLoader, Reconstruction3DDataLoaderJump
from model.autoencoder import *
from utils import *

import argparse

parser = argparse.ArgumentParser(description="STEAL Net")
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate phase 1')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--dataset_type', type=str, default='ped2', choices=['ped2','avenue', 'shanghai'], help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='basename of folder to save weights')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam','sgd'], help='adam or sgd with momentum and cosine annealing lr')

parser.add_argument('--model_dir', type=str, default=None, help='path of model for resume')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch. usually number in filename + 1')

# related to skipping frame pseudo anomaly
parser.add_argument('--pseudo_anomaly_jump', type=float, default=0, help='pseudo anomaly jump frame (skip frame) probability. 0 no pseudo anomaly')
parser.add_argument('--jump', nargs='+', type=int, default=[3], help='Jump for pseudo anomaly (hyperparameter s)')  # --jump 2 3

parser.add_argument('--print_all', action='store_true', help='print all reconstruction loss')

##################

args = parser.parse_args()

# assert 1 not in args.jump

exp_dir = args.exp_dir
exp_dir += 'lr' + str(args.lr) if args.lr != 1e-4 else ''
exp_dir += 'weight'
exp_dir += '_recon'

exp_dir += '_pajump' + str(args.pseudo_anomaly_jump) if args.pseudo_anomaly_jump != 0 else ''
exp_dir += '_jump[' + ','.join([str(args.jump[i]) for i in range(0,len(args.jump))]) + ']' if args.pseudo_anomaly_jump != 0 else ''

print('exp_dir: ', exp_dir)

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

train_folder = os.path.join(args.dataset_path, args.dataset_type, 'training', 'frames')

# Loading dataset
img_extension = '.tif' if args.dataset_type == 'ped1' else '.jpg'
train_dataset = Reconstruction3DDataLoader(train_folder, transforms.Compose([transforms.ToTensor()]),
                                           resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, img_extension=img_extension)
train_dataset_jump = Reconstruction3DDataLoaderJump(train_folder, transforms.Compose([transforms.ToTensor()]),
                                                resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, jump=args.jump, return_normal_seq=args.pseudo_anomaly_jump > 0, img_extension=img_extension)

train_size = len(train_dataset)

train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
train_batch_jump = data.DataLoader(train_dataset_jump, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, drop_last=True)

# Report the training process
log_dir = os.path.join('./exp', args.dataset_type, exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'), 'a')
sys.stdout = f

torch.set_printoptions(profile="full")

loss_func_mse = nn.MSELoss(reduction='none')

if args.start_epoch < args.epochs:
    model = convAE()
    model = nn.DataParallel(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # resume
    if args.model_dir is not None:
        assert args.start_epoch > 0
        # Loading the trained model
        model_dict = torch.load(args.model_dir)
        model_weight = model_dict['model']
        model.load_state_dict(model_weight.state_dict())
        optimizer.load_state_dict(model_dict['optimizer'])
        model.cuda()

    # model.eval()
    for epoch in range(args.start_epoch, args.epochs):
        pseudolossepoch = 0
        lossepoch = 0
        pseudolosscounter = 0
        losscounter = 0

        for j, (imgs, imgsjump) in enumerate(zip(train_batch, train_batch_jump)):
            net_in = copy.deepcopy(imgs)
            net_in = net_in.cuda()

            jump_pseudo_stat = []
            cls_labels = []

            for b in range(args.batch_size):
                total_pseudo_prob = 0
                rand_number = np.random.rand()
                pseudo_bool = False

                # skip frame pseudo anomaly
                pseudo_anomaly_jump = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_jump
                total_pseudo_prob += args.pseudo_anomaly_jump
                if pseudo_anomaly_jump:
                    net_in[b] = imgsjump[0][b]
                    jump_pseudo_stat.append(True)
                    pseudo_bool = True
                else:
                    jump_pseudo_stat.append(False)

                if pseudo_bool:
                    cls_labels.append(0)
                else:
                    cls_labels.append(1)

            ########## TRAIN
            outputs = model(net_in)

            cls_labels = torch.Tensor(cls_labels).unsqueeze(1).cuda()

            loss_mse = loss_func_mse(outputs, net_in)

            modified_loss_mse = []
            for b in range(args.batch_size):
                if jump_pseudo_stat[b]:
                    modified_loss_mse.append(torch.mean(-loss_mse[b]))
                    pseudolossepoch += modified_loss_mse[-1].cpu().detach().item()
                    pseudolosscounter += 1

                else:  # no pseudo anomaly
                    modified_loss_mse.append(torch.mean(loss_mse[b]))
                    lossepoch += modified_loss_mse[-1].cpu().detach().item()
                    losscounter += 1

            assert len(modified_loss_mse) == loss_mse.size(0)
            stacked_loss_mse = torch.stack(modified_loss_mse)
            loss = torch.mean(stacked_loss_mse)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if j % 10 == 0 or args.print_all:
                print("epoch {:d} iter {:d}/{:d}".format(epoch, j, len(train_batch)))
                print('Loss: {:.6f}'.format(loss.item()))

        print('----------------------------------------')
        print('Epoch:', epoch)
        if pseudolosscounter != 0:
            print('PseudoMeanLoss: Reconstruction {:.9f}'.format(pseudolossepoch/pseudolosscounter))
        if losscounter != 0:
            print('MeanLoss: Reconstruction {:.9f}'.format(lossepoch/losscounter))

        # Save the model and the memory items
        model_dict = {
            'model': model,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(model_dict, os.path.join(log_dir, 'model_{:02d}.pth'.format(epoch)))

print('Training is finished')
sys.stdout = orig_stdout
f.close()



