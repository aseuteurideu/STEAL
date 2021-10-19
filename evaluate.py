import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from model.utils import Reconstruction3DDataLoader
from model.autoencoder import *
from utils import *
import glob
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

import argparse
import time

parser = argparse.ArgumentParser(description="STEAL Net")
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model')

parser.add_argument('--img_dir', type=str, default=None, help='save image file')

parser.add_argument('--print_score', action='store_true', help='print score')
parser.add_argument('--vid_dir', type=str, default=None, help='save video frames file')
parser.add_argument('--print_time', action='store_true', help='print forward time')

args = parser.parse_args()

if args.img_dir is not None:
    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)

if args.vid_dir is not None:
    if not os.path.exists(args.vid_dir):
        os.makedirs(args.vid_dir)

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
model = convAE()
model = nn.DataParallel(model)

model_dict = torch.load(args.model_dir)
try:
    model_weight = model_dict['model']
    model.load_state_dict(model_weight.state_dict())
except KeyError:
    model.load_state_dict(model_dict['model_statedict'])
model.cuda()
labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')

# Loading dataset
test_folder = os.path.join(args.dataset_path, args.dataset_type, 'testing', 'frames')
img_extension = '.tif' if args.dataset_type == 'ped1' else '.jpg'
test_dataset = Reconstruction3DDataLoader(test_folder, transforms.Compose([transforms.ToTensor(),]),
                                          resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, img_extension=img_extension)
test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)


videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*/')))
for video in videos_list:
    video_name = video.split('/')[-2]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + img_extension))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-2]
    labels_list = np.append(labels_list, labels[0][8+label_length:videos[video_name]['length']+label_length-7])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-2]]['length']

model.eval()

tic = time.time()
for k,(imgs) in enumerate(test_batch):

    if k == label_length-15*(video_num+1):
        video_num += 1
        label_length += videos[videos_list[video_num].split('/')[-2]]['length']

    imgs = Variable(imgs).cuda()
    with torch.no_grad():
        outputs = model(imgs)
        loss_mse = loss_func_mse(outputs[0, :, 8], imgs[0, :, 8])

    loss_pixel = torch.mean(loss_mse)
    mse_imgs = loss_pixel.item()

    psnr_list[videos_list[video_num].split('/')[-2]].append(psnr(mse_imgs))


    if args.img_dir is not None or args.vid_dir is not None:
        output = (outputs[0,:,8].cpu().detach().numpy() + 1) * 127.5
        output = output.transpose(1,2,0).astype(dtype=np.uint8)

        if args.img_dir is not None:
            cv2.imwrite(os.path.join(args.img_dir, '{:04d}.jpg').format(k), output)
        if args.vid_dir is not None:
            cv2.imwrite(os.path.join(args.vid_dir, 'out_{:04d}.png').format(k), output)

        if args.vid_dir is not None:
            saveimgs = (imgs[0,:,8].cpu().detach().numpy() + 1) * 127.5
            saveimgs = saveimgs.transpose(1,2,0).astype(dtype=np.uint8)
            cv2.imwrite(os.path.join(args.vid_dir, 'GT_{:04d}.png').format(k), saveimgs)

        mseimgs = (loss_func_mse(outputs[0,:,8], imgs[0,:,8])[0].cpu().detach().numpy())

        mseimgs = mseimgs[:,:,np.newaxis]
        mseimgs = (mseimgs - np.min(mseimgs)) / (np.max(mseimgs)-np.min(mseimgs))
        mseimgs = mseimgs * 255
        mseimgs = mseimgs.astype(dtype=np.uint8)
        color_mseimgs = cv2.applyColorMap(mseimgs, cv2.COLORMAP_JET)
        if args.img_dir is not None:
            cv2.imwrite(os.path.join(args.img_dir, 'MSE_{:04d}.jpg').format(k), color_mseimgs)
        if args.vid_dir is not None:
            cv2.imwrite(os.path.join(args.vid_dir, 'MSE_{:04d}.png').format(k), color_mseimgs)

toc = time.time()
if args.print_time:
    time_elapsed = (toc-tic)/len(test_batch)
    print('time: ', time_elapsed)
    print('fps: ', 1/time_elapsed)


# Measuring the abnormality score (S) and the AUC
anomaly_score_total_list = []
vid_idx = []
for vi, video in enumerate(sorted(videos_list)):
    video_name = video.split('/')[-2]
    score = anomaly_score_list(psnr_list[video_name])
    anomaly_score_total_list += score
    vid_idx += [vi for _ in range(len(score))]

anomaly_score_total_list = np.asarray(anomaly_score_total_list)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))
if args.print_score:
    print('vididx,frame,anomaly_score,anomaly_label')
    for a in range(len(anomaly_score_total_list)):
        print(str(vid_idx[a]), ',', str(a), ',', 1-anomaly_score_total_list[a], ',', labels_list[a])

if args.vid_dir is not None:
    a = 0
    vids_len = []
    while a < len(vid_idx):
        start_a = a
        cur_vid_idx = vid_idx[a]
        num_frames = 0
        while vid_idx[a] == cur_vid_idx:
            num_frames += 1
            a += 1
            if a >= len(vid_idx):
                break
        vids_len.append(num_frames)

    a = 0
    while a < len(vid_idx):
        start_a = a
        atemp = a
        cur_vid_idx = vid_idx[a]
        vid_len = vids_len[cur_vid_idx]
        # rectangle position
        idx = 0
        rect_start = []
        rect_end = []
        anom_status = False
        while vid_idx[atemp] == cur_vid_idx:
            if not anom_status:
                if labels_list[atemp] == 1:
                    anom_status = True
                    rect_start.append(idx)
            else:
                if labels_list[atemp] == 0:
                    anom_status = False
                    rect_end.append(idx)

            idx += 1
            atemp += 1
            if atemp >= len(vid_idx):
                break
        if anom_status:
            rect_end.append(idx - 1)

        while vid_idx[a] == cur_vid_idx:
            # GT
            imggt = cv2.imread(os.path.join(args.vid_dir, 'GT_{:04d}.png').format(a))[:,:,[2,1,0]]
            plt.axis('off')
            plt.subplot(231)
            plt.title('Frame', fontsize='small')
            plt.imshow(imggt)

            # Recon
            imgout = cv2.imread(os.path.join(args.vid_dir, 'out_{:04d}.png').format(a))[:,:,[2,1,0]]
            plt.axis('off')
            plt.subplot(232)
            plt.title('Reconstruction', fontsize='small')
            plt.axis('off')
            plt.imshow(imgout)

            # MSE
            imgmse = mpimg.imread(os.path.join(args.vid_dir, 'MSE_{:04d}.png').format(a))
            plt.subplot(233)
            plt.title('Reconstruction Error', fontsize='small')
            plt.axis('off')
            plt.imshow(imgmse)

            # anomaly score plot
            plt.subplot(212)
            plt.plot(range(a-start_a+1), 1-anomaly_score_total_list[start_a:a+1], label='prediction', color='blue')
            plt.xlim(0, vid_len-1)
            plt.xticks(fontsize='x-small')
            plt.xlabel('Frames', fontsize='x-small')
            plt.ylim(-0.01, 1.01)
            plt.ylabel('Anomaly Score', fontsize='x-small')
            plt.yticks(fontsize='x-small')
            plt.title('Anomaly Score Over Time')
            for rs, re in zip(rect_start, rect_end):
                currentAxis = plt.gca()
                currentAxis.add_patch(Rectangle((rs, -0.01), re-rs, 1.02, facecolor="pink"))

            plt.savefig(os.path.join(args.vid_dir, 'frame_{:02d}_{:04d}.png').format(cur_vid_idx, a-start_a), dpi=300)
            plt.close()

            a += 1
            if a >= len(vid_idx):
                break

print('The result of ', args.dataset_type)
print('AUC: ', accuracy*100, '%')
