# Official PyTorch implementation of "Synthetic Temporal Anomaly Guided End-to-End Video Anomaly Detection"
This is the implementation of the paper "Synthetic Temporal Anomaly Guided End-to-End Video Anomaly Detection" (ICCV Workshops 2021: RSL-CV).

[Paper](https://openaccess.thecvf.com/content/ICCV2021W/RSLCV/html/Astrid_Synthetic_Temporal_Anomaly_Guided_End-to-End_Video_Anomaly_Detection_ICCVW_2021_paper.html) || [Presentation Video](https://youtu.be/om7sY1hc3Dw)

## Dependencies
* Python 3.6
* PyTorch = 1.7.0 
* Numpy
* Sklearn

## Datasets
* USCD Ped2 [[dataset](https://drive.google.com/file/d/1w1yNBVonKDAp8uxw3idQkUr-a9Gj8yu1/view?usp=sharing)]
* CUHK Avenue [[dataset](https://drive.google.com/file/d/1q3NBWICMfBPHWQexceKfNZBgUoKzHL-i/view?usp=sharing)]
* ShanghaiTech [[dataset](https://drive.google.com/file/d/1rE1AM11GARgGKf4tXb2fSqhn_sX46WKn/view?usp=sharing)]

Download the datasets into ``dataset`` folder, like ``./dataset/ped2/``, ``./dataset/avenue/``, ``./dataset/shanghai/``

## Training
```bash
git clone https://github.com/aseuteurideu/STEAL
```

* Training baseline
```bash
python train.py --dataset_type ped2
```

* Training STEAL Net
```bash
python train.py --dataset_type ped2 --pseudo_anomaly_jump 0.01 --jump 2 3 4 5
```

Select --dataset_type from ped2, avenue, or shanghai.

For more details, check train.py


## Pre-trained model

| Model           | Dataset       | AUC           | Weight        |
| -------------- | ------------- | ------------- | ------------- | 
| Baseline | Ped2          |   92.5%       | [ [drive](https://drive.google.com/file/d/1QN02heGYVgOgivmfMGkPjR0GHwqT4wRg/view?usp=share_link) ] |
| Baseline | Avenue        |   81.5%       | [ [drive](https://drive.google.com/file/d/1Xqh5hPffEnEsoBIp_tHuHEFziapShl85/view?usp=share_link) ] |
| Baseline | ShanghaiTech  |   71.3%       | [ [drive](https://drive.google.com/file/d/1Bh2E2pJHkf-uTkELTgKGuezqYg1msLen/view?usp=share_link) ] |
| STEAL Net  | Ped2          |   98.4%       | [ [drive](https://drive.google.com/file/d/12bZN6zueV77pf7qs72N965K8R-1TpTKk/view?usp=sharing) ] |
| STEAL Net  | Avenue        |   87.1%       | [ [drive](https://drive.google.com/file/d/1oJMekYqQBeKLBvmP4CUnrpbNbr-3GQI1/view?usp=sharing) ] |
| STEAL Net  | ShanghaiTech  |   73.7%       | [ [drive](https://drive.google.com/file/d/1AyJZL6lvSLKS1rF7QTczrjZSRRemRaxN/view?usp=share_link) ] |

## Evaluation
* Test the model
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth
```
* Test the model and save result image
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth --img_dir folder_path_to_save_image_results
```
* Test the model and generate demonstration video frames
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth --vid_dir folder_path_to_save_video_results
```
Then compile the frames into video. For example, to compile the first video in ubuntu:
```bash
ffmpeg -framerate 10 -i frame_00_%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p video_00.mp4
```


## Bibtex
```
@InProceedings{Astrid_2021_ICCV,
    author    = {Astrid, Marcella and Zaheer, Muhammad Zaigham and Lee, Seung-Ik},
    title     = {Synthetic Temporal Anomaly Guided End-to-End Video Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2021},
    pages     = {207-214}
}
```

## Acknowledgement
The code is built on top of code provided by Park et al. [ [github](https://github.com/cvlab-yonsei/MNAD) ] and Gong et al. [ [github](https://github.com/donggong1/memae-anomaly-detection) ]
