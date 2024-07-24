# Image_segmention_Unet

This project is used as a test for project assessment. It is free for learning.  
Please pay attention to the path of file!

**U-Net++ has a better performance than normal U-Net**

## Prepare for the env
**PyTorch 2.0.0、Python 3.8(ubuntu20.04)、Cuda 11.8**  
```
pip install -r requirements.txt
```
## Dataset
The dataset we used in this project is **EndoVis 2018**. You could download it on the [website](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Downloads/) or contact me.  

## Data_pre_process
The detail of this part is showed in [data_pre.ipynb](https://github.com/Zh-Qr/Image_segmention_Unet/blob/main/data_pre.ipynb)  

## Train
The detail of this part is showed in [train.ipynb](https://github.com/Zh-Qr/Image_segmention_Unet/blob/main/train.ipynb)

## Val
The detail of this part is showed in [val.ipynb](https://github.com/Zh-Qr/Image_segmention_Unet/blob/main/val.ipynb)

## Vision of training process
The detail of this part is showed in [vision_process.ipynb](https://github.com/Zh-Qr/Image_segmention_Unet/blob/main/vision_process.ipynb)

## Result
The vision of training process is showed below:
![piture_train_process](https://github.com/Zh-Qr/Image_segmention_Unet/blob/main/picture/piture_train_process.png)

The vision of val process is showed below:
![piture_val_process](https://github.com/Zh-Qr/Image_segmention_Unet/blob/main/picture/piture_val_process.png)  


Part of the test results are showed below:  
![val1](https://github.com/Zh-Qr/Image_segmention_Unet/blob/main/picture/val_test1.png)  
![val2](https://github.com/Zh-Qr/Image_segmention_Unet/blob/main/picture/val_test2.png)  
![val3](https://github.com/Zh-Qr/Image_segmention_Unet/blob/main/picture/val_test3.png)
