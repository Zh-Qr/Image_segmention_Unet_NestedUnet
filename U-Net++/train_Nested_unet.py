import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adadelta, Nadam ,Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import  plot_model ,Sequence
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import tensorflow as tf
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization, Activation, Conv2DTranspose
from scipy.ndimage import morphology as mp
from PIL import Image,UnidentifiedImageError
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import shutil
import os
from glob import glob  # for getting list paths of image and labels
from random import choice,sample
from matplotlib import pyplot as plt
import cv2 # saving and loading images

print("环境准备完毕")

source_path = '../autodl-tmp/Endovis18/Train'
output_dir = os.path.join(source_path, 'images')

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def is_image_corrupted(filepath):
    """检查图像是否损坏。"""
    try:
        with Image.open(filepath) as img:
            img.verify()  # 检查文件是否损坏
        return False
    except (IOError, SyntaxError) as e:
        print(f'Corrupted image {filepath}: {e}')
        return True

for seq in os.listdir(source_path):
    if seq in ['pixeled_annotations_train', 'images']:
        continue  # Skip special directories
    seq_path = os.path.join(source_path, seq)
    img_folder_path = os.path.join(seq_path, "left_frames")
    if not os.path.isdir(img_folder_path):
        continue  # Skip if it's not a directory
    for f in os.listdir(img_folder_path):
        _, file_extension = os.path.splitext(f)
        if file_extension.lower() not in ['.jpg', '.png']:
            continue  # Skip non-image files
        file_path = os.path.join(img_folder_path, f)
        if is_image_corrupted(file_path):
            continue  # Skip copying if the image is corrupted
        output_file_path = os.path.join(output_dir, f"{seq}_{f}")
        # Copy file to the new location
        shutil.copy(file_path, output_file_path)
    print(f"Copied {seq} to {output_dir}")

    from PIL import Image
import os

def verify_and_delete_images(directory):
    files_in_directory = os.listdir(directory)
    for file in files_in_directory:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(directory, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify that this is an image
            except (IOError, SyntaxError) as e:
                print(f"Deleting corrupt file: {file_path} ({e})")
                os.remove(file_path)  # Delete corrupt file

# Example usage
train_img_dir = '../autodl-tmp/Endovis18/Train/images/'
train_mask_dir = '../autodl-tmp/Endovis18/Train/pixeled_annotations_train/'
verify_and_delete_images(train_img_dir)
verify_and_delete_images(train_mask_dir)

train_img_dir = '../autodl-tmp/Endovis18/Train/images/' 
train_mask_dir = '../autodl-tmp/Endovis18/Train/pixeled_annotations_train/'
def clean_directory(directory):
    valid_extensions = ['.jpg', '.png']
    files_in_directory = os.listdir(directory)
    filtered_files = [file for file in files_in_directory if not any(file.lower().endswith(ext) for ext in valid_extensions)]
    for file in filtered_files:
        file_path = os.path.join(directory, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted '{file}'")
            else:
                print(f"Skipped '{file}', it is not a file.")
        except Exception as e:
            print(f"Error deleting file '{file}': {e}")
            
print("Cleaning image directory:")
clean_directory(train_img_dir)
print("Cleaning mask directory:")
clean_directory(train_mask_dir)
         
checkpoints_dir = os.path.join(train_mask_dir, '.ipynb_checkpoints')
# 检查是否存在这个文件夹
if os.path.exists(checkpoints_dir) and os.path.isdir(checkpoints_dir):
    shutil.rmtree(checkpoints_dir)  # 删除这个文件夹
    print(".ipynb_checkpoints has been removed")
else:
    print("No .ipynb_checkpoints directory found")
    
    
train_imgs = os.listdir(train_img_dir)
train_masks = os.listdir(train_mask_dir)
train_imgs= sorted([ i for i in train_imgs ])
train_masks= sorted([ i for i in train_masks ])
print(len(train_imgs))
print(len(train_masks))

print("Number of images:", len(train_imgs))
print("Number of masks:", len(train_masks))

if len(train_imgs) != len(train_masks):
    print("The number of images and masks are not equal.")

    # 使用文件名，因为列表直接包含文件名
    img_set = set(train_imgs)
    mask_set = set(train_masks)

    extra_imgs = img_set - mask_set
    extra_masks = mask_set - img_set

    print("Extra images:", extra_imgs)
    print("Extra masks:", extra_masks)

    # 从图像和掩码列表中移除多余的条目
    train_imgs = [img for img in train_imgs if img not in extra_imgs]
    train_masks = [mask for mask in train_masks if mask not in extra_masks]

    print("Updated number of images:", len(train_imgs))
    print("Updated number of masks:", len(train_masks))

    
from sklearn.model_selection import train_test_split

val_img_dir =  train_img_dir
val_mask_dir = train_mask_dir


train_imgs,val_imgs,train_masks,val_masks =  train_test_split(train_imgs, train_masks, test_size=0.13, random_state=42)


print(len(train_masks))
print(len(val_masks))

# 数据集加载
num_batch_size = 16
class DataGenerator(Sequence):
    'Generates data for Keras'
    
    def __init__(self, images,image_dir,labels,label_dir ,batch_size, dim=(224,224,3) ,shuffle=True):
        'Initialization'
        self.dim = dim
        self.images = images
        self.image_dir = image_dir
        self.labels = labels
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_imgs = list()
        batch_labels = list()

        # Generate data
        for i in list_IDs_temp:
#             degree=np.random.random() * 360
            # Store sample
            img = load_img(self.image_dir + self.images[i] ,target_size=self.dim)
            img = img_to_array(img)/255.
#             img = ndimage.rotate(img, degree)
#             print(img)
            batch_imgs.append(img)
           # Store class
            label = load_img(self.label_dir + self.labels[i] ,target_size=self.dim)
            label = img_to_array(label)[:,:,0]
            label = label != 0
            label = mp.binary_erosion(mp.binary_erosion(label))
            label = mp.binary_dilation(mp.binary_dilation(mp.binary_dilation(label)))
            label = np.expand_dims((label)*1 , axis=2)
            batch_labels.append(label)
            
        return np.array(batch_imgs,dtype = np.float32 ) ,np.array(batch_labels , dtype = np.float32 )
    
train_generator = DataGenerator(train_imgs,train_img_dir,train_masks,train_mask_dir,batch_size=num_batch_size, dim=(224,224,3) ,shuffle=True)
train_steps = train_generator.__len__()

val_generator = DataGenerator(val_imgs,val_img_dir,val_masks,val_mask_dir,batch_size=num_batch_size, dim=(224,224,3) ,shuffle=True)
val_steps = val_generator.__len__()

# 搭建模型
def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def upsample_concat_block(input_tensor, skip_tensor, num_filters):
    x = UpSampling2D((2, 2))(input_tensor)
    x = concatenate([x, skip_tensor])
    x = conv_block(x, num_filters)
    return x

def NestedUNet(input_shape, num_filters=64, dropout_rate=0.5):
    inputs = Input(input_shape)

    # Encoder (downsampling)
    conv1 = conv_block(inputs, num_filters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, num_filters*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, num_filters*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, num_filters*8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

    # Bottleneck
    conv5 = conv_block(pool4, num_filters*16)
    conv5 = Dropout(dropout_rate)(conv5)

    # Decoder (upsampling)
    up6 = upsample_concat_block(conv5, conv4, num_filters*8)
    up6 = Dropout(dropout_rate)(up6)

    up7 = upsample_concat_block(up6, conv3, num_filters*4)
    up7 = Dropout(dropout_rate)(up7)

    up8 = upsample_concat_block(up7, conv2, num_filters*2)

    up9 = upsample_concat_block(up8, conv1, num_filters)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(up9)

    model = Model(inputs=inputs, outputs=outputs, name='NestedUNet')
    return model

# 实例化
model = NestedUNet(input_shape=(256, 256, 3), num_filters=64)

# 设置评价指标
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# 网络编译
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, iou, recall, precision])
callbacks = []

# 模型训练
num_epoch = 100
results = model.fit(train_generator, steps_per_epoch=train_steps,epochs=num_epoch,callbacks=callbacks,validation_data=val_generator,validation_steps=val_steps)

# 模型保存
history_data = results.history
df = pd.DataFrame(history_data)
df.to_csv('result/training_history.csv')
# 保存整个模型，包括结构、权重和训练配置
model.save('result/complete_model.h5')
# 保存模型的权重
model.save_weights('result/complete_model_weights_only.h5')