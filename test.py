# Author : Hellcat
# Time   : 2018/12/6

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
np.set_printoptions(threshold=np.inf)

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
"""

import numpy as np
import pandas as pd

with open(r'..\fashionAI_keypoints_test\test.csv') as f:
    csv_data = pd.read_csv(f)

import os
import utils
import skimage
import model as modellib
from config import Config

PART_INDEX = {'blouse': [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14],
              'outwear': [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
              'dress': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18],
              'skirt': [15, 16, 17, 18],
              'trousers': [15, 16, 19, 20, 21, 22, 23]}
PART_STR = ['neckline_left', 'neckline_right',
            'center_front',
            'shoulder_left', 'shoulder_right',
            'armpit_left', 'armpit_right',
            'waistline_left', 'waistline_right',
            'cuff_left_in', 'cuff_left_out',
            'cuff_right_in', 'cuff_right_out',
            'top_hem_left', 'top_hem_right',
            'waistband_left', 'waistband_right',
            'hemline_left', 'hemline_right',
            'crotch',
            'bottom_left_in', 'bottom_left_out',
            'bottom_right_in', 'bottom_right_out']
IMAGE_CATEGORY = ['blouse', 'outwear', 'dress', 'skirt', 'trousers'][3]


class FIConfig(Config):
    """
    Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = IMAGE_CATEGORY

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_KEYPOINTS = len(PART_INDEX[IMAGE_CATEGORY])  # 更改当前训练关键点数目
    KEYPOINT_MASK_SHAPE = [56, 56]

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1

    RPN_TRAIN_ANCHORS_PER_IMAGE = 100
    VALIDATION_STPES = 100
    STEPS_PER_EPOCH = 1000
    MINI_MASK_SHAPE = (56, 56)
    KEYPOINT_MASK_POOL_SIZE = 7

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    WEIGHT_LOSS = True
    KEYPOINT_THRESHOLD = 0.005


inference_config = FIConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir='./logs_{}'.format(IMAGE_CATEGORY))

# Get path to saved weights
# print(model.find_last())
model.load_weights(model.find_last()[1], by_name=True)

"""测试单张"""
# path = os.path.join('../fashionAI_keypoints_test', 'Images/skirt/9788a3e3b5f52727824ff614c2505802.jpg')
# image = skimage.io.imread(path)
# results = model.detect_keypoint([image], verbose=0)
# import matplotlib.pyplot as plt
# plt.imshow(image)
# plt.show()

"""多张测试"""
import math

col = ['image_id', 'image_category'] + PART_STR                                        # 列标签
images_path = csv_data[csv_data.image_category.isin([IMAGE_CATEGORY])].image_id        # 路径
kps = np.empty([images_path.shape[0], 26]).astype(str)  # 存储容器

batch_size = inference_config.GPU_COUNT * inference_config.IMAGES_PER_GPU     # model calss硬性要求bs的计算方法
steps = math.ceil(images_path.index[-1] - images_path.index[0] / batch_size)  # 循环次数

print("共有 {} 张图片等待处理... ...".format(images_path.index[-1] - images_path.index[0]))
for step in range(int(steps)):
    start_index = step * batch_size
    if start_index % 100 == 0 and start_index != 0:
        print("  正在生成第 {} 张关键点结果... ...".format(start_index))
        print(kps[start_index-1])
    if step != steps-1:
        paths = [os.path.join('../fashionAI_keypoints_test', path)
                  for path in images_path[images_path.index[start_index:start_index+batch_size]]]
    else:
        paths = [os.path.join('../fashionAI_keypoints_test', path)
                  for path in images_path[images_path.index[start_index:]]]
    images = [skimage.io.imread(path) for path in paths]
    logits = model.detect_keypoint(images, verbose=0)

#     results = [['_'.join(list(p.astype(str)))
#                 for p in logits[i]['keypoints'][0]] for i in range(batch_size)]

    results = []
    for i in range(batch_size):
        try:
            for p in logits[i]['keypoints'][0]:
                results.extend(['_'.join(list(p.astype(str)))])
        except IndexError as e:
            print("第 {} 轮图片出现异常".format(step))
            result = ['-1_-1_-1' for i in range(len(PART_INDEX[IMAGE_CATEGORY]))]

    image_id = [path.split('/')[-1] for path in paths]
    image_category = [path.split('/')[-2] for path in paths]

    for i, (id_, category) in enumerate(zip(image_id, image_category)):
        info_arr = np.array([id_, category] + ['-1_-1_-1' for j in range(24)])
        info_arr[np.array(PART_INDEX[IMAGE_CATEGORY]) + 2] = np.array(results)
        kps[i+start_index] = info_arr

kps_df = pd.DataFrame(kps, columns=col)
kps_df.to_csv(r'./{}.csv'.format(IMAGE_CATEGORY))

