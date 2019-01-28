# Author : Hellcat
# Time   : 2019/1/25

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
import os
import utils
import skimage
import numpy as np
import pandas as pd
import model as modellib
from config import Config


with open(r'..\fashionAI_keypoints_test\test.csv') as f:
    csv_data = pd.read_csv(f)

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
IMAGE_CATEGORY = ['blouse', 'outwear', 'dress', 'skirt', 'trousers'][4]


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


def detect_image(model, img_path):
    img_path.replace('\\', '/')
    image = skimage.io.imread(img_path)
    res = model.detect_keypoint([image], verbose=0)
    # 注意MRCNN可能检测出多个目标，但实际上每张图只有一个目标
    res = ['_'.join(list(p.astype(str))) for p in res[0]['keypoints'][0]]

    image_id = img_path.split('/')[-1]
    image_category = img_path.split('/')[-2]

    res_arr = np.array([image_id, image_category] + ['-1_-1_-1' for i in range(24)])
    res_arr[np.array(PART_INDEX[IMAGE_CATEGORY]) + 2] = np.array(res)
    return res_arr


if __name__ == "__main__":
    inference_config = FIConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir='./logs_{}'.format(IMAGE_CATEGORY))

    # Get path to saved weights
    # print(model.find_last())
    model.load_weights(model.find_last()[1], by_name=True)
    model.keras_model.summary()

    col = ['image_id', 'image_category'] + PART_STR
    images_path = csv_data[csv_data.image_category.isin([IMAGE_CATEGORY])].image_id
    kps = np.empty([images_path.shape[0], 26]).astype(str)
    for i, path in enumerate(images_path):
        try:
            kps[i] = detect_image(model, os.path.join('../fashionAI_keypoints_test', path))
        except IndexError as e:
            print("图片 {} 没有检测到任何目标……".format(path))
        if i % 100 == 0:
            print('已经处理 {} 张图片……'.format(i))

    kps_df = pd.DataFrame(kps, columns=col)
    kps_df.to_csv(r'./{}.csv'.format(IMAGE_CATEGORY))

