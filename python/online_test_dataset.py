# online mode dataset


import torch as t
import numpy as np
import os
import time
import shutil
from config_2d import opt
from torchvision import transforms

import codecs


def gaussian_normalize(image_array):
    MEAN = np.mean(image_array)
    STD = np.std(image_array)
    image_result = (image_array - MEAN) / STD
    return image_result


# normal lize data to 0~1
def normalize(image_array):
    MAX_NUM = np.max(image_array)
    MIN_NUM = np.min(image_array)
    image_result = (image_array - MIN_NUM) / (MAX_NUM - MIN_NUM)
    return image_result


def flow_train_test(src_root="", train_target_root="", test_target_root="", train_head_list=[], test_head_list=[]):
    if os.path.exists(train_target_root):
        shutil.rmtree(train_target_root)
        os.mkdir(train_target_root)
    elif not os.path.exists(train_target_root):
        os.mkdir(train_target_root)
    if os.path.exists(test_target_root):
        shutil.rmtree(test_target_root)
        os.mkdir(test_target_root)
    elif not os.path.exists(test_target_root):
        os.mkdir(test_target_root)

    all_file_names = os.listdir(src_root)
    for file in all_file_names:
        if file.split("_")[0] in train_head_list:
            shutil.copyfile(os.path.join(src_root, file), os.path.join(train_target_root, file))
        elif file.split("_")[0] in test_head_list:
            shutil.copyfile(os.path.join(src_root, file), os.path.join(test_target_root, file))
        else:
            print("miss target file, filename:", file)


def read_train_test():
    train_test_file = codecs.open(opt.train_test_file, mode="r", encoding="utf8")
    file_array = train_test_file.readlines()
    train_list = file_array[1].strip().strip("[]").split(",")
    train_list = [x.strip().strip("'") for x in train_list]
    test_list = file_array[3].strip().strip("[]").split(",")
    test_list = [x.strip().strip("'") for x in test_list]
    return train_list, test_list


class dataset_2d():
    def __init__(self, data_path, train=False, shuffle=True, seed=0):

        list_filenames = os.listdir(data_path)
        self.train = train
        self.list_filenames = list_filenames
        self.circle_root_path = data_path
        self.user_dict = {}
        self.user_img_count = {}
        self.train_data_list = []
        self.test_data_list = []
        self.train_pos_neg_ratio = []
        self.test_pos_neg_ratio = []
        self.train_trans = transforms.Compose([
            # transforms.Resize(768),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(512),
        ])
        self.test_trans = transforms.Compose([
            # transforms.Resize(768),
            # transforms.CenterCrop(512),
        ])
        self.test_data_list = os.listdir(self.circle_root_path)

    def __getitem__(self, item):  # item is index

        image_path = os.path.join(self.circle_root_path, self.test_data_list[item])
        image = np.load(image_path)
        image = image.reshape(1, 512, 512)
        # image = gaussian_normalize(image)
        image = t.from_numpy(image).float()
        label = "2"  # online mode

        return image, label

    def __len__(self):
        return len(self.test_data_list)


if __name__ == "__main__":
    root_path = "test_data"
    dataset = dataset_2d(root_path, train=True)
    temp_file = dataset.__getitem__(1)[0]
    # read_train_test()
    pass
