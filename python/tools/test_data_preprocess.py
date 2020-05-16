import os
import shutil
import cv2
import copy
import SimpleITK as sitk
from PIL import Image
import fire
from matplotlib import pyplot as plt
import numpy as np
import csv


def modify_mask(base):
    xmin = 512
    ymin = 512
    xmax = 0
    ymax = 0
    xflag = False;
    yflag = False;
    for i in range(base.shape[0]):
        for j in range(base.shape[1]):
            if base[i][j] == 1:
                if i < xmin:
                    xmin = i
                if i > xmax:
                    xmax = i
                if j < ymin:
                    ymin = j
                if j > ymax:
                    ymax = j
    xmin = xmin - 30 if xmin - 30 >= 0 else 0
    ymin = ymin - 30 if ymin - 30 >= 0 else 0
    xmax = xmax + 30 if ymax + 30 <= 511 else 511
    ymax = ymax + 30 if ymax + 30 <= 511 else 511
    x_mean = int((xmax - xmin) / 2)
    mask_new = np.zeros((512, 512))
    for i in range(512):
        for j in range(512):
            if i > xmin and i < xmax and j > ymin and j < ymax:
                mask_new[i][j] = 255
            else:
                mask_new[i][j] = 0
    return mask_new, xmin, ymin, xmax, ymax



def seg_img(img, mask):
    cordi = np.where(mask == 255, img, -2000)  # mask 所在的保留原数据，其余的置为零
    return cordi



def get_circle(origin, large_ratio=1.2, small_ratio=0.8):
    seged_img = copy.deepcopy(origin)
    center_x0_list = np.where(seged_img == 1)[0]
    center_y0_list = np.where(seged_img == 1)[1]
    if len(center_x0_list) == 0:
        return seged_img
    try:

        cx0 = int(np.mean(center_x0_list))
        cy0 = int(np.mean(center_y0_list))
    except:
        print(center_x0_list)
        print("center Nan occured->x0")

    seged_img1 = cv2.resize(seged_img, (int(512 * 0.8), int(512 * 0.8)), interpolation=cv2.INTER_CUBIC)
    # plt.imshow(seged_img,cmap="gray")
    # 去掉浮点数转换为二值图像,resize 之后部分点的值变化了
    for i in range(seged_img1.shape[0]):
        for j in range(seged_img1.shape[1]):
            seged_img1[i][j] = 0 if seged_img1[i][j] == 0 else 255

    seged_img1 = seged_img1.astype(np.uint8)
    center_x1_list = np.where(seged_img1 == 255)[0]
    center_y1_list = np.where(seged_img1 == 255)[1]
    if len(center_x1_list) == 0:
        return seged_img
    try:
        cx1 = int(np.mean(center_x1_list))
        cy1 = int(np.mean(center_y1_list))
    except:
        print(len(center_x1_list))
        print(center_x1_list)
        print("center Nan occured->x1")

    seged_img2 = cv2.resize(seged_img, (int(512 * 1.5), int(512 * 1.5)), interpolation=cv2.INTER_CUBIC)
    for i in range(seged_img2.shape[0]):
        for j in range(seged_img2.shape[1]):
            seged_img2[i][j] = 0 if seged_img2[i][j] == 0 else 255

    plt.imshow(seged_img2, cmap="gray")
    # 去掉浮点数转换为二值图像
    center_x2_list = np.where(seged_img2 == 255)[0]
    center_y2_list = np.where(seged_img2 == 255)[1]
    cx2 = int(np.mean(center_x2_list))
    cy2 = int(np.mean(center_y2_list))
    cx2, cy2  # 放大图中心坐标
    x2, y2 = np.where(seged_img2 == 255)

    dx = cx0 - cx2
    dy = cy0 - cy2
    new_img = np.zeros_like(seged_img)

    for index, item in enumerate(x2):
        if x2[index] + dx < 512 and x2[index] + dx >= 0 and y2[index] + dy >= 0 and y2[index] + dy < 512:
            new_img[x2[index] + dx][y2[index] + dy] = 255

    # x1, y1 = np.where(seged_img1 == 255)

    dx = cx0 - cx1
    dy = cy0 - cy1
    #
    # for index, item in enumerate(x1):
    #     new_img[x1[index] + dx][y1[index] + dy] = 0

    return new_img


def get2dct(ct_root_path = "F:",terminal_path = "F:"):
    print(ct_root_path,terminal_path)
    im_file = ct_root_path + "\\IM\\"
    dcm_file = ct_root_path + "\\dcm\\"
    mask_reader = sitk.ImageSeriesReader()
    masks_filename = mask_reader.GetGDCMSeriesFileNames(im_file)
    mask_reader.SetFileNames(masks_filename)
    masks = mask_reader.Execute()
    masks_array = sitk.GetArrayFromImage(masks)
    dcm_reader = sitk.ImageSeriesReader()
    dcm_filename = dcm_reader.GetGDCMSeriesFileNames(dcm_file)
    dcm_reader.SetFileNames(dcm_filename)
    dcms = dcm_reader.Execute()
    dcms_array = sitk.GetArrayFromImage(dcms)


    for file_id in range(masks_array.shape[0]):
        file_name_mark = dcm_filename[file_id].split("/")[-1].split(".")[0]
        single_file_name = file_name_mark + "_" + "0" + "_" + "2"
        circle_mask = get_circle(masks_array[file_id], large_ratio=1.2, small_ratio=0.8)
        seged_img = seg_img(dcms_array[file_id], circle_mask)

        # seged_img = cv2.resize(seged_img,(128,128),interpolation=cv2.INTER_CUBIC)
        # circle_img = get_circle(seged_img)
        final_file_path = os.path.join(terminal_path, single_file_name)

        np.save(final_file_path + ".npy", seged_img)


if __name__ == "__main__":
    # terminal_path = "F:\\test_out\\"
    # if not os.path.exists(terminal_path):
    #     os.mkdir(terminal_path)
    fire.Fire(get2dct)
    # get2dct(ct_root_path="F:\\liver\\liverdata\\1",terminal_path=terminal_path)