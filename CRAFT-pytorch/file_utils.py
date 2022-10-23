# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        ## box file
        tmp_crop_img_list = []

        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):
                #print(box)

                min_x = img.shape[1]-1
                max_x = 0
                min_y = img.shape[0]-1
                max_y = 0

                for x, y in box:
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y

                #print(min_x, max_x, min_y, max_y)

                if min_x < 0:
                    min_x = 0
                if max_x > img.shape[1]:
                    max_x = img.shape[1]-1
                if min_y < 0:
                    min_y = 0
                if max_y > img.shape[0]:
                    max_y = img.shape[0]-1

                #print(min_x, max_x, min_y, max_y)

                crop_img = img[int(min_y):int(max_y), int(min_x):int(max_x)]

                tmp_crop_img_list.append([crop_img, min_x, min_y])

        # 이미지 순서 sort (세로 = 위부터, 가로 = 왼쪽부터)
        if tmp_crop_img_list != 1:
            if img.shape[1] > img.shape[0]:
                tmp_crop_img_list.sort(key=lambda x: x[1])
            else:
                tmp_crop_img_list.sort(key=lambda x: x[2])

        for i, crop_img_list in enumerate(tmp_crop_img_list):
            crop_img, _, _ = crop_img_list
            res_img_file = dirname + filename + f'_{i}.jpg'
            # Save result image
            cv2.imwrite(res_img_file, crop_img)

        # txt 파일 제거
        if os.path.exists(res_file):
            os.remove(res_file)

