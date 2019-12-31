#!usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
数据增强流程：
    读取候选背景图片列表  ——>  读取图片   ——> 读取标注信息 ——> 截取区域数据 ——> 确定缩放系数 ——>确定嵌入坐标 ——>保持图片和xml文件
可配置参数：
    除嵌入外的其他增强方式
    每个框合成的增强样本数量
    是否允许多目标混合
    是否允许多类别混合
    最大/最小缩放系数
文件名秒格式：
    aug + box_index + aug_index + filename
"""
import os
import numpy as np
from random import choice, random, shuffle

from PIL import Image
from tqdm import tqdm
from xl_tool.xl_io import file_scanning

from image_augmentation.config import IMAGE_FORMAT
from image_augmentation.general.annonation import get_boximgs
from image_augmentation.general.preprocessing import blending_one_image
from image_augmentation.transform.voc import Text2XML


def get_background_images(path):
    files = file_scanning(path, full_path=True, file_format=IMAGE_FORMAT)
    return files


def aug_images(cat_name, image_files, xml_files, background_path, cat_aug_path, xml_folder, xml_source,
               x_proportion=0.4, y_proportion=0.5, boxes_per_image=1, max_num=1000, min_box_edge=100):
    background_images = get_background_images(background_path)
    im_xmls = list(zip(image_files, xml_files))
    shuffle(im_xmls)
    pbar = tqdm(im_xmls)
    count = 0
    for image_file, xml_file in pbar:
        if count >= max_num:
            continue
        base_name = os.path.basename(image_file).split(".")[0]
        boximgs = get_boximgs(image_file, xml_file)
        boximgs = [boximg_info for boximg_info in boximgs if boximg_info["name"] == cat_name]
        shuffle(boximgs)
        box_count = 0
        for i, boximg_info in enumerate(boximgs, 1):
            if box_count > boxes_per_image:
                break
            if sum(boximg_info["img"].size) < min_box_edge:
                # print("图片过小，不适合增强！",boximg_info["img"].size)
                continue
            background_img_array = np.array(Image.open(choice(background_images)))
            aug_image, coordinate = blending_one_image("",background_img_array=background_img_array,
                                                       blending_img_array=np.array(boximg_info["img"]),
                                                              x_proportion=x_proportion,
                                                              y_proportion=y_proportion)
            objects_info = [[boximg_info["name"]] + coordinate]
            text2xml = Text2XML()
            boximg_file = "aug_{}_{}_{}.jpg".format(i, 0, base_name)
            xml = text2xml.get_xml(xml_folder, boximg_file, boximg_file, xml_source, aug_image.size, objects_info)
            boxxml_file = "aug_{}_{}_{}.xml".format(i, 0, base_name)
            aug_image.save(cat_aug_path + "/" + boximg_file)
            with open(cat_aug_path + "/" + boxxml_file, "w") as f:
                f.write(xml)
            count += 1
            box_count += 1

        pbar.set_description("图片增强进度：")


def get_xml_image(image_path, xml_path):
    valid_images = []
    valid_xmls = []
    image_files = file_scanning(image_path, file_format=IMAGE_FORMAT, full_path=True)
    for image_file in image_files:
        xml_file = xml_path + "/" + os.path.basename(image_file).split(".")[0] + ".xml"
        if os.path.exists(xml_file):
            valid_images.append(image_file)
            valid_xmls.append(xml_file)
    return valid_images, valid_xmls


def get_propotions(cat):
    big_propotion_cats = {"pizza"}
    small_propotion_cats = {"hamburger"}
    if cat in big_propotion_cats:
        return 0.8, 0.8
    elif cat in small_propotion_cats:
        return 0.4, 0.5
    else:
        return 0.6, 0.6


def main():
    data_path = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建" \
                r"数据集\3_公开数据集抽取\原始标注文件"
    aug_path = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建" \
               r"数据集\3_公开数据集抽取\增强文件"
    background_path = r"E:\Programming\Python\8_Ganlanz\food_recognition\data" \
                      r"set\自建数据集\4_背景底图"
    folder = r"Food2019"
    source = 'FoodDection'
    dirs = os.listdir(data_path)
    for dir_ in dirs:
        x_proportion, y_proportion = get_propotions(dir_)
        cat_path = data_path + "/" + dir_
        cat_aug_path = aug_path + "/" + dir_
        os.makedirs(cat_aug_path, exist_ok=True)
        image_files, xml_files = get_xml_image(cat_path, cat_path)
        print("类别：{}\t有效标注文件数量：{}".format(dir_, len(image_files)))
        aug_images(dir_, image_files, xml_files, background_path, cat_aug_path, folder, source,
                   x_proportion=x_proportion, y_proportion=y_proportion)


if __name__ == '__main__':
    main()
