#!usr/bin/env python3
# -*- coding: UTF-8 -*-
# !usr/bin/env python3
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
import re

import numpy as np
from random import choice, shuffle

from PIL import Image
from tqdm import tqdm
from xl_tool.xl_io import file_scanning, read_json
# from xl_tool.xl_io import l

from image_augmentation.general.config import IMAGE_FORMAT
from image_augmentation.general.annonation import get_boximgs
from image_augmentation.general.preprocessing import blending_one_image, blending_images, linear_contrast, grey_world, \
    affine_with_rotate_scale
from image_augmentation.transform.voc import Text2XML

aug_dict = {"contrast": linear_contrast, "grey": grey_world, "affine": affine_with_rotate_scale}


def get_background_images(path):
    files = file_scanning(path, full_path=True, file_format=IMAGE_FORMAT)
    return files


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


def aug_images_from_xml(cat_name, image_files, xml_files, background_path, cat_aug_path, xml_folder,
                        xml_source, x_proportion=0.4, y_proportion=0.5, boxes_per_image=1, max_num=1000,
                        min_box_edge=100):
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
            aug_image, coordinate = blending_one_image("", background_img_array=background_img_array,
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


def aug_images_from_image(blending_method, image_files, background_path, cat_aug_path, xml_folder, xml_source,
                          cat_name, target_number=None, x_proportion=0.4, y_proportion=0.5, max_num=10000,
                          simple_augs=("contrast", "grey", "affine", None), min_size_sum=200):
    background_images = get_background_images(background_path)
    if min_size_sum:
        image_files = [file for file in image_files if sum(Image.open(file).size)>min_size_sum]
    pbar = tqdm(image_files) if not target_number else tqdm([choice(image_files) for _ in range(target_number)])
    count = 0
    for image_file in pbar:
        if count >= max_num:
            continue
        base_name = os.path.basename(image_file).split(".")[0]
        base_name = cat_name.replace(" ", "_") if re.search(r"[^a-zA-Z0-9_\-.]", base_name) else base_name
        background_img_array = np.array(Image.open(choice(background_images)))
        if sum(background_img_array.shape[:2]) < 200:
            continue
        save_img = f"{cat_aug_path}/aug_{1}_{count}_{base_name}.jpg"
        aug_image, coordinate = blending_one_image(blending_method, background_img_array=background_img_array,
                                                   blending_img_array=np.array(Image.open(image_file)),
                                                   x_proportion=x_proportion,
                                                   y_proportion=y_proportion, save_img=save_img)
        if simple_augs:
            aug_method = choice(simple_augs)
            if aug_method:
                image_array = np.array(Image.open(save_img))
                Image.fromarray(aug_dict[aug_method](image_array)).save(save_img)
        text2xml = Text2XML()
        objects_info = [[cat_name] + coordinate]
        boximg_file = f"aug_{1}_{count}_{base_name}.jpg"
        xml = text2xml.get_xml(xml_folder, boximg_file, boximg_file, xml_source, aug_image.size, objects_info)
        boxxml_file = f"aug_{1}_{count}_{base_name}.xml"
        aug_image.save(cat_aug_path + "/" + boximg_file)
        with open(cat_aug_path + "/" + boxxml_file, "w") as f:
            f.write(xml)
        count += 1
        pbar.set_description("图片增强进度：")


def aug_images_mul_object(object_path, background_config_file, save_path, target_number=5000,
                          distribute=(0.05, 0.35, 0.4, 0.15)):
    """
    流程：选择目标
    """
    folder = r"Food2019"
    source = 'FoodDection'
    cats = os.listdir(object_path)
    cats.remove("unknown")
    background_configs = read_json(background_config_file)
    # background_config = x
    cats_files = {cat: file_scanning(f"{object_path}/{cat}", file_format=IMAGE_FORMAT) for cat in cats}
    count = 0
    cats_count = {cat: 0 for cat in cats}
    while count < target_number:
        ratio = count / target_number
        if ratio <= distribute[0]:
            sp_dis = 1
            number = 1
        elif distribute[0] < ratio <= distribute[1]:
            sp_dis = choice([1, 2])
            number = 2
        elif distribute[1] < ratio <= distribute[2]:
            sp_dis = choice([1, 2])
            number = 3
        else:
            sp_dis = 1
            number = 4
        choose_cats = [choice(cats) for _ in range(number)]
        for c in choose_cats:
            cats_count[c] += 1
        choose_files = [choice(cats_files[cat]) for cat in choose_cats]
        background_config = choice(background_configs)
        background_file, region = background_config["file"], background_config['region']
        save_file = f"{save_path}/aug_{number}_{count}.jpg"
        background_img, image_sizes, positions = blending_images(background_file, choose_files, sp_dis, save_file,
                                                                 region)
        objects_info = [[cat, ] + list(positions[i]) for i, cat in enumerate(choose_cats)]
        text2xml = Text2XML()
        boximg_file = f"aug_{number}_{count}.jpg"
        xml = text2xml.get_xml(folder, boximg_file, boximg_file, source, background_img.size, objects_info)
        boxxml_file = f"{save_path}/aug_{number}_{count}.xml"
        with open(boxxml_file, "w") as f:
            f.write(xml)
        count += 1
    print(cats_count)


def test_single_xml():
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
        aug_images_from_xml(dir_, image_files, xml_files, background_path, cat_aug_path, folder, source,
                            x_proportion=x_proportion, y_proportion=y_proportion)


def test_single():
    data_path = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\5_抽取目标\网络"
    aug_path_d = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\7_增强图片\单类别_直接融合"
    aug_path_p = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\7_增强图片\单类别_金字塔融合"
    background_path = r"E:\Programming\Python\8_Ganlanz\food_recognition\data" \
                      r"set\自建数据集\4_背景底图"
    folder = r"Food2019"
    source = 'FoodDection'
    dirs = os.listdir(data_path)
    for dir_ in dirs:
        x_proportion, y_proportion = get_propotions(dir_)
        cat_path = data_path + "/" + dir_
        cat_aug_path_p = aug_path_p + "/" + dir_
        cat_aug_path_d = aug_path_d + "/" + dir_
        os.makedirs(cat_aug_path_p, exist_ok=True)
        os.makedirs(cat_aug_path_d, exist_ok=True)
        image_files = file_scanning(cat_path, file_format=IMAGE_FORMAT)
        print("类别：{}\t有效标注文件数量：{}".format(dir_, len(image_files)))
        # aug_images_from_image("", image_files, background_path, cat_aug_path_p, folder, source,cat_name=dir_,
        #            x_proportion=x_proportion, y_proportion=y_proportion)
        aug_images_from_image("direct", image_files, background_path, cat_aug_path_d, folder, source, cat_name=dir_,
                              x_proportion=x_proportion, y_proportion=y_proportion)


def test_mul_aug():
    object_path = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\5_抽取目标"
    background_config_file = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\4_背景底图\background_config.json"
    save_path = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\7_增强图片\多类别组合"
    aug_images_mul_object(object_path, background_config_file, save_path)


def main():
    # test_mul_aug()
    test_single()


if __name__ == '__main__':
    main()
