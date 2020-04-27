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
import gc
import time
import numpy as np
from random import choice, shuffle

from PIL import Image
from tqdm import tqdm
from xl_tool.xl_io import file_scanning, read_json
# from xl_tool.xl_io import l
import numpy as np
from random import uniform
import imgaug.augmenters as iaa
from xl_tool.data.image.config import IMAGE_FORMAT
from xl_tool.data.image.annonation import get_boximgs
from xl_tool.data.image.general import linear_contrast, grey_world, \
    affine_with_rotate_scale
from xl_tool.data.image.annonation import Text2XML
from xl_tool.data.image.blending import PoissonBlending, PyramidBlending, DirectBlending, SegBlend, \
    ObjectReplaceBlend
from PIL import Image
from cv2 import namedWindow, imshow, waitKey, WINDOW_FREERATIO, destroyAllWindows
from albumentations import Flip, Blur
import logging

def flip(image_array):
    fl = Flip()
    return fl(image=image_array)["image"]


def blur(image_array):
    fl = Blur(blur_limit=4)
    return fl(image=image_array)["image"]


aug_dict = {"contrast": linear_contrast, "grey": grey_world, "affine": affine_with_rotate_scale}


def cv_show_image(image_file, window_name="图片"):
    namedWindow(window_name, WINDOW_FREERATIO)  # 0表示压缩图片，图片过大无法显示
    imshow(window_name, image_file)
    k = waitKey(0)  # 无限期等待输入，需要有这个否则会死机
    if k == 27:  # 如果输入ESC退出
        destroyAllWindows()


def blending_choose(blending_method="direct", background_img_file=None, blending_img_file=None,
                    background_img_array=None, blending_img_array=None,
                    x=None, y=None, x_proportion=0.6, y_proportion=0.6,
                    x_shift=(0.5, 1.5), y_shift=(0.5, 1.9), blending_region=None, save_img=""):
    """选择不同的合成方法"""
    if blending_method == "poisson":
        blender = PoissonBlending()
    elif blending_method == "direct":
        blender = DirectBlending()
    else:
        blender = PyramidBlending(num_pyramid=5)
    # blender = DirectBlending() if blending_method == "direct" else
    blending_result, [x, y, x1, y1] = blender.blending_one_image(background_img_file, blending_img_file,
                                                                 background_img_array, blending_img_array,
                                                                 x, y, x_proportion, y_proportion,
                                                                 blending_region=blending_region,
                                                                 x_shift=x_shift, y_shift=y_shift,
                                                                 save_img=save_img, )
    return blending_result, [x, y, x1, y1]


def blending_images(background_img_file, blending_img_files, sp_dis, save_img="", blending_region=None,
                    blending_method="direct"):
    blender = DirectBlending() if blending_method == "direct" else PyramidBlending()
    background_img, image_sizes, positions = blender.blending_images(background_img_file, blending_img_files, sp_dis,
                                                                     save_img,
                                                                     blending_region)
    return background_img, image_sizes, positions


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
            aug_image, coordinate = blending_choose("", background_img_array=background_img_array,
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


def get_propotions(cat):
    big_propotion_cats = {"pizza", "saury", "chicken_brittle_bone"}
    small_propotion_cats = {"chicken_wing", "drumstick", "toast", "scallop", "oyster"}
    tiny_propotion_cats = ("egg_tart", "pumpkin_block", "prawn", "chicken_feet", "pumpkin_block", "prawn")
    if cat in big_propotion_cats:
        return 0.7, 0.7
    elif cat in small_propotion_cats:
        return 0.4, 0.4
    elif cat in tiny_propotion_cats:
        return 0.3, 0.3
    else:
        return 0.5, 0.5


def get_background_images(path):
    files = file_scanning(path, full_path=True, file_format=IMAGE_FORMAT)
    return files


def replace_augmentation(object_path, real_a_path, aug_path, object_classes):
    """替换目标增强"""
    print("替换目标增强")
    blender = ObjectReplaceBlend()
    for d in [i for i in os.listdir(f"{real_a_path}") if i in object_classes]:
        files = file_scanning(f"{real_a_path}/{d}", "jpg", sub_scan=True)
        single_object = f"{object_path}/{d}"
        os.makedirs(f"{aug_path}/{d}", exist_ok=True)
        object_images = [Image.open(i) for i in file_scanning(single_object, "jpg", sub_scan=True)]
        object_images = [i for i in object_images if (i.size[0] > 8 and i.size[1] > 8)]
        aspects = [i.size[0] / i.size[1] for i in object_images]
        if not aspects:
            continue
        try:
            object_images, aspects = zip(*sorted(zip(object_images, aspects), key=lambda x: x[1]))
        except Exception as e:
            pass
        tq = tqdm(files)
        for image_file in tq:
            xml_folder = r"Food2019"
            xml_source = 'FoodDection'
            xml_file = image_file.replace("jpg", "xml")
            try:
                save_img = f"{aug_path}/{d}/{'replace_aug_' + os.path.basename(image_file)}"
                if os.path.exists(save_img):
                    continue
                aug_image, boxes = blender.blending_one_image(image_file, object_images, aspects, xml_file,
                                                              random_choice=False, aspect_jump=1.0)

                aug_image.save(save_img)
                text2xml = Text2XML()
                objects_info = [[d] + coordinate for coordinate in boxes]
                boximg_file = os.path.basename(save_img)
                xml = text2xml.get_xml(xml_folder, boximg_file, boximg_file, xml_source, aug_image.size, objects_info)
                with open(save_img.replace("jpg", "xml"), "w") as f:
                    f.write(xml)
            except :
                pass
            tq.set_description(f"替换增强进度 {d}:")


def aug_images_from_image(blending_method, image_files, background_path, cat_aug_path, xml_folder, xml_source,
                          background_config_file,
                          cat_name="", target_number=None, x_proportion=0.4, y_proportion=0.5, max_num=10000,
                          simple_augs=("contrast", "grey", None),
                          min_size_sum=200, object_valid=True):
    """单目标合成"""
    background_configs = read_json(background_config_file)
    if min_size_sum:
        image_files = [file for file in image_files if sum(Image.open(file).size) > min_size_sum]
    pbar = tqdm(((image_files * 10)[:target_number]))
    count = 0
    for image_file in pbar:
        if count >= max_num:
            continue
        base_name = os.path.basename(image_file).split(".")[0]
        base_name = cat_name.replace(" ", "_") if re.search(r"[^a-zA-Z0-9_\-.]", base_name) else base_name
        background_config = choice(background_configs)
        background_file, region = background_config["file"], background_config['region']
        background_img_array = np.array(Image.open(background_file))
        save_img = f"{cat_aug_path}/aug_{1}_{count}_{blending_method}_{os.path.basename(background_file).split('.')[0]}_{base_name}.jpg"
        if not object_valid:  # 是否允许生成图片用于目标检测（如果允许，则不允许小目标组合）
            blending_img_array, x_p, y_p = object_combination(image_file, x_proportion, y_proportion)
        else:
            blending_img_array, x_p, y_p = np.array(Image.open(image_file)), x_proportion, y_proportion
        # 根据范围限定最大占比
        region_proportion = (
            (region[2] - region[0]) / background_img_array.shape[1],
            (region[3] - region[1]) / background_img_array.shape[0])
        x_p = min(region_proportion[0], x_proportion)
        y_p = min(region_proportion[1], y_proportion)
        # 根据候选区域的所占比例进行浮动
        x_shift = (0.9, 1.2) if region_proportion[0] <= x_proportion else (
            max(0.8 - region_proportion[0] + x_proportion, 0.2), max(1.1 + region_proportion[0] - x_proportion, 1.9))
        y_shift = (0.9, 1.2) if region_proportion[1] <= y_proportion else (
            max(0.8 - region_proportion[1] + y_proportion, 0.2), min(1.1 + region_proportion[1] - y_proportion, 1.9))
        aug_image, coordinate = blending_choose(blending_method, background_img_array=background_img_array,
                                                blending_img_array=blending_img_array,
                                                x_proportion=x_p, x_shift=x_shift, y_shift=y_shift,
                                                y_proportion=y_p, save_img=save_img, blending_region=region)
        if simple_augs:
            aug_method = choice(simple_augs)
            if aug_method:
                image_array = np.array(Image.open(save_img))
                Image.fromarray(aug_dict[aug_method](image_array)).save(save_img)
        text2xml = Text2XML()
        objects_info = [[cat_name] + coordinate]
        boximg_file = os.path.basename(save_img)
        xml = text2xml.get_xml(xml_folder, boximg_file, boximg_file, xml_source, aug_image.size, objects_info)
        boxxml_file = boximg_file.replace("jpg", "xml")
        aug_image.save(cat_aug_path + "/" + boximg_file)
        with open(cat_aug_path + "/" + boxxml_file, "w") as f:
            f.write(xml)
        count += 1
        pbar.set_description("图片增强进度：")


def object_combination(image_file, x_proportion, y_proportion):
    """小目标组合"""
    if x_proportion < 0.4:
        base = Image.open(image_file)
        size = base.size
        from random import choice
        expand = [choice([1, 2]) for i in range(2)]
        new_size = [expand[i] * size[i] for i in range(2)]
        # print("---------------------------------",expand )
        new = Image.new("RGB", new_size)
        for i in range(expand[0]):
            for j in range(expand[1]):
                new.paste(base, (i * size[0], j * size[1]))
        x_proportion = 0.3 if expand[0] <= 1 else 0.5
        y_proportion = 0.3 if expand[1] <= 1 else 0.5
        return np.array(new), x_proportion, y_proportion
    else:
        return np.array(Image.open(image_file)), x_proportion, y_proportion


def aug_images_mul_object(object_path, background_config_file, save_path, target_number=5000,
                          distribute=(0.05, 0.35, 0.4, 0.15),cats=None):
    """
    多目标合成
    """
    logging.info("开始多目标合成")
    folder = r"Food2019"
    source = 'FoodDection'
    cats = os.listdir(object_path) if not cats else cats
    print("---类别数量：", len(cats))
    try:
        cats.remove("empty")
    except:
        logging.info("未找到图片")
        pass
    background_configs = read_json(background_config_file)
    # background_config = x
    cats_files = {cat: file_scanning(f"{object_path}/{cat}", file_format=IMAGE_FORMAT) for cat in cats}
    count = 0
    cats_count = {cat: 0 for cat in cats}
    print("---开始合成")
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
        print(count)
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
