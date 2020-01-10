#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import xml.dom
import xml.dom.minidom
import re

from PIL import Image
from xl_tool.xl_io import file_scanning
from image_augmentation.general.config import IMAGE_FORMAT


def get_object_from_xml(obj_element):
    name = obj_element.getElementsByTagName("name")[0].firstChild.data
    xmin = obj_element.getElementsByTagName("xmin")[0].firstChild.data
    ymin = obj_element.getElementsByTagName("ymin")[0].firstChild.data
    xmax = obj_element.getElementsByTagName("xmax")[0].firstChild.data
    ymax = obj_element.getElementsByTagName("ymax")[0].firstChild.data
    return name, int(xmin), int(ymin), int(xmax), int(ymax)


def xml_object_extract(xml_file, image_file, save_path, object_classes=None, min_size_sum=100, w_h_limits=(10, 0.1)):
    """从xml中提取目标"""
    dom_tree = xml.dom.minidom.parse(xml_file)
    dom = dom_tree.documentElement
    objects = dom.getElementsByTagName("object")
    for obj in objects:
        # print("提取目标")
        name, xmin, ymin, xmax, ymax = get_object_from_xml(obj)
        if object_classes:
            if name not in object_classes:
                continue
        img = Image.open(image_file)
        if sum(img.size) <= min_size_sum:
            continue
        w_h = img.size[0] / img.size[1]
        if w_h > w_h_limits[0] or w_h < w_h_limits[1]:
            continue
        img_crop = img.crop((xmin, ymin, xmax, ymax))
        save_dir = save_path + "/" + name
        os.makedirs(save_dir, exist_ok=True)
        crop_file = os.path.join(save_dir, os.path.basename(image_file))
        img_crop.save(crop_file)
