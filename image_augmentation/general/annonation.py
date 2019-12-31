#!usr/bin/env python3
# -*- coding: UTF-8 -*-

import xml.dom
import xml.dom.minidom
from PIL import Image
from image_augmentation.general.preprocessing import read_with_rgb

coordinate_name = ["xmin", "ymin", "xmax", "ymax"]


def get_bndbox(xml_file):
    """获取文件里面的bounding box坐标"""
    dom = xml.dom.minidom.parse(xml_file)
    doc = dom.documentElement
    bndboxes = []
    for object_node in doc.getElementsByTagName('object'):
        name = object_node.getElementsByTagName('name')[0].firstChild.data
        coordinates = [int(object_node.getElementsByTagName(i)[0].firstChild.data) for i in coordinate_name]
        bndboxes.append({"name": name, "coordinates": coordinates})
    # print(bndboxes)
    return bndboxes


def get_boximgs(image_file, xml_file) -> list:
    img = read_with_rgb(image_file)
    bndboxes = get_bndbox(xml_file)
    boximgs = []
    for bndbox in bndboxes:
        boximg = img.crop(bndbox["coordinates"])
        boximgs.append({"img": boximg, "name": bndbox["name"]})
    return boximgs


def main():
    get_boximgs(
        r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\3_公开数据集抽取\原始标注文件\hamburger\n07697313_13461.JPEG",
        r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\3_公开数据集抽取\原始标注文件\hamburger\n07697313_13461.xml")


if __name__ == '__main__':
    main()
