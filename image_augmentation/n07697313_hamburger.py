#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os

from image_augmentation.general.config import IMAGE_FORMAT
from xl_tool.xl_io import file_scanning
extract_path = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\3_公开数据集抽取\原始标注文件"
imagenet_data = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\3_公开数据集抽取\原始标注文件\hamburger"
import xml.dom.minidom

image_files = file_scanning(imagenet_data, file_format=IMAGE_FORMAT,full_path=False)
xml_files = (file_scanning(imagenet_data, file_format=".xml",full_path=False))
xml_no_ext = set([file.replace(".xml", "") for file in xml_files])
annonationed_image_files = [file for file in image_files if file.replace(".JPEG", "") in xml_no_ext]
for file in image_files:
    if file not in annonationed_image_files:
        os.remove(imagenet_data + "/" + file)
for xml_file in xml_files:
    dom = xml.dom.minidom.parse(imagenet_data + "/" + xml_file)
    doc = dom.documentElement
    doc.getElementsByTagName('filename')[0].firstChild.data = doc.getElementsByTagName('filename')[0].firstChild.data + ".JPEG"
    doc.getElementsByTagName('database')[0].firstChild.data = "FoodDection"
    doc.getElementsByTagName('folder')[0].firstChild.data = "Food2019"
    doc.getElementsByTagName('name')[0].firstChild.data = "hamburger"
    y = doc.getElementsByTagName("source")[0]
    element = dom.createElement("path")
    text = dom.createTextNode(doc.getElementsByTagName('filename')[0].firstChild.data)
    element.appendChild(text)
    doc.insertBefore(element,y)

    with open(imagenet_data + "/" + xml_file, "w") as tmpf:
        doc.writexml(tmpf)
    test = ""
    with open(imagenet_data + "/" + xml_file, "r") as f:
        result = f.read()
    with open(imagenet_data + "/" + xml_file, "w") as f:
        f.write(result.replace("/path>", "/path>\n    "))
