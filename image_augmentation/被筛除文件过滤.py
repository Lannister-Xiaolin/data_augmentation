#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import shutil

from xl_tool.xl_io import file_scanning
origin = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\3_公开数据集抽取\原始标注文件"
selected = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\3_公开数据集抽取\已筛选"
deleted = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\3_公开数据集抽取\被筛除"

origin_cats = os.listdir(origin)
selected_cats = os.listdir(selected)

for cat in selected_cats:
    if cat in origin_cats:
        selected_files = file_scanning(f"{selected}/{cat}",full_path=False,file_format="jpg")
        origin_files = file_scanning(f"{origin}/{cat}",full_path=False,file_format="jpg")
        deleted_files = set(origin_files) - set(selected_files)
        os.makedirs(f"{deleted}/{cat}",exist_ok=True)
        for file in deleted_files:
            shutil.copy(f"{origin}/{cat}/{file}", f"{deleted}/{cat}/{file}")
            xml = os.path.splitext(file)[0]+".xml"
            shutil.copy(f"{origin}/{cat}/{xml}", f"{deleted}/{cat}/{xml}")