from xl_tool.xl_io import file_scanning
import os
import shutil
from random import shuffle
#
# path = r"G:\food_dataset\7_增强图片\experienment\single_poisson"
# path2 = r"G:\food_dataset\7_增强图片\experienment\single_direct"
# target = r"G:\food_dataset\7_增强图片\experienment\poisson_blend_800"
# real_target = r"G:\food_dataset\7_增强图片\experienment\direct_blend_800"
#
# def copy_to_dataset(path,target):
#     for d in os.listdir(path):
#         if d not in ['pumpkin_block',
#                           'chives',
#                           'whole_chicken',
#                           'chicken_breast',
#                           'chicken_feet',
#                           'chicken_brittle_bone',
#                           'ribs',
#                           'hamburger_steak',
#                           'lamb_chops',
#                           'prawn',
#                           'salmon',
#                           'oyster',
#                           'scallop',
#                           'peanut']:
#             continue
#         files = file_scanning(f"{path}/{d}", file_format="jpg|jpe",sub_scan=True)
#         shuffle(files)
#         os.makedirs(f"{target}/train/{d}",exist_ok=True)
#         os.makedirs(f"{target}/val/{d}", exist_ok=True)
#         for i, file in enumerate(files[:800]):
#             t = f"{target}/train/{d}" if i<700 else f"{target}/val/{d}"
#             shutil.copy(file, f"{t}/{os.path.basename(file)}")
#             try:
#                 shutil.copy(file.replace("jpg","xml"), f"{t}/{os.path.basename(file).replace('jpg','xml')}")
#             except:
#                 pass
#
#
#
#
# copy_to_dataset(path,target)
# copy_to_dataset(path2,real_target)


import random
path = r"G:\food_dataset\7_增强图片\temp\single_direct"
dirs = [i for i in os.listdir(path) if i not in {"val", "train"}]
target = r"G:\food_dataset\7_增强图片\temp\spilited_dataset"
train = target + "/train"
val = target + "/val"
split = 0.8
seed =10
max_num = 1500
for d in dirs:
    class_ = d
    class_dir = f"{path}/{class_}"
    xml_files = file_scanning(class_dir, full_path=True,sub_scan=True,file_format="xml")[:max_num]
    random.seed(seed)
    shuffle(xml_files)
    split_index = int(0.8 * len(xml_files))
    os.makedirs(f"{train}/{class_}",exist_ok=True)
    os.makedirs(f"{val}/{class_}", exist_ok=True)
    for i,xml_file in enumerate(xml_files):
        if os.path.exists(xml_file.replace("xml","jpg")) or os.path.exists(xml_file.replace("xml","jpeg")):
            image_file = xml_file.replace("xml", "jpg")
            if i<split_index:
                shutil.copy(xml_file, f"{train}/{class_}/{os.path.basename(xml_file)}")
                shutil.copy(image_file, f"{train}/{class_}/{os.path.basename(image_file)}")
            else:
                shutil.copy(xml_file, f"{val}/{class_}/{os.path.basename(xml_file)}")
                shutil.copy(image_file, f"{val}/{class_}/{os.path.basename(image_file)}")