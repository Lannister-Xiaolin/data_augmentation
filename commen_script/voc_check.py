from xl_tool.data.image.annonation import get_bndbox
from xl_tool.data.image.general import read_with_rgb
from xl_tool.xl_io import file_scanning, read_txt
from tqdm import tqdm
import random
import shutil
from PIL import Image
import os
import zipfile


def zip_compress(path, zip_filename, compression=zipfile.ZIP_DEFLATED):
    """ compress file or directory"""
    print(zip_filename)
    z = zipfile.ZipFile(zip_filename, "w", compression)
    if os.path.isfile(path):
        z.write(path)
    else:
        for root, dirs, files in os.walk(path, topdown=True):
            if files:
                for file in files:
                    z.write(f"{root}/{file}")
            else:
                z.write(root)
    z.close()


def voc_annotation_check(xml_files, width_limit=1, height_limit=1, repair=False, repair_path=None):
    for xml_file in tqdm(xml_files):
        image_file = xml_file.replace("xml", "jpg")
        try:
            width, height = Image.open(image_file).size
            read_with_rgb(image_file).save(image_file)
            # print("hhhhh")
        except FileNotFoundError:
            width, height = 2000, 2000
        bndboxes = get_bndbox(xml_file)
        for box in (bndboxes):
            box_w = box['coordinates'][2] - box['coordinates'][0]
            box_h = box['coordinates'][3] - box['coordinates'][1]
            if (box_w) <= width_limit or (box_h
            ) <= height_limit or box_w > width or box_h > height:
                print(xml_file)
                import re
                import os
                if repair:
                    text = read_txt(xml_file)
                    xml_file = xml_file if not repair_path else repair_path + "/" + os.path.basename(xml_file)
                    pattern = r"\<object\>.{{8,12}}{}.{{80,160}}{}." \
                              r"{{10,30}}{}.{{10,30}}{}.{{10,30}}{" \
                              r"}.{{20,50}}\</object\>\n".format(box['name'], *box["coordinates"])
                    text1 = re.sub(pattern, "", text, flags=re.S)
                    with open(xml_file, "w", encoding="utf-8") as f:
                        f.write(text1)
                continue


def rgb_confirm(path):
    for file in file_scanning(path, "jpg", sub_scan=True):
        read_with_rgb(file).save(file)


def split_train_val(path, train, val, seed=10, max_num=1000):
    dirs = [i for i in os.listdir(path) if i not in {"val", "train"}]
    for d in dirs:
        class_ = d
        class_dir = f"{path}/{class_}"
        xml_files = file_scanning(class_dir, full_path=True, sub_scan=True, file_format="xml")[:max_num]
        random.seed(seed)
        random.shuffle(xml_files)
        split_index = int(0.8 * len(xml_files))
        os.makedirs(f"{train}/{class_}", exist_ok=True)
        os.makedirs(f"{val}/{class_}", exist_ok=True)
        for i, xml_file in enumerate(xml_files):
            if os.path.exists(xml_file.replace("xml", "jpg")) or os.path.exists(xml_file.replace("xml", "jpeg")):
                image_file = xml_file.replace("xml", "jpg")
                if i < split_index:
                    shutil.copy(xml_file, f"{train}/{class_}/{os.path.basename(xml_file)}")
                    shutil.copy(image_file, f"{train}/{class_}/{os.path.basename(image_file)}")
                else:
                    shutil.copy(xml_file, f"{val}/{class_}/{os.path.basename(xml_file)}")
                    shutil.copy(image_file, f"{val}/{class_}/{os.path.basename(image_file)}")


#
# xml_files = file_scanning(r"G:\food_dataset\8_生成数据集\food_object_dataset\origin_data", "xml", sub_scan=True)[50000:]
# print("扫描到文件数量：", len(xml_files))
# voc_annotation_check(xml_files, repair=True)
# rgb_confirm(r"G:\food_dataset\8_生成数据集\food_object_dataset\origin_data")
root = r"G:\food_dataset\8_生成数据集\food_object_dataset"
shutil.rmtree(f"{root}/spilited_dataset/train")
shutil.rmtree(f"{root}/spilited_dataset/val")
import threading

threads = []
threads.append(threading.Thread(target=shutil.rmtree, args=(f"{root}/spilited_dataset/train",)))
threads.append(threading.Thread(target=shutil.rmtree, args=(f"{root}/spilited_dataset/val",)))
for thread in threads:
    thread.start()
    # threads = list(range(thread_num))
for thread in threads:
    thread.join()

# threads.append(
#     threading.Thread(target=copy_files_train_val, args=(replace_aug_files, direct_aug_cats, unspecified_scenario),
#                      kwargs=dict(val_split=0.8, limit=replace_limit)))
# threads.append(
#     threading.Thread(target=copy_files_train_val, args=(poisson_aug_files, direct_aug_cats, unspecified_scenario),
#                      kwargs=dict(val_split=0.8, limit=aug_limits[1])))
# threads.append(threading.Thread(target=copy_files_train_val, args=(real_files, real_cats, unspecified_scenario,),
#                                 kwargs=dict(val_split=0.8, )))
# threads.append(threading.Thread(target=copy_files_train_val, args=(website_files, website_cats, unspecified_scenario),
#                                 kwargs=dict(val_split=0.8, limit=aug_limits[0])))
# threads.append(threading.Thread(target=copy_files_train_val, args=(object_files, object_cats, unspecified_scenario,),
#                                 kwargs=dict(val_split=0.8, limit=object_limit)))

os.makedirs(f"{root}/spilited_dataset/train",exist_ok=True)
os.makedirs(f"{root}/spilited_dataset/val",exist_ok=True)
split_train_val(f"{root}/origin_data", f"{root}/spilited_dataset/train",
                f"{root}/spilited_dataset/val", seed=10,
                max_num=5000)
#
split_train_val(r"G:\food_dataset\7_增强图片\single_direct",
                f"{root}/spilited_dataset_direct/train", f"{root}/spilited_dataset_direct/val", seed=10, max_num=1000)
split_train_val(r"G:\food_dataset\7_增强图片\single_poisson",
                f"{root}/spilited_dataset_poisson/train", f"{root}/spilited_dataset_poisson/val", seed=10, max_num=200)
split_train_val(r"G:\food_dataset\7_增强图片\replace",
                f"{root}/spilited_dataset_replace/train", f"{root}/spilited_datasett_replace/val", seed=10, max_num=300)
# zip_compress(r"G:\food_dataset\8_生成数据集\food_object_dataset\spilited_dataset",
#              r"G:\food_dataset\8_生成数据集\food_object_dataset\spilited_dataset.zip")
# zip_compress(r"G:\food_dataset\8_生成数据集\unspecified_scenario", r"G:\food_dataset\8_生成数据集\unspecified_scenario_85.zip")
