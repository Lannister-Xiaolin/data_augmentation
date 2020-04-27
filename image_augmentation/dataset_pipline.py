#!usr/bin/env python3
# -*- coding: UTF-8 -*-
"""数据集生成模块，包括数据增强，数据集切分等
整个流程如下：
    真实场景标框数据与分类数据入库
        从混合标注的数据中切分分别放入指定文件夹
    目标抽取（分真实与非真实）—记录已生成文件，不进行重复生成
    图片增强
    数据组合:
        分类数据集：
            指定场景数据集，组成如下：
                直接融合数据   单目标 + 单类多目标
                真实场景数据
                    训练验证集分割
                        直接融合：8:2 真实场景6:4
            不限场景数据集，组成如下：
                直接融合数据  单目标 + 单类多目标
                纯目标数据    单目标 + 单类多目标
                单类别网络图片
                真实场景数据
        检测数据集：
            指定场景数据集
                原始标注
                单目标增强
                多目标增强
                真实场景数
"""
import os
import re
import shutil
import random
# random.seed = 100
from tqdm import tqdm
from xl_tool.xl_io import file_scanning, read_txt
from xl_tool.xl_general import count_files
from xl_tool.data.image.config import IMAGE_FORMAT
from xl_tool.data.image.annonation import xml_object_extract
import threading
from augmentation_batch import aug_images_from_image, get_propotions, aug_images_mul_object, replace_augmentation
import time, zipfile

real_data_pattern = ""
website_pattern = "百度|baidu|必应|biying|搜狗|sougou"
public_pattern = "^\d{12}\.|^n\d{8}"


def update_bg_config_file():
    files = file_scanning(r"G:\food_dataset\4_背景底图", "xml", sub_scan=False)
    configs = []
    for file in files:
        temp = {}
        temp["file"] = file.replace("xml", "jpg")
        from xl_tool.data.image.annonation import get_bndbox
        box = get_bndbox(file)[0]
        temp["region"] = [int(i) for i in box["coordinates"]]
        configs.append(temp)
    from xl_tool.xl_io import save_to_json
    save_to_json(configs, r"G:\food_dataset\4_背景底图\background_config.json", indent=4)


# background_config_file =

def data_to_my_path(temp_path, website_path, real_path, public_path=None):
    """从混合的标注数据中抽取数据到按照来源进行划分的数据源
    """
    dirs = os.listdir(temp_path)
    if not dirs:
        print("未发现任何数据！")
        return False
    dir_files = [file_scanning(d, file_format=IMAGE_FORMAT + "|xml") for d in dirs]
    for d, files in zip(dirs, dir_files):
        for file in files:
            basename = os.path.basename(file)
            if re.search(website_pattern, file):
                os.makedirs(f"{website_path}/{d}", exist_ok=True)
                dst = f"{website_path}/{d}/{basename}"
            elif re.search(public_pattern, file):
                if public_path:
                    os.makedirs(f"{public_path}/{d}", exist_ok=True)
                    dst = f"{public_path}/{d}/{basename}"
                else:
                    continue
            else:
                os.makedirs(f"{real_path}/{d}", exist_ok=True)
                dst = f"{real_path}/{d}/{basename}"
            shutil.copy(file, dst)


def object_extract(public_path, wesite_path, real_path, save_path,
                   object_classes, filter_frozen=True, filter_processed=True, min_size_sum=100, w_h_limits=(10, 0.1)):
    """目标抽取，抽取来源为网络/公开数据集/真实场景"""
    files = file_scanning(public_path, file_format=IMAGE_FORMAT, sub_scan=True) + \
            file_scanning(wesite_path, sub_scan=True, file_format=IMAGE_FORMAT)
    real_files = file_scanning(real_path, file_format=IMAGE_FORMAT, sub_scan=True)
    real_start = len(files)
    files = files + real_files
    save_website, save_real = f"{save_path}/待筛选_网络", f"{save_path}/待筛选_真实"
    forzen_object = read_txt(f"{save_path}/frozen_object.txt", return_list=True)
    processed_files = set(read_txt(f"{save_path}/processed_files.txt", return_list=True))
    if filter_frozen:
        object_classes = list(set(object_classes) - set(forzen_object))
    pbar = tqdm(list(enumerate(files)))
    for i, file in pbar:
        basename = os.path.basename(file)
        if filter_processed and (basename in processed_files):
            continue
        xml_file = re.sub(IMAGE_FORMAT, "xml", file)
        save_dir = save_website if i < real_start else save_real
        # print(save_dir)
        if os.path.exists(xml_file):
            xml_object_extract(xml_file, file, save_dir,
                               object_classes=object_classes, min_size_sum=min_size_sum, w_h_limits=w_h_limits)
            processed_files.add(basename)
    pbar.set_description("目标抽取进度：")
    print("--->目标抽取完成")
    count_files(save_website)
    count_files(save_real)
    with open(f"{save_path}/processed_files.txt", "w") as f:
        f.write("\n".join(processed_files))


def aug_single_object(object_path, aug_path, background_path, auged_classes, object_classes, background_config_file,
                      real_a_path,
                      target_numbers=(100, 1500),
                      save_old=False, object_valid=False):
    aug_path_d = f"{aug_path}/single_direct"
    aug_path_p = f"{aug_path}/single_pyramid"
    aug_path_po = f"{aug_path}/single_poisson"
    aug_path_r = f"{aug_path}/replace"
    os.makedirs(aug_path_d, exist_ok=True)
    os.makedirs(aug_path_p, exist_ok=True)
    os.makedirs(aug_path_po, exist_ok=True)
    if save_old:
        zip_compress(aug_path_d, f"{os.path.split(aug_path)[0]}/single_direct.zip")
        zip_compress(aug_path_p, f"{os.path.split(aug_path)[0]}/single_pyramid.zip")
    delete_old_scenario(aug_path_p)
    delete_old_scenario(aug_path_d)
    delete_old_scenario(aug_path_po)
    folder = r"Food2019"
    source = 'FoodDection'
    dirs = [d for d in os.listdir(object_path) if
            (os.path.isdir(f"{object_path}/{d}") and (d not in auged_classes) and d in object_classes)]
    replace_augmentation(object_path, real_a_path, aug_path_r, object_classes)
    for dir_ in dirs:
        x_proportion, y_proportion = get_propotions(dir_)
        cat_path = object_path + "/" + dir_
        cat_aug_path_p = aug_path_p + "/" + dir_
        cat_aug_path_d = aug_path_d + "/" + dir_
        cat_aug_path_po = aug_path_po + "/" + dir_
        os.makedirs(cat_aug_path_p, exist_ok=True)
        os.makedirs(cat_aug_path_po, exist_ok=True)
        os.makedirs(cat_aug_path_d, exist_ok=True)
        image_files = file_scanning(cat_path, file_format=IMAGE_FORMAT)
        print("类别：{}\t有效标注文件数量：{}".format(dir_, len(image_files)))
        threads = []
        threads.append(threading.Thread(target=aug_images_from_image,
                                        args=("pyramid", image_files, background_path, cat_aug_path_p, folder, source,
                                              background_config_file),
                                        kwargs={"cat_name": dir_, "target_number": target_numbers[2],
                                                "x_proportion": x_proportion,
                                                "y_proportion": y_proportion, "object_valid": object_valid}))
        threads.append(threading.Thread(target=aug_images_from_image,
                                        args=("direct", image_files, background_path, cat_aug_path_d, folder, source,
                                              background_config_file),
                                        kwargs={"cat_name": dir_, "target_number": target_numbers[0],
                                                "x_proportion": x_proportion,
                                                "y_proportion": y_proportion, "object_valid": object_valid}))
        threads.append(threading.Thread(target=aug_images_from_image,
                                        args=("poisson", image_files, background_path, cat_aug_path_po, folder, source,
                                              background_config_file),
                                        kwargs={"cat_name": dir_, "target_number": target_numbers[1],
                                                "x_proportion": x_proportion,
                                                "y_proportion": y_proportion, "object_valid": object_valid}))
        for thread in threads:
            time.sleep(0.02)
            thread.start()
            # threads = list(range(thread_num))
        for thread in threads:
            thread.join()
    print("--->单目标增强完成")


def copy_files_train_val(files_lists, cats, save_path, val_split=0.8, limit=100000):
    for files, cat in zip(files_lists, cats):
        random.shuffle(files)
        files = files[:limit]
        val_start = int(val_split * len(files))
        print(cat, val_start, len(files))
        train_dst_dir = f"{save_path}/train/{cat}"
        val_dst_dir = f"{save_path}/val/{cat}"
        os.makedirs(train_dst_dir, exist_ok=True)
        os.makedirs(val_dst_dir, exist_ok=True)
        for i, file in enumerate(files):
            dst = f"{train_dst_dir}/{os.path.basename(file)}" if i < val_start else \
                f"{val_dst_dir}/{os.path.basename(file)}"
            # print(dst)
            (shutil.copy(file, dst))


def delete_old_scenario(scenario_path):
    sub_datasets = os.listdir(scenario_path)
    for sub in sub_datasets:
        path = f"{scenario_path}/{sub}"
        dirs = os.listdir(path)
        for d in dirs:
            if d != "empty":
                if os.path.isfile(f"{path}/{d}"):
                    os.remove((f"{path}/{d}"))
                else:
                    shutil.rmtree(f"{path}/{d}")


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


def remove_duplicat(files):
    basename = set()
    temp = []
    for sub_files in files:
        temp_sub = []
        for file in sub_files:
            name = os.path.basename(file)
            if name in basename:
                continue
            else:
                temp_sub.append(file)
                basename.add(name)
        temp.append(temp_sub)
    return temp


def create_classify_dataset(single_direct_aug_path, wesite_c_path, object_w_path, real_a_path, dataset_path,
                            object_classes, created_dataset_classes, single_replace_aug_path, single_poisson_aug_path,
                            website_limit=600, object_limit=600, aug_limits=[1100, 300, 100], replace_limit=600,
                            save_old=False):
    specified_scenario = f"{dataset_path}/specified_scenario"

    unspecified_scenario = f"{dataset_path}/unspecified_scenario"
    unspecified_scenario_nonaug_nonreal = f"{dataset_path}/specified_scenario_nonaug_nonaug_nonreal"
    unspecified_scenario_nonaug = f"{dataset_path}/specified_scenario_nonaug"
    if save_old:
        zip_compress(dataset_path,
                     f"{os.path.split(os.path.abspath(dataset_path))[0]}"
                     f"/生成数据集_{str(time.localtime().tm_mon).rjust(2, '0')}{time.localtime().tm_yday}.zip")
    try:
        threads = []
        threads.append(threading.Thread(target=delete_old_scenario,args=(specified_scenario, )))
        threads.append(threading.Thread(target=delete_old_scenario, args=(unspecified_scenario,)))
        threads.append(threading.Thread(target=delete_old_scenario, args=(unspecified_scenario_nonaug_nonreal,)))
        threads.append(threading.Thread(target=delete_old_scenario, args=(unspecified_scenario_nonaug,)))
        for thread in threads:
            time.sleep(0.02)
            thread.start()
            # threads = list(range(thread_num))
        for thread in threads:
            thread.join()
    except:
        pass
    object_classes = object_classes + ["empty"]
    website_cats = [d for d in os.listdir(single_direct_aug_path) if
                    (d in object_classes and (d not in created_dataset_classes))]
    direct_aug_cats = [d for d in os.listdir(single_direct_aug_path) if
                       (d in object_classes and (d not in created_dataset_classes))]
    real_cats = [d for d in os.listdir(real_a_path) if d in object_classes and (d not in created_dataset_classes)]
    object_cats = [d for d in os.listdir(object_w_path) if d in object_classes and (d not in created_dataset_classes)]
    direct_aug_files = [file_scanning(f"{single_direct_aug_path}/{d}", file_format=IMAGE_FORMAT, sub_scan=True)
                        for d in direct_aug_cats]
    replace_aug_files = [file_scanning(f"{single_replace_aug_path}/{d}", file_format=IMAGE_FORMAT, sub_scan=True)
                         for d in direct_aug_cats]
    poisson_aug_files = [file_scanning(f"{single_poisson_aug_path}/{d}", file_format=IMAGE_FORMAT, sub_scan=True)
                         for d in direct_aug_cats]
    real_files = [list(set(file_scanning(f"{real_a_path}/{d}", file_format=IMAGE_FORMAT, sub_scan=True) + file_scanning(
        f"{real_a_path}/{d}".replace("0_已标框", "1_已分类"), file_format=IMAGE_FORMAT, sub_scan=True)))
                  for d in real_cats]
    real_files = remove_duplicat(real_files)
    website_files = [list(set(
        file_scanning(f"{wesite_c_path}/{d}", file_format=IMAGE_FORMAT, sub_scan=True) + file_scanning(
            f"{wesite_c_path}/{d}".replace("1_已分类", "0_已标框"), file_format=IMAGE_FORMAT, sub_scan=True)))
        for d in website_cats]
    website_files = remove_duplicat(website_files)
    object_files = [file_scanning(f"{object_w_path}/{d}", file_format=IMAGE_FORMAT, sub_scan=True)
                    for d in object_cats]
    # print("--->指定场景数据-->复接融合数据")
    # copy_files_train_val(direct_aug_files[:aug_limits[0]], direct_aug_cats, specified_scenario, val_split=0.8, limit=aug_limits[0])
    # copy_files_train_val(replace_aug_files[:replace_limit], direct_aug_cats, specified_scenario, val_split=0.8, limit=replace_limit)
    # copy_files_train_val(poisson_aug_files[:aug_limits[1]], direct_aug_cats, specified_scenario, val_split=0.8, limit=aug_limits[1])
    # print("--->指定场景数据-->复制真实场景数据")
    # copy_files_train_val(real_files, real_cats, specified_scenario, val_split=0.6)

    threads = []
    threads.append(threading.Thread(target=copy_files_train_val, args=(direct_aug_files,direct_aug_cats, unspecified_scenario),kwargs=dict(val_split=0.8, limit=aug_limits[0])))
    threads.append(threading.Thread(target=copy_files_train_val, args=(replace_aug_files, direct_aug_cats, unspecified_scenario),kwargs=dict(val_split=0.8, limit=replace_limit)))
    threads.append(threading.Thread(target=copy_files_train_val, args=(poisson_aug_files, direct_aug_cats, unspecified_scenario),kwargs=dict(val_split=0.8, limit=aug_limits[1])))
    threads.append(threading.Thread(target=copy_files_train_val, args=(real_files, real_cats, unspecified_scenario,),kwargs=dict(val_split=0.8,)))
    threads.append(threading.Thread(target=copy_files_train_val, args=(website_files, website_cats, unspecified_scenario),kwargs=dict(val_split=0.8, limit=aug_limits[0])))
    threads.append(threading.Thread(target=copy_files_train_val, args=(object_files, object_cats, unspecified_scenario,),kwargs=dict(val_split=0.8, limit=object_limit)))
    for thread in threads:
        time.sleep(0.02)
        thread.start()
        # threads = list(range(thread_num))
    for thread in threads:
        thread.join()

    # print("--->不限场景数据-->复制直接融合数据")
    # copy_files_train_val(direct_aug_files, direct_aug_cats, unspecified_scenario, val_split=0.8, limit=aug_limits[0])
    # copy_files_train_val(replace_aug_files, direct_aug_cats, unspecified_scenario, val_split=0.8, limit=replace_limit)
    # copy_files_train_val(poisson_aug_files, direct_aug_cats, unspecified_scenario, val_split=0.8, limit=aug_limits[1])
    # # copy_files_train_val(p, direct_aug_cats, unspecified_scenario, val_split=0.8, limit=aug_limits[1])
    # print("--->不限场景数据-->复制真实场景数据")
    # copy_files_train_val(real_files, real_cats, unspecified_scenario, val_split=0.8)
    # print("--->不限场景数据-->复制网络图片")
    # copy_files_train_val(website_files, website_cats, unspecified_scenario, val_split=0.8, limit=website_limit)
    # print("--->不限场景数据-->复制目标数据")
    # copy_files_train_val(object_files, object_cats, unspecified_scenario, val_split=0.8, limit=object_limit)

    # print("--->不限场景数据，无数据增强-->复制真实场景数据")
    # copy_files_train_val(real_files, real_cats, unspecified_scenario_nonaug, val_split=0.6)
    # print("--->不限场景数据，无数据增强-->复制网络图片")
    # copy_files_train_val(website_files[:(website_limit + 400)], website_cats, unspecified_scenario_nonaug,
    #                      val_split=0.8)
    # print("--->不限场景数据，无数据增强-->复制目标数据")
    # copy_files_train_val(object_files[:(object_limit + 400)], object_cats, unspecified_scenario_nonaug, val_split=0.8)
    #
    # print("--->不限场景数据，无数据增强，无数据增强和真实场景-->复制网络图片")
    # copy_files_train_val(website_files[:(website_limit + 400)], website_cats, unspecified_scenario_nonaug_nonreal,
    #                      val_split=0.8)
    # print("--->不限场景数据，无数据增强，无数据增强和真实场景-->复制目标数据")
    # copy_files_train_val(object_files[:(object_limit + 400)], object_cats, unspecified_scenario_nonaug_nonreal,
    #                      val_split=0.8)


def aug_mul_object(object_w_path, background_config_file, aug_path_mul, auged_classes, target_number=5000,cats=None):
    # save_path = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\7_增强图片\多类别组合"
    os.makedirs(aug_path_mul, exist_ok=True)
    # try:
    #     if not (set(os.listdir(aug_path_mul)) - set(auged_classes)):
    #         return
    # except:
    #     pass
    aug_images_mul_object(object_w_path, background_config_file, aug_path_mul, target_number=target_number,cats=cats)


base_path = r"G:\food_dataset"
background_config_file = f"{base_path}/4_背景底图/background_config.json"
website_a_path = f"{base_path}/2_网络图片/0_已标框"
wesite_c_path = f"{base_path}/2_网络图片/1_已分类"
real_a_path = f"{base_path}/1_真实场景/0_已标框"
# real_a_path = f"{base_path}/1_真实场景/0_已标框"
public_a_path = f"{base_path}/3_公开数据集抽取/0_已筛框"
temp_path = f"{base_path}/temp"
object_save_path = f"{base_path}/5_抽取目标"
object_w_path = f"{base_path}/5_抽取目标/混合"
aug_path = f"{base_path}/7_增强图片"
aug_path_mul = f"{base_path}/7_增强图片/mul_object"

background_path = f"{base_path}/4_背景底图"
single_direct_aug_path = f"{base_path}/7_增强图片/single_direct"
single_pyramid_aug_path = f"{base_path}/7_增强图片/single_pyramid"
single_replace_aug_path = f"{base_path}/7_增强图片/replace"
single_poisson_aug_path = f"{base_path}/7_增强图片/single_poisson"
dataset_path = f"{base_path}/8_生成数据集"


def auto_pipline():
    # auged_classes = ['broccoli', 'corn_kernels', 'hamburger', 'pizza', 'pork_belly_piece', "",'unknown']
    # created_dataset_classes = ['broccoli', 'corn kernels', 'hamburger', 'pizza', 'pork_belly_piece', 'unknown']
    object_classes = ['bacon', 'broccoli', 'chicken_wing',
                      'corn', 'drumstick', 'pizza',
                      'steak', 'tilapia', 'toast',
                      "sausage", "saury", "lamb_kebab",
                      "flammulina_velutipes", "egg_tart", "sweet_potato"] \
                     + \
                     ['barbecued_pork', 'chicken_breast', 'chicken_brittle_bone',
                      'chicken_feet', 'chives', 'eggplant',
                      'green_pepper', 'hamburger_steak', 'lamb_chops',
                      'oyster', 'peanut', 'potato',
                      'prawn', 'pumpkin_block', 'ribs',
                      'salmon', 'scallion_cake', 'scallop',
                      'shiitake_mushroom', 'whole_chicken'] +\
                     ['almond', 'asparagus', 'baby_cabbage',
                      'banana_slice', 'band_fish_chunks', 'basa_fish_fillet',
                      'beancurd_skin', 'beef_tendon', 'broccolini',
                      'cashew', 'cherry_tomatoes', 'chestnut',
                      'cod_pieces', 'cowpea', 'croissant',
                      'dough_sticks', 'dried_tofu', 'dumplings',
                      'fish_tofu', 'french_fries', 'fried_chicken_chunks',
                      'fuzhu', 'garlic_clove', 'gluten',
                      'lemon_slice', 'lotus_root_slice', 'meatball',
                      'melon_seeds', 'okra', 'pippi_shrimp',
                      'popcorn_chicken', 'pork_ribs', 'potato_cake',
                      'potato_chunks', 'potato_slice', 'puff_pastry',
                      'pumpkin_seed', 'quail_eggs', 'squid',
                      'squid_tentacles', 'steamed_bread', 'sweet_pepper',
                      'tofu', 'trotters', 'waffle', 'walnut_meat',
                      'white_pomfret', 'white_shell', 'whole_chicken_wing',
                      'yellow_croaker']
    auged_classes = ['empty', ]
    # auged_classes = ['empty', 'bacon', 'broccoli', 'chicken_wing',
    #                  'corn', 'drumstick', 'hamburger', 'pizza',
    #                  'steak', 'tilapia', 'toast',
    #                  "sausage", "saury", "lamb_kebab",
    #                  "flammulina_velutipes", "egg_tart", "sweet_potato"]
    created_dataset_classes = []
    save_old = False
    # print("1——标注数据入库")
    # try:
    #     data_to_my_path(temp_path, website_a_path, real_a_path)
    # except:
    #     pass
    # print("2——目标抽取")
    # object_extract(public_a_path, website_a_path, real_a_path, object_save_path,
    #                object_classes=object_classes, filter_frozen=True, filter_processed=True,
    #                min_size_sum=100, w_h_limits=(10, 0.1))
    # input("请确认目标数据合并和筛选完成！！")
    # update_bg_config_file()
    # print("3——图片增强-单类别直接融合")
    # aug_path = f"{base_path}/7_增强图片/"  #
    # aug_single_object(f"{object_save_path}/混合", aug_path, background_path, auged_classes, object_classes,
    #                   background_config_file, real_a_path=real_a_path,
    #                   target_numbers=(1000, 200, 50), save_old=save_old, object_valid=True)
    # # aug_path = f"{base_path}/7_增强图片/experienment"  # 临时
    # aug_single_object(f"{object_save_path}/混合", aug_path, background_path, auged_classes, object_classes,
    #                   background_config_file, real_a_path=real_a_path,
    #                   target_numbers=(1000, 200, 50), save_old=save_old, object_valid=True)
    # print("4——分类数据组合")
    # create_classify_dataset(single_direct_aug_path=single_direct_aug_path, wesite_c_path=wesite_c_path,
    #                         object_w_path=object_w_path, real_a_path=real_a_path, replace_limit=300,
    #                         aug_limits=[1000, 200, 100],
    #                         created_dataset_classes=created_dataset_classes, website_limit=800, object_limit=500,
    #                         dataset_path=dataset_path, object_classes=object_classes
    #                         , save_old=save_old, single_replace_aug_path=single_replace_aug_path,
    #                         single_poisson_aug_path=single_poisson_aug_path)
    print("5——目标检测数据集组合")
    aug_mul_object(object_w_path, background_config_file, aug_path_mul, auged_classes,
                   target_number=1000 * len(object_classes),cats=object_classes)


if __name__ == '__main__':
    auto_pipline()
