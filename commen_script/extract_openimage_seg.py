import os
import pandas as pd
import shutil
from math import fabs
from PIL import Image
from xl_tool.xl_io import file_scanning,read_txt
# class_des = pd.read_csv("G:/Public_dataset/OpenImage/class-descriptions-boxable.csv",names=["code","name"])
class_description_file= "G:/Public_dataset/OpenImage/class-descriptions-boxable.csv"
food_code = "G:/Public_dataset/OpenImage/seg_food_code.txt"
mask_files_path = "G:/Public_dataset/OpenImage/segmentation_mask"
target = "G:/Public_dataset/OpenImage/extact_data"
origin_path = r"G:\Public_dataset\OpenImage\image\train_0/"
seg_boxes_file = "G:/Public_dataset/OpenImage/train-annotations-object-segmentation.csv"
def extact_food_data_from_openimage(origin_path, class_description_file, food_code,mask_files_path,target):
    class_des = pd.read_csv(class_description_file, names=["code", "name"])
    class_des["code"] = class_des["code"].map(lambda x:x.replace("/",""))
    class_des = class_des.set_index("code")
    class_des = class_des.to_dict()["name"]
    mask_id = set(read_txt(food_code,return_list=True))
    mask_files = file_scanning(mask_files_path,file_format="jpg|jpeg|png",sub_scan=True)
    food_seg_files = []
    for file in mask_files:
        cls_id = os.path.basename(file).split("_")[-2]
        image_file = os.path.basename(file).split("_")[0]+".jpg"
        if cls_id in mask_id:
            food_seg_files.append((file,cls_id, class_des[cls_id],image_file))
    print("总共发现图片：",len(food_seg_files))
    with open(target+"/valid_files.txt", "w") as f:
        f.write("\n".join(["\t".join(food_seg_file) for food_seg_file in food_seg_files]))
    for file,cls_id,name,image_file in food_seg_files:
        os.makedirs( target + f"/mask/{name}/", exist_ok=True)
        os.makedirs(target + f"/image/{name}/", exist_ok=True)
        shutil.copy(file, target + f"/mask/{name}/" + os.path.basename(file))
        try:
            shutil.copy(origin_path + image_file, target + f"/image/{name}/" + image_file)
            print("fuck you ")
        except (FileExistsError, FileNotFoundError):
            continue


def get_seg_area(seg_boxes_file,class_description_file,extact_path):
    seg_boxes = pd.read_csv(seg_boxes_file,index_col="MaskPath")
    valid_mask_files = file_scanning(r"G:\Public_dataset\OpenImage\extact_data\mask", file_format="jpg|png|jpeg",sub_scan=True)
    mask_base_names = [os.path.basename(file) for file in valid_mask_files]
    all_info = seg_boxes.loc[mask_base_names]
    class_des = pd.read_csv(class_description_file, names=["code", "name"])
    class_des["code"] = class_des["code"].map(lambda x: x.replace("/", ""))
    class_des = class_des.set_index("code")
    class_des = class_des.to_dict()["name"]
    all_info["LabelName2"] = all_info["LabelName"].map(lambda x :class_des[x.replace('/','')])
    for i in range(len(all_info)):
        image_file = extact_path+"/image/" + all_info.iloc[i].loc["LabelName2"] +"/" + all_info.iloc[i].loc["ImageID"] + ".jpg"
        mask_file = extact_path+"/mask/" + all_info.iloc[i].loc["LabelName2"] +"/" + all_info.iloc[i].name
        box_coordinates = all_info.iloc[i].iloc[3:7]
        os.makedirs(extact_path+"/box_image/" + all_info.iloc[i].loc["LabelName2"], exist_ok=True)
        os.makedirs(extact_path+"/box_mask/" + all_info.iloc[i].loc["LabelName2"], exist_ok=True)
        save_image_path = extact_path+"/box_image/" + all_info.iloc[i].loc["LabelName2"]
        save_mask_path = extact_path+"/box_mask/" + all_info.iloc[i].loc["LabelName2"]
        get_one_seg_area(image_file,mask_file,box_coordinates,save_image_path,save_mask_path,all_info.iloc[i].loc["LabelName"] +"_"+ all_info.iloc[i].loc["BoxID"])

def get_one_seg_area(image_file,mask_file,box_coordinates,save_image_path,save_mask_path,label):

    try:
        image = Image.open(image_file)
        mask = Image.open(mask_file)
        # assert ((image.size == mask.size) and (fabs(image.size[0]/image.size[1] - mask.size[0]/mask.size[1])<0.01))
        if image.size != mask.size:
            mask = mask.resize(image.size)
        box_coordinates = [int(image.size[0]*box_coordinates[0]), int(image.size[1]*box_coordinates[2]),int(image.size[0]*box_coordinates[1]), int(image.size[1]*box_coordinates[3])]
        crop_image = image.crop(box_coordinates)
        crop_mask = mask.crop(box_coordinates)
        crop_image.save(save_image_path+"/crop_"+os.path.basename(image_file).split(".")[0]+"_" + label.replace("/","")+".jpg")
        crop_mask.save(save_mask_path+"/crop_"+os.path.basename(mask_file))
    except AssertionError:
        print(f"!!!!!尺寸不匹配：{os.path.basename(image_file)}_{image.size}\n{os.path.basename(mask_file)}_{mask.size}\n",fabs(image.size[0]/image.size[1] - mask.size[0]/mask.size[1]))
    except FileNotFoundError:
        print("fuck you ")
    # return True
get_seg_area(seg_boxes_file,class_description_file, target)


