from xl_tool.augmentation.image.general.blending import SegBlend
from xl_tool.xl_io import file_scanning
from random import choice
import os
from tqdm import tqdm

def test_seg_mask():
    blender = SegBlend()
    seg_path = r"G:\Public_dataset\OpenImage\extact_data\box_mask_clean"
    save_path = r"G:\Public_dataset\OpenImage\extact_data\seg_blend"
    background_img_files = file_scanning(r"G:\Public_dataset\OpenImage\extact_data\background", file_format="jpg")
    labels = os.listdir(seg_path)
    for label in labels:
        mask_img_files = file_scanning(seg_path + "/" + label, file_format="png", sub_scan=True)
        os.makedirs(save_path + "/" + label, exist_ok=True)
        if label.lower() == "pizza":
            x_proportion = 0.6
            y_proportion = 0.6
        else:
            x_proportion = 0.18
            y_proportion = 0.18
        for mask_img_file in tqdm(mask_img_files):
            try:
                background_img_file = choice(background_img_files)
                blending_img_file = get_image(mask_img_file)
                save_file = save_path + "/" + label + "/" + "aug_seg_blend_" + os.path.basename(blending_img_file)
                blender.blending_one_image(background_img_file, blending_img_file, mask_img_file, x_proportion=x_proportion,
                                           y_proportion=y_proportion,
                                           save_img=save_file, x_shift=(0.9, 1.1), y_shift=(0.9,
                                                                                            1.1))
            except:
                pass
        # background_img_file = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\4_背景底图\1.jpg"
    # blending_img_file = r"F:\Large_dataset\segmentation\mask\crop_00c1d2697deeaea1_m014j1m.jpg"
    # mask_img_file =  r"F:\Large_dataset\segmentation\image\crop_00bb5720a7ba062e_m014j1m_b26b27f6.png"
    # blender.blending_one_image(background_img_file,blending_img_file,mask_img_file,x_proportion=0.2,
    #                            save_img=r"F:\Large_dataset\segmentation\segmentation_blend\crop_00bb5720a7ba062e_m014j1m_b26b27f6.jpg")


def get_mask(file):
    return file.replace("jpg", "png").replace("box_image_clean", "box_mask")
def get_image(file):
    return file.replace("png", "jpg").replace("box_mask_clean", "box_image_clean")

test_seg_mask()
