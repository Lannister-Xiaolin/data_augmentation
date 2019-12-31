#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from PIL import Image
from cv2 import namedWindow, imshow, waitKey, WINDOW_FREERATIO, destroyAllWindows
from image_augmentation.general.blending import PyramidBlending, DirectBlending


def read_with_rgb(image_file):
    """将非rgb格式图片转成rgb"""
    img = Image.open(image_file)
    if img.mode is not "RGB":
        img = img.convert("RGB")
    return img


def cv_show_image(image_file, window_name="图片"):
    namedWindow(window_name, WINDOW_FREERATIO)  # 0表示压缩图片，图片过大无法显示
    imshow(window_name, image_file)
    k = waitKey(0)  # 无限期等待输入，需要有这个否则会死机
    if k == 27:  # 如果输入ESC退出
        destroyAllWindows()


def blending_one_image(blending_method="direct", background_img_file=None, blending_img_file=None,
                       background_img_array=None, blending_img_array=None,
                       x=None, y=None, x_proportion=0.6, y_proportion=0.6,
                       x_shift=(0.5, 1.5), y_shift=(1.0, 1.9), save_img=""):
    blender = DirectBlending() if blending_method == "direct" else PyramidBlending()
    blending_result, [x, y, x1, y1] = blender.blending_one_image(background_img_file, blending_img_file, background_img_array, blending_img_array,
                               x, y, x_proportion, y_proportion, x_shift, y_shift, save_img)
    return blending_result, [x, y, x1, y1]


def main():
    embedding_img = Image.open(
        r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\3_公开数据集抽取\原始标注文件\hamburger\n07697313_12.JPEG")
    background_img = Image.open(r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\4_背景底图\1.jpg")
    aug = blending_one_image(background_img, embedding_img)
    aug.show()
    # print(embedding_rescale((1000, 1000), (900, 900), x=None, y=None, x_proportion=0.7, y_proportion=0.7))


if __name__ == '__main__':
    main()
