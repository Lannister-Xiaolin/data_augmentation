#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from copy import deepcopy

from PIL import Image
from PIL.Image import NEAREST, BILINEAR, LANCZOS
import random
import cv2
import numpy as np, sys


class BlendingBase:
    def __init__(self):
        pass

    @classmethod
    def max_blending_size(cls, background_size, x, y, x_proportion, y_proportion):
        """计算最大允许嵌入图片大小"""
        if x:
            max_x = min(int(background_size[0] * x_proportion), background_size[0] - x)
        else:
            max_x = int(background_size[0] * x_proportion)
        if y:
            max_y = min(int(background_size[1] * y_proportion), background_size[1] - y)
        else:
            max_y = int(background_size[1] * y_proportion)
        return max_x, max_y

    @classmethod
    def calculate_rescale(cls, background_size, blending_size, x, y, x_proportion, y_proportion):
        """计算融合嵌入图片缩放系数
        Args:
            background_size:背景图片尺寸，宽 X 高
            blending_size:背景图片尺寸，宽 X 高
            x:嵌入位置,宽度方向，可为空
            y:嵌入位置，高度方向，可为空
            x_proportion:宽度方向最大占比
            y_proportion:高度方向最大占比
        """
        max_x, max_y = cls.max_blending_size(background_size, x, y, x_proportion, y_proportion)
        x_scale = 1 if max_x > blending_size[0] else max_x / blending_size[0]
        y_scale = 1 if max_y > blending_size[1] else max_y / blending_size[1]
        scale = min(x_scale, y_scale)
        if scale < 0.99:
            pass
            # print("嵌入图像{}过大，需进行缩放，缩放比例：{:.2f}".format(blending_size, scale))
        return scale

    @classmethod
    def save_image(cls, img: Image.Image, save):
        try:
            img.save(save)
        except Exception as e:
            print("增强图片存储失败！:".format(save))
            pass

    @classmethod
    def valid_size_confirm(cls, image_sizes, region_size, sp_dis, min_rescale):
        """
        该函数暂时无用
        Args:
            image_sizes:尺寸列表
            region_size: 嵌入区域大小
            sp_dis ： 组合方式
            min_rescale: 最大缩放区域
        """
        if sp_dis in (1, 5):
            return (sum([size[0] for size in image_sizes]) * min_rescale) < region_size[0]
        elif sp_dis(2, 6):
            return (sum([size[1] for size in image_sizes]) * min_rescale) < region_size[1]
        elif sp_dis in (3, 4):
            for i in range(2, -1, -1):
                one_combine = image_sizes[i]
                valid_x = (one_combine[0] + max([image_sizes[j][0] for j in range(3) if j != i])) < region_size[0]
                valid_y = (one_combine[1] < region_size[1] and sum([image_sizes[j][1] for j in range(3) if
                                                                    j != i])) < region_size[1]
                if valid_x and valid_y:
                    return True
            return False
        else:
            return False
        pass

    @classmethod
    def paralleling_caculate(cls, image_sizes, region_size, sp_dis):
        # print("输入尺寸：", image_sizes, "\t区域大小:", region_size)
        first, second = (0, 1) if sp_dis in (1, 5) else (1, 0)
        first_sum = sum([size[first] for size in image_sizes])
        if first_sum < region_size[first]:
            rescales = [1 if size[second] < region_size[second] else region_size[second] / size[second] for size in
                        image_sizes]
        else:
            # print("图片过大，进行缩放！")
            rescale = region_size[first] * 0.95 / first_sum
            rescales = [rescale if size[second] * rescale < region_size[second] else region_size[second] / size[second]
                        for size in
                        image_sizes]
        image_sizes = [[int(c * rescale) for c in size] for (rescale, size) in zip(rescales, image_sizes)]
        margin_first_max = region_size[first] - sum([size[first] for size in image_sizes])
        margin_seconds = [region_size[second] - size[second] for size in image_sizes]
        positions = []
        margin_first_remain = margin_first_max
        first_start = 0
        for size, margin_second in zip(image_sizes, margin_seconds):
            position = [None, None, None, None]
            margin = int(margin_first_remain * random.random())
            # print(margin)
            margin_first_remain = margin_first_remain - margin
            position_first = margin + first_start
            position_second = int(margin_second * random.random())
            first_start = position_first + size[first]
            position[first], position[second], position[first + 2], position[second + 2] = \
                position_first, position_second, first_start, position_second + size[second]
            positions.append(position)
        # print("缩放比例：", rescales, "\n缩放后大小：", image_sizes)
        return image_sizes, rescales, positions

    @classmethod
    def calculate_mul_blending_positions(cls, image_sizes, region_size, sp_dis):
        """计算嵌入位置
            1，2为并列排列
            3为混合排列，根据尺寸随机切分成上下或者左右2块，然后再做并列排列（暂未完成）
        """
        if sp_dis in (1, 5, 2, 6):
            return cls.paralleling_caculate(image_sizes, region_size, sp_dis)
        else:
            pass


class DirectBlending(BlendingBase):
    """
    直接对图片进行融合嵌入，主要包括单目标融合和多目标融合
    """

    def __init__(self):
        super().__init__()

    def blending_one_image(self, background_img_file=None, blending_img_file=None,
                           background_img_array=None, blending_img_array=None, x=None, y=None,
                           x_proportion=0.7, y_proportion=0.7, x_shift=(0.5, 1.5),
                           y_shift=(1, 1.9), save_img="", resample=BILINEAR):
        """单张图片融合嵌入
        下一版完善功能：
            最小比例占比限制
            长宽比扰动
        Args:
            background_img_file:背景图片
            blending_img_file:嵌入图片
            background_img_array:背景图片数组
            blending_img_array:融合图片数组
            x: 嵌入位置x坐标，如为空，根据中心位置计算
            y:同上
            x_shift:针对x中心位置的摆动
            y_shift:针对y中心位置的摆动
            x_proportion: 嵌入图片x长度与背景图片x长度的最大占比
            y_proportion: 嵌入图片y长度与背景图片y长度的最大占比
            resample: 图像插值方式
            save_img:是否保持图片，是则输入路径，否则为空字符
        Returns:
            数组：Image格式的融合图片 + 融合后的bounding box坐标
                blending image,[xmin,ymin,xmax,ymax]
        """
        background_img = Image.open(background_img_file) if background_img_array is None else \
            Image.fromarray(background_img_array)
        blending_img = Image.open(blending_img_file) if blending_img_array is None else \
            Image.fromarray(blending_img_array)
        size = blending_img.size
        scale = self.calculate_rescale(background_img.size, blending_img.size, x, y, x_proportion, y_proportion)
        new_size = (int(size[0] * scale), int(size[1] * scale))
        blending_img = blending_img if 0.99 < scale < 1.01 else blending_img.resize(new_size, resample=resample)
        if not x:
            x = int(((background_img.size[0] - blending_img.size[0]) // 2) * random.uniform(*x_shift))
        if not y:
            y = int(((background_img.size[1] - blending_img.size[1]) // 2) * random.uniform(*y_shift))
        background_img.paste(blending_img, (x, y))

        blending_result = background_img
        if save_img:
            self.save_image(blending_result, save_img)
        # blending_result.show()
        return blending_result, [x, y, x + blending_img.size[0], y + blending_img.size[1]]

    def blending_images(self, background_img_file, blending_img_files, sp_dis, save_img="",blending_region=None):
        """
        多目标融合：
        多目标融合的位置分布主要包括以下几种方式：
            左右并列排列  1
            上下并列排列  2
            混合排列   3
        """
        background_img = Image.open(background_img_file)
        # print("背景图片大小：", background_img.size)
        blending_imgs = [Image.open(blending_img_file) for blending_img_file in blending_img_files]
        image_sizes = [img.size for img in blending_imgs]
        blending_region = blending_region if blending_region else (0, 0, background_img.size[0], background_img.size[1])
        x, y = blending_region[0], blending_region[1]
        region_size = blending_region[2] - blending_region[0], blending_region[3] - blending_region[1]
        image_sizes, rescales, positions = self.calculate_mul_blending_positions(image_sizes, region_size, sp_dis)
        positions = [(position[0] + x, position[1] + y, position[2] + x, position[3] + y) for position in positions]
        # print("嵌入坐标位置：", positions)
        blending_imgs = [blending_img.resize(size) for (blending_img, size) in zip(blending_imgs, image_sizes)]
        for blending_img, position in zip(blending_imgs, positions):
            # print(blending_img.size, position)
            background_img.paste(blending_img, (position[0], position[1]))
        if save_img:
            self.save_image(background_img, save_img)
        # background_img.show()
        return background_img,image_sizes, positions


class PyramidBlending(BlendingBase):
    def __init__(self, num_pyramid=6):
        super().__init__()
        self.num_pyramid = num_pyramid
        pass

    def generate_gaussian_pyramid(self, image_array):
        """计算高斯金字塔"""
        array = image_array.copy()
        gaussian_pyramid = [array]
        for i in range(self.num_pyramid):
            array = cv2.pyrDown(array)
            gaussian_pyramid.append(array)
        return gaussian_pyramid

    def generate_laplacian_pyramid(self, gaussian_pyramid):
        """计算高斯金字塔"""
        laplacian_pyramid = [gaussian_pyramid[self.num_pyramid - 1]]
        for i in range(self.num_pyramid - 1, 0, -1):
            ge = cv2.pyrUp((gaussian_pyramid[i]),
                           dstsize=(gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0]))
            laplacian = cv2.subtract(gaussian_pyramid[i - 1], ge)
            laplacian_pyramid.append(laplacian)
        return laplacian_pyramid

    def surround_reconstruct(self, lp_background, lp_blending, normalize_x, normalize_y):
        ls = []
        for la, lb in zip(lp_background, lp_blending):
            temp = deepcopy(la)
            x, y = int(normalize_x * la.shape[1]), int(normalize_y * la.shape[0])
            temp[y:y + lb.shape[0], x:x + lb.shape[1]] = lb
            ls.append(temp)
        ls_array = ls[0]
        for i in range(1, self.num_pyramid):
            ls_array = cv2.pyrUp(ls_array)
            ls_array = cv2.add(ls_array, ls[i])
        return ls_array

    def rescale_blending(self, blending, background, x, y, x_proportion, y_proportion, x_shift, y_shift):
        blending_size = (blending.shape[1], blending.shape[0],)
        background_size = (background.shape[1], background.shape[0])
        scale = self.calculate_rescale(background_size, blending_size, x, y, x_proportion, y_proportion)
        new_size_cv = (int(blending_size[0] * scale), int(blending_size[1] * scale))
        blending = blending if 0.99 < scale < 1.01 else cv2.resize(blending, dsize=new_size_cv)
        if not x:
            x = int(((background_size[0] - blending.shape[1]) // 2) * random.uniform(*x_shift))
        if not y:
            y = int(((background_size[1] - blending.shape[0]) // 2) * random.uniform(*y_shift))
        return x, y, blending

    def blending_one_image(self, background_img_file=None, blending_img_file=None,
                           background_img_array=None, blending_img_array=None,
                           x=None, y=None, x_proportion=0.6, y_proportion=0.6,
                           x_shift=(0.5, 1.5), y_shift=(1.0, 1.9), save_img=""):
        """基于opencv的金字塔融合"""
        background = cv2.imdecode(np.fromfile(background_img_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED) \
            if background_img_array is None else background_img_array[:, :, ::-1]
        blending = cv2.imdecode(np.fromfile(blending_img_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED) \
            if blending_img_array is None else blending_img_array[:, :, ::-1]
        # print(blending.shape)
        x, y, blending = self.rescale_blending(blending, background, x, y,
                                               x_proportion, y_proportion, x_shift, y_shift)
        # print(blending.shape)

        g_background = self.generate_gaussian_pyramid(background)
        g_blending = self.generate_gaussian_pyramid(blending)
        lp_background = self.generate_laplacian_pyramid(g_background)
        lp_blending = self.generate_laplacian_pyramid(g_blending)
        blending_result = Image.fromarray(
            self.surround_reconstruct(lp_background, lp_blending,
                                      x / background.shape[1], y / background.shape[0])[:, :, ::-1])
        if save_img:
            self.save_image(blending_result, save_img)
        # blending_result.show()
        return blending_result, [x, y, x + blending.shape[1], y + blending.shape[0]]


def test_blending_pyramid():
    background_img_file = r"D:\1.jpg"
    blending_img_file = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\public_dataset\food101\food\pizza" \
                        r"\702165.jpg"
    blender = PyramidBlending(6)
    blender.blending_one_image(background_img_file, blending_img_file)


def test_blending_direct():
    background_img_file = r"D:\1.jpg"
    blending_img_file = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\public_dataset\food101\food\pizza" \
                        r"\702165.jpg"
    blender = DirectBlending()
    blender.blending_one_image(background_img_file, blending_img_file)


def test_mul_blending():
    a = DirectBlending()
    backgroud_file = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\4_背景底图\2.jpg"
    blending_files = [
        r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\5_抽取目标\broccoli\000000000009.jpg",
        r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\5_抽取目标\broccoli\000000000009.jpg",
        r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\5_抽取目标\orange\000000000009.jpg"]
    print(a.blending_images(backgroud_file, blending_files, 2, (20, 80, 620, 440)))


def main():
    # test_blending_pyramid()
    # test_blending_direct()
    test_mul_blending()


if __name__ == '__main__':
    main()
