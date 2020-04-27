import os
import shutil
import json
from PIL import Image
import numpy as np
cat2id = {
    "bacon": 0,
    "barbecued_pork": 1,
    "broccoli": 2,
    "chicken_breast": 3,
    "chicken_brittle_bone": 4,
    "chicken_feet": 5,
    "chicken_wing": 6,
    "chives": 7,
    "corn": 8,
    "drumstick": 9,
    "egg_tart": 10,
    "eggplant": 11,
    "empty": 12,
    "flammulina_velutipes": 13,
    "green_pepper": 14,
    "hamburger_steak": 15,
    "lamb_chops": 16,
    "lamb_kebab": 17,
    "oyster": 18,
    "peanut": 19,
    "pizza": 20,
    "potato": 21,
    "prawn": 22,
    "pumpkin_block": 23,
    "ribs": 24,
    "salmon": 25,
    "saury": 26,
    "sausage": 27,
    "scallion_cake": 28,
    "scallop": 29,
    "shiitake_mushroom": 30,
    "steak": 31,
    "sweet_potato": 32,
    "tilapia": 33,
    "toast": 34,
    "whole_chicken": 35
}
print(dict(map(lambda i: (i[1], i[0]), cat2id.items())))


def serving_request_image_food(image_files, model_name="efficientnetb1",
                               serving_host="http://10.125.31.57:8501/v1/models/{}:predict", target_size=(240, 240)):
    """
    tensorflow.serving食物请求sdk
    Args:
        image_files: 图片文件路径列表，单个文件也必须为列表
        model_name: 模型名称，当前有效模型为默认名称efficientnetb2
        serving_host: 模型请求接口，目前只有默认值有效，多模型部署时，会更改接口
    Returns:
        正常时，返回top3结果，列表，格式如下：
            [{'broccoli': 0.769328654, 'pizza': 0.23008424, 'hamburger': 0.000517699867},
            {'pizza': 0.877195954, 'broccoli': 0.103550591, 'hamburger': 0.0165584981}]
        出现错误时：
            返回错误提示字典
    """
    data = np.stack([np.array(Image.open(image_file).resize(target_size)) / 255.0 for image_file in image_files])
    data = data.tolist()
    cat = dict(map(lambda i: (i[1], i[0]), cat2id.items()))
    data = json.dumps({
        "signature_name": "serving_default",
        "instances": data
    })
    result = requests.post(serving_host.format(model_name),
                           headers={"content-type": "application/json"},
                           data=data)
    # print(json.loads(result.text))
    try:
        predict = np.array(json.loads(result.text)["predictions"])

        labels_top3 = predict.argsort()[:, -3:].tolist()
        prob_top3 = (np.sort(predict)[:, -3:]).tolist()
        # print("fuck ")
        result = [{cat[labels_top3[i][j]]: prob_top3[i][j] for j in range(2, -1, -1)} for i in range(len(labels_top3))]
        return result
    except KeyError:
        print(result)
        return json.loads(result)
import requests
from xl_tool.xl_io import file_scanning
def split_by_data(path=r"G:\food_dataset\s3_userdata\origin",):
    files = file_scanning(path,"jpg")
    for file in files:
        date = os.path.basename(file).split(".")[0].split("_")[-1][:8]
        os.makedirs(f"{path}/{date}",exist_ok=True)
        shutil.move(file, f"{path}/{date}/{os.path.basename(file)}")
split_by_data()

def auto_classifier(path=r"G:\food_dataset\s3_userdata\origin\20200320",target=r"G:\food_dataset\s3_userdata\classified"):
    import tqdm
    files = file_scanning(path, "jpg",sub_scan=True)
    pbar = tqdm.tqdm(files)
    for file in pbar:
        result =sorted(serving_request_image_food([file], "efficientnetb1")[0].items(),key=lambda i:i[1])
        print(result)
        os.makedirs(f"{target}/{result[-1][0]}", exist_ok=True)
        shutil.copy(file, f"{target}/{result[-1][0]}/{os.path.basename(file)}")
auto_classifier()
