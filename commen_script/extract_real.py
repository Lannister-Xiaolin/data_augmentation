from xl_tool.xl_io import file_scanning
import os
import shutil
target = r"G:\food_dataset\5_抽取目标\混合"
root = r"G:\food_dataset\5_抽取目标\待筛选_真实"
cats = os.listdir(root)
old = os.listdir(target)
for cat in cats:
    if cat  in  old:
        print(cat)
        shutil.rmtree(f"{root}/{cat}")
    # for i in os.listdir(f"{root}/{cat}"):
    #     if "network" in i:
    #         continue
    #     else:
    #         # shutil.
    #         # os.makedirs(f"{target}/{cat}/{i}", exist_ok=True)
    #         # shutil.copytree(f"{root}/{cat}/{i}", f"{target}/{cat}/")
    #         shutil.rmtree(f"{root}/{cat}/{i}")

# file_scanning(r"")
