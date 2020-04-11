from xl_tool.xl_io import file_scanning
import os
path = input("请输入文件位置")
jpg = file_scanning(path, file_format="jpg|jpeg|png")
xmls = file_scanning(path,file_format="xml")
for j in jpg:
    if j.replace("jpg","xml") not in xmls:
        os.remove(j)