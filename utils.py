import os
import numpy as np
import pandas as pd
import sys
import re


#则删除文件夹内的文件
def removeFile(file):
    for i in os.listdir(file):
        os.remove(os.path.join(file,i))

#创建文件路径，如果已存在就删除里面的文件
def CreateFile(filename, flagPath="", mask=False):
    flagPath=""
    if mask:
        flagPath="mask"
    try:
        os.makedirs("./"+flagPath + filename + "ValData")
        os.makedirs("./"+flagPath + filename + "TrainData")
    except FileExistsError:
        #removeFile("./"+flagPath + filename + "ValData")
        #removeFile("./"+flagPath + filename + "TrainData")
        print("文件夹已存在")
        pass

