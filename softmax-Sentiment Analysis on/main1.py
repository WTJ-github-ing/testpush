# _*_coding:UTF-8_*_
# 时间:2022/5/5 13:53
# 文件名称：main.py.py
# 工具：PyCharm
import numpy
import csv
import random
from feature import Bag,Gram
from comparison_plot import alpha_gradient_plot
import time
start1 = time.perf_counter()

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 数据读取
with open('train.tsv') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    temp = list(tsvreader)#156061行

# 初始化
data = temp[1:] #去头 156060行
max_item=1000  #数据量太大，设置一下在Bag里，只用前1000个样本。

random.seed(2021)
numpy.random.seed(2021)

# 特征提取
bag=Bag(data,max_item)
bag.get_words()
bag.get_matrix()

gram=Gram(data, dimension=2, max_item=max_item)
gram.get_words()
gram.get_matrix()

# 画图
alpha_gradient_plot(bag,gram,10000,10)  # 计算10000次
#alpha_gradient_plot(bag,gram,10000,10)  # 计算100000次

end1 = time.perf_counter()
print("final is in : %s Seconds " % (end1-start1))