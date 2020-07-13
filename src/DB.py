# -*- coding: utf-8 -*-
# 数据集处理，保存文件至csv文件格式
from __future__ import print_function

import pandas as pd
import os

DB_dir = 'database'  # 原始数据集文件
DB_csv = 'data.csv' # 创建csv文件


class Database(object):
  def __init__(self):
    self._gen_csv() # 生成csv文件
    self.data = pd.read_csv(DB_csv) # 读取CSV文件
    self.classes = set(self.data["cls"]) # 

  def _gen_csv(self):
    if os.path.exists(DB_csv):
      return
    with open(DB_csv, 'w', encoding='UTF-8') as f:
      f.write("img,cls")
      for root, _, files in os.walk(DB_dir, topdown=False):  # 读取文件路径并将文件路径和图像所属类别写进csv文件
        cls = root.split('/')[-1]
        for name in files:
          if not name.endswith('.jpg'):
            continue
          img = os.path.join(root, name)
          f.write("\n{},{}".format(img, cls))

  def __len__(self):# 返回CSV文件个数，即图像个数
    return len(self.data) 

  def get_class(self): # 返回图像类
    return self.classes

  def get_data(self): # 返回图像信息
    return self.data


if __name__ == "__main__":
  db = Database()
  data = db.get_data()
  classes = db.get_class()

  print("DB length:", len(db))
  print(classes)
