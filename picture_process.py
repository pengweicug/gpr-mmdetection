# Copyright (c) pengwei. All rights reserved.

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import mmcv
from PIL import Image

#超长图片切割缝合类

class LongPictureProcess:
  #图片分片函数
  def slicing(self,img_path:str):
    #读取图片
    img=Image.open(img_path)
    #img = mmcv.imread(img_path, channel_order='rgb')
    #判定图片是否需要切割，按照2：1规格进行切割，直到不足2000像素为止，再以切割处为中心进行二次切割
    w,h=img.size
    #第一次切片
    i=1
    while w-h*2*i>0:
      n=((i-1)*2*h,0,i*2*h,h)
      temp = img.crop(n)
      temp.save(path.replace(".jpg", str(i - 1) + '.jpg'))
      i=i+1
    #第二次切片
    j=1
    while w-h*2*i>0:
      n=((i-1)*2*h,0,i*2*h,h)
      temp = img.crop(n)
      temp.save(path.replace(".jpg", str(i - 1) + '.jpg'))
      i=i+1
  #图片缝合函数
  def joint(self,img_path:str):



    
  
  
    
