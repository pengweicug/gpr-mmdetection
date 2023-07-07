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
    #判定图片是否需要切割，按照2：1规格进行切割，直到不足2h像素为止，再以切割处为中心进行二次切割
    w,h=img.size
    i=1
    while w-h*2*i>0:
      #第一次切片
      n=((i-1)*2*h,0,i*2*h,h)
      temp = img.crop(n)
      #第一次切片图片名为单数
      temp.save(img_path.replace(".jpg", str(2*i-1) + '.jpg'))
      #第二次切片
      if w-h*3*i>=0：
        m=((i-1)*2*h+h,0,i*2*h+h,h)
        temp = img.crop(m)
      else:
        m=((i-1)*2*h+h,0,w,h)
        temp = img.crop(m) 
      #第二次切片图片名为双数
      temp.save(img_path.replace(".jpg", str(2*i) + '.jpg'))
      i=i+1
  #图片缝合函数
  def joint(self,img_path:str):



    
  
  
    
