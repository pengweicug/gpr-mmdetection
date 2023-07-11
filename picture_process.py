# Copyright (c) pengwei. All rights reserved.
import math
from PIL import Image
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS


#超长图片切割缝合
#图片分片函数
def slicing(self,img_path:str,config_file,checkpoint_file):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
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
      #temp.save(img_path.replace(".jpg", str(2*i-1) + '.jpg'))
      first_result = first_result+self.adjust_coordinate(temp,i,h,model)
      #第二次切片
      if w-h*3*i>=0:
        m=((i-1)*2*h+h,0,i*2*h+h,h)
      else:
        m=((i-1)*2*h+h,0,w,h)
      temp = img.crop(m)
      #第二次切片图片名为双数
      #temp.save(img_path.replace(".jpg", str(2*i) + '.jpg'))
      second_result = second_result + self.adjust_coordinate(temp, i, h,model)
      i=i+1
    result=merge_target(first_result, second_result)
    pic_visualizer(result, img, model)


#将切片后的像素坐标调整到原始图片的像素坐标
def adjust_coordinate(temp,i,h,model):
    result = inference_detector(model, temp)
    pred_instances = result.pred_instances
    bboxes = pred_instances.bboxes
    for box in bboxes:
      box[0] = box[0] + (i - 1) * 2 * h
      box[2] = box[2] + (i - 1) * 2 * h
    return result

#将两次切片的识别结果进行合并
def merge_target(first_result,second_result,w,h):
  first_pred_instances = first_result.pred_instances
  first_bboxes = first_pred_instances.bboxes
  second_pred_instances = second_result.pred_instances
  second_bboxes = second_pred_instances.bboxes
  #对第二次切片图片识别的结果保留穿越切缝处的病害,舍弃非切缝处的病害
  count=math.ceil(w / (2 * h))
  for second_box in second_bboxes:
      if(count>1):
          cutline_list=range(2*h,2*h,2*h*(count-1))
      for cutline in cutline_list:
        if(second_box[0]<cutline and second_box[2]>cutline):
            new_second_bboxes = new_second_bboxes.put(second_box)
  #将第二次切片保留的病害与第一次切片图片识别的病害进行相交判断,若相交则舍弃第一次切片相应的病害
  for first_box in first_bboxes:
    for second_box in new_second_bboxes:
      if((second_box[0]<first_box[0]<second_box[2] and second_box[1]<first_box[1]<second_box[3])
              or (second_box[0]<first_box[2]<second_box[2] and second_box[1]<first_box[3]<second_box[3])):
        first_bboxes.remove(first_box)
  #将两次切片结果合并
  result=first_result+second_result
  return result


#结果可视化
def pic_visualizer(result,img,model):
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    visualizer.add_datasample(
      'result',
      img,
      data_sample=result,
      draw_gt=False,
      wait_time=0,
    )
    visualizer.show()

if __name__ == '__main__':
  slicing()









    
  
  
    
