import sys

from utils.dataloaders import *
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device #, time_synchronized
import argparse
import os
import shutil
from models.experimental import attempt_load
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from xml.etree import ElementTree as ET
import warnings

warnings.filterwarnings('ignore')


# 定义一个创建一级分支object的函数
def create_object(root, xi, yi, xa, ya, obj_name):  # 参数依次，树根，xmin，ymin，xmax，ymax
    # 创建一级分支object
    _object = ET.SubElement(root, 'object')
    # 创建二级分支
    name = ET.SubElement(_object, 'name')
    # print(obj_name)
    name.text = str(obj_name)
    pose = ET.SubElement(_object, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(_object, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(_object, 'difficult')
    difficult.text = '0'
    # 创建bndbox
    bndbox = ET.SubElement(_object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = '%s' % xi
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = '%s' % yi
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = '%s' % xa
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = '%s' % ya


# 创建xml文件的函数
def create_tree(sources, image_name, h, w):
    imgdir = sources.split('/')[-1]
    # 创建树根annotation
    annotation = ET.Element('annotation')
    # 创建一级分支folder
    folder = ET.SubElement(annotation, 'folder')
    # 添加folder标签内容
    folder.text = (imgdir)

    # 创建一级分支filename
    filename = ET.SubElement(annotation, 'filename')
    filename.text = image_name

    # 创建一级分支path
    path = ET.SubElement(annotation, 'path')

    path.text = '{}/{}'.format(sources, image_name)  # 用于返回当前工作目录

    # 创建一级分支source
    source = ET.SubElement(annotation, 'source')
    # 创建source下的二级分支database
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    # 创建一级分支size
    size = ET.SubElement(annotation, 'size')
    # 创建size下的二级分支图像的宽、高及depth
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'

    # 创建一级分支segmented
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'
    return annotation


def detect(opt, model, img, img0):
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # img = img_transpose(img0, imgsz, 32)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(
        pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
    return det


def main(opt):
    source, weights, imgsz, = opt.source, opt.weights, opt.img_size

    # Initialize
    device = select_device(opt.device)

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, device=None, inplace=True, fuse=True)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # print(dataset)

    # images_list = os.listdir(source)
    # images_style = ['.jpg', '.png', '.bmp']
    # images_list = [x for x in images_list if x[-4:] in images_style]
    # print(images_list)

    for path, img, im0s, cap, vid_cap in dataset:
        image_name = os.path.split(path)[-1]
        # print('path:', path)

        # 检测饲料袋
        boxes = detect(opt, model, img, im0s)
        # print(len(boxes))
        (h, w) = im0s.shape[:2]
        annotation = create_tree(source, image_name, h, w)
        # print(annotation)

        for box in boxes:
            if float(box[4]) > opt.conf_thres:
                x1, y1, x2, y2, label_id = int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[5])
                label = names[int(label_id)]
                # print(x1, y1, x2, y2, label)
                create_object(annotation, x1, y1, x2, y2, label)

        tree = ET.ElementTree(annotation)
        annotation_path_root = source.replace(source.split('/')[-1], 'annotations')
        tree.write('{}/{}.xml'.format(annotation_path_root, image_name[:-4]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default=r'E:/Python_Learn_Source/YOLO/yolov5_7.0_Attention_Multiple/runs/train/yolov5s/weights/best_730.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default=r'C:/Users/17616/Desktop/VOCData/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.05, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)
    with torch.no_grad():
        main(args)



