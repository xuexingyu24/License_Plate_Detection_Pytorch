"""
    generate positive, negative, positive images whose size are 24*24 from Pnet and feed into RNet
"""
import sys
sys.path.append('..')
import cv2
import os
import numpy as np
from utils.util import*
import torch
import random
from imutils import paths
from MTCNN import create_mtcnn_net

img_dir = "../data_set/ccpd_val"
pos_save_dir = "../data_set/val/24/positive"
part_save_dir = "../data_set/val/24/part"
neg_save_dir = "../data_set/val/24/negative"

if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

# store labels of positive, negative, part images
f1 = open(os.path.join('anno_store', 'pos_24_val.txt'), 'w')
f2 = open(os.path.join('anno_store', 'neg_24_val.txt'), 'w')
f3 = open(os.path.join('anno_store', 'part_24_val.txt'), 'w')

# anno_file: store labels of the wider face training data
img_paths = []
img_paths += [el for el in paths.list_images(img_dir)]
random.shuffle(img_paths)
num = len(img_paths)
print("%d pics in total" % num)

image_size = (94, 24)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
for annotation in img_paths:
    im_path = annotation
    print(im_path)
    
    basename = os.path.basename(im_path)
    imgname, suffix = os.path.splitext(basename)
    imgname_split = imgname.split('-')
    rec_x1y1 = imgname_split[2].split('_')[0].split('&')
    rec_x2y2 = imgname_split[2].split('_')[1].split('&')  
    x1, y1, x2, y2 = int(rec_x1y1[0]), int(rec_x1y1[1]), int(rec_x2y2[0]), int(rec_x2y2[1])
    
    boxes = np.zeros((1,4), dtype=np.int32)
    boxes[0,0], boxes[0,1], boxes[0,2], boxes[0,3] = x1, y1, x2, y2

    image = cv2.imread(im_path)

    bboxes = create_mtcnn_net(image, 50, device, p_model_path='../train/pnet_Weights', r_model_path=None, o_model_path=None)
    dets = np.round(bboxes[:, 0:4])

    if dets.shape[0] == 0:
        continue

    img = cv2.imread(im_path)
    idx += 1

    height, width, channel = img.shape

    for box in dets:
        x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
        width = x_right - x_left + 1
        height = y_bottom - y_top + 1

        # ignore box that is too small or beyond image border
        if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
            continue

        # compute intersection over union(IoU) between current box and all gt boxes
        Iou = IoU(box, boxes)
        cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
        resized_im = cv2.resize(cropped_im, image_size, interpolation=cv2.INTER_LINEAR)

        # save negative images and write label
        if np.max(Iou) < 0.3 and n_idx < 3.2*p_idx+1:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            f2.write(save_file + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
        else:
            # find gt_box with the highest iou
            idx_Iou = np.argmax(Iou)
            assigned_gt = boxes[idx_Iou]
            x1, y1, x2, y2 = assigned_gt

            # compute bbox reg label
            offset_x1 = (x1 - x_left) / float(width)
            offset_y1 = (y1 - y_top) / float(height)
            offset_x2 = (x2 - x_right) / float(width)
            offset_y2 = (y2 - y_bottom) / float(height)

            # save positive and part-face images and write labels
            if np.max(Iou) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1

            elif np.max(Iou) >= 0.4 and d_idx < 1.2*p_idx + 1:
                save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1

    print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

f1.close()
f2.close()
f3.close()







