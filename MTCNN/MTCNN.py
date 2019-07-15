import sys
import os
sys.path.append(os.getcwd())
import argparse
import torch
from model.MTCNN_nets import PNet, ONet
import math
import numpy as np
from utils.util import *
import cv2
import time

def create_mtcnn_net(image, mini_lp_size, device, p_model_path=None, o_model_path=None):

    bboxes = np.array([])

    if p_model_path is not None:
        pnet = PNet().to(device)
        pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))
        pnet.eval()

        bboxes = detect_pnet(pnet, image, mini_lp_size, device)

    if o_model_path is not None:
        onet = ONet().to(device)
        onet.load_state_dict(torch.load(o_model_path, map_location=lambda storage, loc: storage))
        onet.eval()

        bboxes = detect_onet(onet, image, bboxes, device)

    return bboxes

def detect_pnet(pnet, image, min_lp_size, device):

    # start = time.time()

    thresholds = 0.6 # lp detection thresholds
    nms_thresholds = 0.7

    # BUILD AN IMAGE PYRAMID
    height, width, channel = image.shape
    min_height, min_width = height, width

    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image
    scales = []

    factor_count = 0
    while min_height > min_lp_size[1] and min_width > min_lp_size[0]:
        scales.append(factor ** factor_count)
        min_height *= factor
        min_width *=factor
        factor_count += 1

    # it will be returned
    bounding_boxes = []

    with torch.no_grad():
        # run P-Net on different scales
        for scale in scales:
            sw, sh = math.ceil(width * scale), math.ceil(height * scale)
            img = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_LINEAR)
            img = torch.FloatTensor(preprocess(img)).to(device)
            offset, prob = pnet(img)
            probs = prob.cpu().data.numpy()[0, 1, :, :]  # probs: probability of a face at each sliding window
            offsets = offset.cpu().data.numpy()  # offsets: transformations to true bounding boxes
            # applying P-Net is equivalent, in some sense, to moving 12x12 window with stride 2
            stride, cell_size = (2,5), (12,44)
            # indices of boxes where there is probably a lp
            # returns a tuple with an array of row idx's, and an array of col idx's:
            inds = np.where(probs > thresholds)

            if inds[0].size == 0:
                boxes = None
            else:
                # transformations of bounding boxes
                tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
                offsets = np.array([tx1, ty1, tx2, ty2])
                score = probs[inds[0], inds[1]]
                # P-Net is applied to scaled images
                # so we need to rescale bounding boxes back
                bounding_box = np.vstack([
                    np.round((stride[1] * inds[1] + 1.0) / scale),
                    np.round((stride[0] * inds[0] + 1.0) / scale),
                    np.round((stride[1] * inds[1] + 1.0 + cell_size[1]) / scale),
                    np.round((stride[0] * inds[0] + 1.0 + cell_size[0]) / scale),
                    score, offsets])
                boxes = bounding_box.T
                keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
                boxes[keep]

            bounding_boxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        
        if bounding_boxes != []:
            bounding_boxes = np.vstack(bounding_boxes)
            keep = nms(bounding_boxes[:, 0:5], nms_thresholds)
            bounding_boxes = bounding_boxes[keep]
        else:
            bounding_boxes = np.zeros((1,9))
        # use offsets predicted by pnet to transform bounding boxes
        bboxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5],  x1, y1, x2, y2, score

        bboxes[:, 0:4] = np.round(bboxes[:, 0:4])

        # print("pnet predicted in {:2.3f} seconds".format(time.time() - start))

        return bboxes

def detect_onet(onet, image, bboxes, device):

    # start = time.time()

    size = (94,24)
    thresholds = 0.8  # face detection thresholds
    nms_thresholds = 0.7
    height, width, channel = image.shape

    num_boxes = len(bboxes)
    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bboxes, width, height)

    img_boxes = np.zeros((num_boxes, 3, size[1], size[0]))

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3))

        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = \
            image[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        # resize
        img_box = cv2.resize(img_box, size, interpolation=cv2.INTER_LINEAR)

        img_boxes[i, :, :, :] = preprocess(img_box)

    img_boxes = torch.FloatTensor(img_boxes).to(device)
    offset, prob = onet(img_boxes)
    offsets = offset.cpu().data.numpy()  # shape [n_boxes, 4]
    probs = prob.cpu().data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds)[0]
    bboxes = bboxes[keep]
    bboxes[:, 4] = probs[keep, 1].reshape((-1,))  # assign score from stage 2
    offsets = offsets[keep]
    
    bboxes = calibrate_box(bboxes, offsets)
    keep = nms(bboxes, nms_thresholds, mode='min')
    bboxes = bboxes[keep]
    bboxes[:, 0:4] = np.round(bboxes[:, 0:4])
    # print("onet predicted in {:2.3f} seconds".format(time.time() - start))

    return bboxes

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MTCNN Demo')
    parser.add_argument("--test_image", dest='test_image', help=
    "test image path", default="test/28.jpg", type=str)
    parser.add_argument("--scale", dest='scale', help=
    "scale the iamge", default=1, type=int)
    parser.add_argument('--mini_lp', dest='mini_lp', help=
    "Minimum lp to be detected. derease to increase accuracy. Increase to increase speed",
                        default=(50, 15), type=int)

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = cv2.imread(args.test_image)
    image = cv2.resize(image, (0, 0), fx = args.scale, fy = args.scale, interpolation=cv2.INTER_CUBIC)

    start = time.time()

    bboxes = create_mtcnn_net(image, args.mini_lp, device, p_model_path='weights/pnet_Weights', o_model_path='weights/onet_Weights')

    print("image predicted in {:2.3f} seconds".format(time.time() - start))

    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, :4]
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        
    image = cv2.resize(image, (0, 0), fx = 1/args.scale, fy = 1/args.scale, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()