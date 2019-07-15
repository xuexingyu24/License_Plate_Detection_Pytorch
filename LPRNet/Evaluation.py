#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:07:10 2019

@author: xingyu
"""

from model.LPRNET import LPRNet, CHARS
from model.STN import STNet
from data.load_data import LPRDataLoader, collate_fn
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import torchvision
import matplotlib.pyplot as plt

def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.numpy().transpose((1,2,0))
    inp = 127.5 + inp/0.0078125
    inp = inp.astype('uint8') 
    inp = inp[:,:,::-1]
    return inp

def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        dataset = LPRDataLoader([args.img_dirs], args.img_size)   
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn) 
        imgs, labels, lengths = next(iter(dataloader))
        
        input_tensor = imgs.cpu()
        transformed_input_tensor = STN(imgs.to(device)).cpu()
        
        in_grid = convert_image(torchvision.utils.make_grid(input_tensor))
        out_grid = convert_image(torchvision.utils.make_grid(transformed_input_tensor))
        
        # Plot the results side-by-side
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')
        
        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

def decode(preds, CHARS):
    # greedy decode
    pred_labels = list()
    labels = list()
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        pred_label = list()
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = pred_label[0]
        for c in pred_label: # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)
        
    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)
    
    return labels, pred_labels

def eval(lprnet, STN, dataloader, dataset, device):
    
    lprnet = lprnet.to(device)
    STN = STN.to(device)
    TP = 0
    for imgs, labels, lengths in dataloader:   # img: torch.Size([2, 3, 24, 94])  # labels: torch.Size([14]) # lengths: [7, 7] (list)
        imgs, labels = imgs.to(device), labels.to(device)
        transfer = STN(imgs)
        logits = lprnet(transfer) # torch.Size([batch_size, CHARS length, output length ])
    
        preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
        _, pred_labels = decode(preds, CHARS)  # list of predict output

        start = 0
        for i, length in enumerate(lengths):
            label = labels[start:start+length]
            start += length
            if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                TP += 1
            
    ACC = TP / len(dataset) 
    
    return ACC
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LPR Evaluation')
    parser.add_argument('--img_size', default=(94, 24), help='the image size')
    parser.add_argument('--img_dirs', default="./data/ccpd_weather", help='the images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--batch_size', default=128, help='batch size.')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=args.dropout_rate)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load('weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))
#    checkpoint = torch.load('saving_ckpt/lprnet_Iter_023400_model.ckpt')
#    lprnet.load_state_dict(checkpoint['net_state_dict'])
    lprnet.eval() 
    print("LPRNet loaded")
    
#    torch.save(lprnet.state_dict(), 'weights/Final_LPRNet_model.pth')
    
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
#    checkpoint = torch.load('saving_ckpt/stn_Iter_023400_model.ckpt')
#    STN.load_state_dict(checkpoint['net_state_dict'])
    STN.eval()
    print("STN loaded")
    
#    torch.save(STN.state_dict(), 'weights/Final_STN_model.pth')
    
    dataset = LPRDataLoader([args.img_dirs], args.img_size)   
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn) 
    print('dataset loaded with length : {}'.format(len(dataset)))
    
    ACC = eval(lprnet, STN, dataloader, dataset, device)
    print('the accuracy is {:.2f} %'.format(ACC*100))
    
    visualize_stn()

