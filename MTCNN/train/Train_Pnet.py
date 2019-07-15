import sys
sys.path.append('..')
import os 
import torch
from torch.utils.data import Dataset
from Data_Loading import ListDataset
from model.MTCNN_nets import PNet
import time
import copy
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)

train_path = '../data_preprocessing/anno_store/imglist_anno_12.txt'
val_path = '../data_preprocessing/anno_store/imglist_anno_12_val.txt'
batch_size = 64
dataloaders = {'train': torch.utils.data.DataLoader(ListDataset(train_path), batch_size=batch_size, shuffle=True),
               'val': torch.utils.data.DataLoader(ListDataset(val_path), batch_size=batch_size, shuffle=True)}
dataset_sizes = {'train': len(ListDataset(train_path)), 'val': len(ListDataset(val_path))}
print('training dataset loaded with length : {}'.format(len(ListDataset(train_path))))
print('validation dataset loaded with length : {}'.format(len(ListDataset(val_path))))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load the model and weights for initialization
model = PNet(is_train=True).to(device)
model.apply(weights_init)
print("Pnet loaded")

train_logging_file = 'Pnet_train_logging.txt'

optimizer = torch.optim.Adam(model.parameters())
since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_accuracy = 0.0
best_loss = 100

loss_cls = nn.CrossEntropyLoss()
loss_offset = nn.MSELoss()

num_epochs = 16
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs-1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # set model to training mode
        else:
            model.eval()  # set model to evaluate mode

        running_loss, running_loss_cls, running_loss_offset = 0.0, 0.0, 0.0
        running_correct = 0.0
        running_gt = 0.0

        # iterate over data
        for i_batch, sample_batched in enumerate(dataloaders[phase]):

            input_images, gt_label, gt_offset = sample_batched['input_img'], sample_batched[
                'label'], sample_batched['bbox_target']
            input_images = input_images.to(device)
            gt_label = gt_label.to(device)
            # print('gt_label is ', gt_label)
            gt_offset = gt_offset.type(torch.FloatTensor).to(device)
            # print('gt_offset shape is ',gt_offset.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                pred_offsets, pred_label = model(input_images)
                pred_offsets = torch.squeeze(pred_offsets)
                pred_label = torch.squeeze(pred_label)
                # calculate the cls loss
                # get the mask element which >= 0, only 0 and 1 can effect the detection loss
                mask_cls = torch.ge(gt_label, 0)
                valid_gt_label = gt_label[mask_cls]
                valid_pred_label = pred_label[mask_cls]

                # calculate the box loss
                # get the mask element which != 0
                unmask = torch.eq(gt_label, 0)
                mask_offset = torch.eq(unmask, 0)
                valid_gt_offset = gt_offset[mask_offset]
                valid_pred_offset = pred_offsets[mask_offset]

                loss = torch.tensor(0.0).to(device)
                cls_loss, offset_loss = 0.0, 0.0
                eval_correct = 0.0
                num_gt = len(valid_gt_label)

                if len(valid_gt_label) != 0:
                    loss += 0.02*loss_cls(valid_pred_label, valid_gt_label)
                    cls_loss = loss_cls(valid_pred_label, valid_gt_label).item()
                    pred = torch.max(valid_pred_label, 1)[1]
                    eval_correct = (pred == valid_gt_label).sum().item()

                if len(valid_gt_offset) != 0:
                    loss += 0.6*loss_offset(valid_pred_offset, valid_gt_offset)
                    offset_loss = loss_offset(valid_pred_offset, valid_gt_offset).item()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()*batch_size
                running_loss_cls += cls_loss*batch_size
                running_loss_offset += offset_loss*batch_size
                running_correct += eval_correct
                running_gt += num_gt

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_loss_cls = running_loss_cls / dataset_sizes[phase]
        epoch_loss_offset = running_loss_offset / dataset_sizes[phase]
        epoch_accuracy = running_correct / (running_gt + 1e-16)

        print('{} Loss: {:.4f} accuracy: {:.4f} cls Loss: {:.4f} offset Loss: {:.4f}'
              .format(phase, epoch_loss, epoch_accuracy, epoch_loss_cls, epoch_loss_offset))
        with open(train_logging_file, 'a') as f:
            f.write('{} Loss: {:.4f} accuracy: {:.4f} cls Loss: {:.4f} offset Loss: {:.4f}'
                    .format(phase, epoch_loss, epoch_accuracy, epoch_loss_cls, epoch_loss_offset)+'\n')
        f.close()

        # deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best loss: {:4f}'.format(best_loss))

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'pnet_Weights')