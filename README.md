# License_Plate_Detection_Pytorch
This is a two stage lightweight and robust license plate recognition in MTCNN and LPRNet using Pytorch. [MTCNN](https://arxiv.org/abs/1604.02878v1) is a very well-known real-time detection model primarily designed for human face recognition. The MTCNN network is modified for license plate detection. [LPRNet](https://arxiv.org/abs/1806.10447), another real-time end-to-end DNN, is utilized for the subsquent recognition. This network is attributed by its superior performance with low computational cost without preliminary character segmentation. The [Spatial Transformer Layer](https://arxiv.org/abs/1506.02025) is embeded in this work to allow a better characteristics for recognition. Here is the illustration of the proposed pipeline:

<p align="left">
<img src="https://github.com/xuexingyu24/License_Plate_Detection_Pytorch/tree/master/test/pipeline.png">
</p>

## MTCNN
The modified MTCNN structure is presented as below. Only proposal net (Pnet) and output net (Onet) are used in this work since it is found that skipping Rnet will not hurt the accuracy in this case.  The Onet accepts 24(height) x 94(width) BGR image which is consistent with input for LPRNet. 

<p align="left">
<img src="https://github.com/xuexingyu24/License_Plate_Detection_Pytorch/tree/master/test/MTCNN.png">
</p>
