# population-TL-model

This source code contains the implementation of the motion model presented in the article: "Predictive online 3D target tracking with population-based generative networks for image-guided radiotherapy"



A novel population-based generative network is proposed to address the problem of 3D target location prediction from partial, 2D image-based surrogates during radiotherapy treatments, thus enabling out-of-plane tracking of treatment targets using images acquired in real-time. The proposed model is trained to simultaneously create a low-dimensional manifold representation of 3D non-rigid deformations and to predict, ahead of time, the motion of the treatment target. The predictive capabilities of the model allow to correct target location errors that can arise due to system latency, using only a baseline static volume of the patient anatomy. Importantly, the method does not require supervised information such as ground truth registration fields, organ segmentation, or anatomical landmarks. 

![alt text](https://github.com/lisetvr/population-TL-model/blob/main/model_figure.png?raw=true|width=100px)
