# population-TL-model

This source code contains the implementation of the motion model presented in the article: "Predictive online 3D target tracking with population-based generative networks for image-guided radiotherapy"



A novel population-based generative network is proposed to address the problem of 3D target location prediction from partial, 2D image-based surrogates during radiotherapy treatments, thus enabling out-of-plane tracking of treatment targets using images acquired in real-time. The proposed model is trained to simultaneously create a low-dimensional manifold representation of 3D non-rigid deformations and to predict, ahead of time, the motion of the treatment target. The predictive capabilities of the model allow to correct target location errors that can arise due to system latency, using only a baseline static volume of the patient anatomy. Importantly, the method does not require supervised information such as ground truth registration fields, organ segmentation, or anatomical landmarks. 


<img src="https://github.com/lisetvr/population-TL-model/blob/main/model_figure.png" width="420" height="400">

## Cite
If you find this code useful for your research, please cite our [paper](https://doi.org/10.1007/s11548-021-02425-x):
```
@article{romaguera2021predictive,
  title={Predictive online 3D target tracking with population-based generative networks for image-guided radiotherapy},
  author={Romaguera, Liset V{\'a}zquez and Mezheritsky, Tal and Mansour, Rihab and Tanguay, William and Kadoury, Samuel},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  volume={16},
  number={7},
  pages={1213--1225},
  year={2021},
  publisher={Springer}
}


```
