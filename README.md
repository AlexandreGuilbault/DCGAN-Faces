# DCGAN-Faces
Pytorch implementation of a DCGAN to generate faces from CelebA dataset

## Datasets

CelebA dataset from : https://www.kaggle.com/jessicali9530/celeba-dataset#img_align_celeba.zip is being used.

## Dependencies

Pytorch     0.4.1\
torchvision 0.2.1\
matplotlib  2.2.2\
numpy       1.15.1

## Outputs

Some examples of the generated images after 250 epochs of training.

![Example 1](https://github.com/AlexandreGuilbault/DCGAN-Faces/blob/master/img/1.png?raw=true)
![Example 2](https://github.com/AlexandreGuilbault/DCGAN-Faces/blob/master/img/13.png?raw=true)
![Example 3](https://github.com/AlexandreGuilbault/DCGAN-Faces/blob/master/img/16.png?raw=true)
![Example 4](https://github.com/AlexandreGuilbault/DCGAN-Faces/blob/master/img/17.png?raw=true)


## Issues

Catastrophic forgetting seemed to happen on a 64x64 bits image size with the full dataset. Reduced to 32x32 image size and a dataset of 30,000 images on the last version.

![Catastrophic forgetting?](https://github.com/AlexandreGuilbault/DCGAN-Faces/blob/master/img/CatastrophicForgetting.gif?raw=true)

## Acknowledgments

Some ideas have been taken from Soumith Chintala tips and tricks for the GAN file : https://github.com/soumith/ganhacks
The Aligned CelebA dataset has been taken from Jessica Li Kaggle dataset : https://www.kaggle.com/jessicali9530/celeba-dataset#img_align_celeba.zip
