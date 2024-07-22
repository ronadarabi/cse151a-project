# CSE 151A Group Project

## Collaborators 
* Justin 
* Rona 
* Diego
* Jose
* Logan  
* Daniel

## Preprocessing and Training

A new dataset was acquired after testing indicated there wasn't enough data to learn a colorization function properly, so we downloaded [this](https://www.kaggle.com/datasets/shmalex/instagram-images) dataset, consisting of 1,211,625 scraped Instagram posts. The images were then cropped to 512x512 where possible, using `resize.py` leaving us with approximately 1,000,000 images to train on. We aslo moved from conversion to LAB format to just using RGB as converting from jpg to lab would inflate the storage requirements ~8x. We trained a simple convolutional neural network using `train_conv.py` which seemed to begin to learn a colorization function. Training results were logged to [wandb.ai](https://wandb.ai/danielpwarren/convnet-colorizer)

Future models will change in architecture, possibly UNet-style convolutional networks or generative adversarial networks.

Our first model begins to learn a colorization function, but slightly distorts the output image. One possible solution is to use a UNet which adds residual connections to previous convolution layers.