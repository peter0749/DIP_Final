---
title: 'Digital Image Processing'
disqus: hackmd
---

Digital Image Processing
===

## Table of Contents

- [Digital Image Processing](#Digital-Image-Processing)
    - [Requirements](#Requirements)
    - [Downloads](#Downloads)
    - Usage
        - [Arguments](#Arguments)
        - [Configs](#Configs)
    - [Train](#Train)
    - [Inference](#Inference)
    - [Appendix and FAQ](#Appendix-and-FAQ)

## Requirements
- torch (version: 1.2.0)
- torchvision (version: 0.4.0)
- Pillow (version: 6.1.0)
- matplotlib (version: 3.1.1)
- yacs (version: >= 0.1.4)

## Downloads
Clone or download the whole project
```
git clone https://github.com/peter0749/DIP_Final.git
```

Preparing for training data and some of pretrained models.
- [MSCOCO train2014](http://cocodataset.org/#download) is needed to train the network.
- Our pretrained model is released [here](https://drive.google.com/open?id=1USw5v3w6y7iDRynrS2HYzYJ6MIfSzAyb).
- Also needs to download the pretrain embedding GloVe from [here](http://nlp.stanford.edu/data/glove.6B.zip).
- Our exsample images and videos can be found [here](https://drive.google.com/drive/u/1/folders/1zUhVLaoPmyNA89jYhrBw-jrtlLSY6a4-)

## Usage
### Arguments
#### Global arguments
- ```config_file```: Path to the config file
- ```mode```: Run the mode for the model, ```train``` for training the model, and ```inference``` for inference the image through a trained model.
#### Inference arguments
- ```--imsize```: Size for resizing input images (resize shorter side of the image)
- ```--cropsize```: Size for crop input images (crop the image into squares)
- ```--cencrop```: Flag for crop the center reigion of the image (default: randomly crop)
- ```--check-point```: Check point path for loading trained network
- ```--content_path```: Content image path to evalute the network
- ```--style_path```: Style image path to evalute the network
- ```--mask_path```: Mask image path for masked stylization
- ```--style-strength```: Content vs Style interpolation weight (1.0: style, 0.0: content, default: 1.0)
- ```--interpolatoin-weights```: Weights for multiple style interpolation
- ```--patch-size```: Patch size of style decorator (default: 3)
- ```--patch-stride```: Patch stride of style decorator (default: 1)

### Configs
As you can in the ```configs/testing.yaml``` file, there are many hyperparameters used in training the model, so if you want to retrain the model with different settings from us, please feel free to rewrite it.
```yaml
MODEL:
  USE_DATAPARALLEL: True # Use multi-gpu for training
  USABLE_GPUS: [0,1,2,3] # Specify the gpu device numbers you use

DATASET:
  DATA_ROOT: /tmp2/Avatar-Net/dataset/train2014 # Change the data root to the folder you download MSCOCO dataset.

IMG_PROCESSING:
  IMSIZE: 512 # Fix the image size for the first input to the model
  CROPSIZE: 256 # Crop the image size for data augmentation
  CENCROP: False # Whether center crop the image or not.

LOSS:
  FEATURE_WEIGHT: 0.1 # The affect weight on content while training 
  TV_WEIGHT: 1.0 # The affect weight on generalizing the model 

TRAINING:
  MAX_ITER: 8000 # Training iterations
  LEARNING_RATE: 0.001 # Learning rate while training
  BATCH_SIZE: 16 # You can reduce the batch size to meet your gpu memory limitation.
  USE_CUDA: True # Whether use cuda device for training 
  CHECK_PER_ITER: 100 # The time period for training, and do validation on each time step

OUTPUT:
  CHECKPOINT_PREFIX: avatarnet_batch16_iter8000_1.0e-03 # Name for the checkpoint for your training
```


## Train
```bash 
python main.py configs/testing.yaml train
```
## Inference
#### Single style: 
Only use one style image to transfer the style to the content image.
```bash
python main.py configs/testing.yaml inference --ckpt ./checkpoints/{checkpoint} --imsize 512 --cropsize 512 --cencrop --content_path {content image path} --style_path {style image path} --style-strength 1.0
```

#### Multiple style
Use more than two style images to transfer the multi-style to the content image.
```bash
python main.py configs/testing.yaml inference --ckpt ./checkpoints/{checkpoint} --imsize 512 --cropsize 512 --content_path {content image path} --style_path {style image 1 path} {style image 2 path} --interpolation-weights 0.5 0.5
```

#### Multiple style with Mask
Use more than two style images to transfer the corresponded style to the masked region from the original content image.
```bash
python main.py configs/testing.yaml inference --ckpt ./checkpoints/{checkpoint} --imsize 512 --cropsize 512 --content_path {content image path} --style_path {style image 1 path} {style image 2 path} --mask_path {mask image 1 path} {mask image 2 path} --interpolation-weights 1.0 1.0
```


## Appendix and FAQ

:::info
**Find this document incomplete?** Leave a comment!
:::

###### tags: `AvatarNet` `Style Transfer` `Multi-style transfer`
