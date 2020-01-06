# DIP_Final

## Train
```bash
python main.py configs/testing.yaml train
```

## Inference
-------------
#### Single style
```bash
python main.py configs/testing.yaml inference --ckpt ./checkpoints/{checkpoint} --imsize 512 --cropsize 512 --cencrop --content_path {content image path} --style_path {style image path} --style-strength 1.0
```
#### Multiple style
```bash
python main.py configs/testing.yaml inference --ckpt ./checkpoints/{checkpoint} --imsize 512 --cropsize 512 --content_path {content image path} --style_path {style image 1 path} {style image 2 path} --interpolation-weights 0.5 0.5
```

#### Multiple style with Mask
```bash
python main.py configs/testing.yaml inference --ckpt ./checkpoints/{checkpoint} --imsize 512 --cropsize 512 --content ./sample_images/content/blonde_girl.jpg --style ./sample_images/style/mondrian.jpg ./sample_images/style/abstraction.jpg --mask ./sample_images/mask/blonde_girl_mask1.jpg ./sample_images/mask/blonde_girl_mask2.jpg --interpolation-weights 1.0 1.0
```
