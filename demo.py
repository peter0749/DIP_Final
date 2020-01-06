import os
import sys
import argparse

import cv2
from PIL import Image
import torch

from network import AvatarNet
from utils import _transformer, _normalizer, imload


'''
def imsave(tensor, path):
    denormalize = _normalizer(denormalize=True)
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torchvision.utils.make_grid(tensor)
    torchvision.utils.save_image(denormalize(tensor).clamp_(0.0, 1.0), path)
    return None
'''

def video(args, cfg):
    denormalize = _normalizer(denormalize=True)
    transformer = _transformer()
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.TRAINING.USE_CUDA else 'cpu')
    network = AvatarNet(args.layers)
    check_point = torch.load(args.ckpt)
    network.load_state_dict(check_point['state_dict'])
    network = network.to(device)
    network = network.eval()
    style_img = imload(args.style_image).to(device)
    cap = cv2.VideoCapture(args.input_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            with torch.no_grad():
                content = transformer(Image.fromarray(frame[...,:3][...,::-1])).unsqueeze(0).to(device) # BGR->RGB->PIL.Image
                stylized_img = network(content, [style_img], args.style_strength, args.patch_size, args.patch_stride,
                                        None, args.interpolation_weights, False)[0].cpu() # (3, H, W)
                stylized_img = np.transpose(np.round(denormalize(stylized_img).clamp(0, 1).numpy() * 255).astype(np.uint8), (1,2,0))[...,::-1] # (3, H, W)->(H,W,3)RGB->BGR
                out.write(stylized_img)
    cap.release()
    out.release()

