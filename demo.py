import os
import sys

import numpy as np
import cv2
from PIL import Image
import torch

from tqdm import tqdm
from avatar_net import AvatarNet
from utils import _transformer, _normalizer, imload
from collections import deque


def noise_estimation(path, imsize=512, quatile=0.25):
    cap = cv2.VideoCapture(path)
    hist_rgb = np.zeros((3, 511), dtype=np.float32) # [-255, ..., 0, +255]
    N = 0
    last_frame = None
    with tqdm() as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                h, w = frame.shape[:2]
                h = int(h * imsize / w)
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                frame = frame.astype(np.float32)
                if not last_frame is None:
                    diff = (frame - last_frame).reshape(-1, 3) # (N, 3)
                    rgb_q = np.quantile(diff, quatile, axis=0, keepdims=True) # (1, 3)
                    diff = diff[np.all(diff<rgb_q, axis=1),:].astype(np.int32)
                    N += diff.shape[0]
                    for c in range(3):
                        val = np.clip(diff[:,c]+255, 0, 510)
                        hist_rgb[c,val] += 1
                last_frame = frame
                pbar.update(1)
            else:
                break
    cap.release()
    hist_rgb = hist_rgb / N # sample from sample
    return hist_rgb


def noise_reduction(img, pdf, ite=2000):
    img = img.astype(np.float32)
    f_ite = float(ite)
    for c in range(3):
        p = pdf[c]
        n = (np.random.choice(len(p), img.shape[0]*img.shape[1], replace=True)-255).reshape(img.shape[0], img.shape[1]) # sampled noise pattern
        img[:,:,c] += (n/f_ite)
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    return img


def video(cfg, args):
    denormalize = _normalizer(denormalize=True)
    transformer = _transformer(args.imsize)
    transformer_style = _transformer(args.imsize, args.cropsize if args.cropsize>0 else None, args.cencrop)
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.TRAINING.USE_CUDA else 'cpu')
    network = AvatarNet()
    check_point = torch.load(args.ckpt)
    network.load_state_dict(check_point['state_dict'])
    network = network.to(device)
    network = network.eval()
    style_imgs_original = deque([imload(style, args.imsize, args.cropsize if args.cropsize>0 else None, args.cencrop).to(device) for style in args.style_path])
    max_len = args.video_max_len
    alpha = args.video_alpha
    style_imgs_frame = deque(maxlen=max_len)
    style_img_style_interp = [1.0]
    masks = None
    #if args.mask_path:
    #    masks = [maskload(mask).to(device) for mask in args.mask_path]
    cap = cv2.VideoCapture(args.input_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = None
    stylized_img = None
    with tqdm() as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                with torch.no_grad():
                    content = transformer(Image.fromarray(frame[...,:3][...,::-1])).unsqueeze(0).to(device) # BGR->RGB->PIL.Image
                    style_imgs_frame_interp = []
                    if not stylized_img is None:
                        style_imgs_frame.append(transformer_style(Image.fromarray(stylized_img[...,:3][...,::-1])).unsqueeze(0).to(device))
                        interp_val = style_img_style_interp[0]
                        for _ in range(len(style_imgs_frame)):
                            interp_val *= alpha
                            style_imgs_frame_interp.insert(0, interp_val)
                    interp_weights = np.array(style_img_style_interp+style_imgs_frame_interp)
                    interp_weights = interp_weights / interp_weights.sum()
                    stylized_img = network(content, style_imgs_original+style_imgs_frame, args.style_strength, args.patch_size, args.patch_stride,
                                            masks, interp_weights, False)[0].cpu() # (3, H, W)
                    stylized_img = np.transpose(np.round(denormalize(stylized_img).clamp(0, 1).numpy() * 255).astype(np.uint8), (1,2,0))[...,::-1] # (3, H, W)->(H,W,3)RGB->BGR
                    if out is None:
                        img_size_output = tuple(stylized_img.shape[:2][::-1])
                        out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'HFYU'), fps, img_size_output)
                    out.write(stylized_img)
                    pbar.update(1)
            else:
                break
    cap.release()
    out.release()


def denoise(cfg, args):
    noise_pattern = noise_estimation(args.input_video, args.imsize)
    cap = cv2.VideoCapture(args.input_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = None
    with tqdm() as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                denoised_image = noise_reduction(frame, noise_pattern)
                if out is None:
                    img_size_output = tuple(denoised_image.shape[:2][::-1])
                    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'HFYU'), fps, img_size_output)
                out.write(denoised_image)
                pbar.update(1)
            else:
                break
    cap.release()
    out.release()

'''
def denoise(cfg, args):
    cap = cv2.VideoCapture(args.input_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = None
    maxlen = 3
    mid_frame = (maxlen-1)//2
    noisy = deque(maxlen=maxlen)
    with tqdm() as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            noisy.append(frame)
            if ret == True:
                if frame_width>args.imsize:
                    w = frame_width
                    h = int(frame_height * args.imsize / w)
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                if len(noisy)<maxlen:
                    denoised_image = cv2.fastNlMeansDenoisingColored(noisy[-1],None,10,10,7,21)
                else:
                    denoised_image = cv2.fastNlMeansDenoisingColoredMulti(noisy, mid_frame, len(noisy), None, 3, 7, 21)
                if out is None:
                    img_size_output = tuple(denoised_image.shape[:2][::-1])
                    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'HFYU'), fps, img_size_output)
                out.write(denoised_image)
                pbar.update(1)
            else:
                break
    cap.release()
    out.release()
'''
