import argparse, os
import cv2
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
import matplotlib.pyplot as plt

from srresnet import _NetG
from functools import partial
import pickle

parser = argparse.ArgumentParser(description="PyTorch SRResNet Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_srresnet.pth", type=str, help="model path")
parser.add_argument("--video", default="input.mp4", type=str, help="video name")
parser.add_argument("--output", default="output.png", type=str, help="image name")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--fps", default=30.0, type=float, help="fps for output video (check on ffplay first!)")

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
model = _NetG()
ckpt = torch.load(opt.model, pickle_module=pickle)["model"].state_dict()
if opt.cuda:
    model.load_state_dict(ckpt, strict=False)
else:
    model.load_state_dict(ckpt, strict=False, map_location='cpu')

if cuda:
    model = model.cuda()
    im_input = im_input.cuda()
else:
    model = model.cpu()

model = model.eval()

cap = cv2.VideoCapture(args.video)
fps = opt.fps
out = None
with torch.no_grad():
    with tqdm() as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                im_input = frame.astype(np.float32).transpose(2,0,1)
                im_input = im_input.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])
                im_input = Variable(torch.from_numpy(im_input/255.).float())
                out = model(im_input)
                out = out.cpu()
                im_h = out[0].numpy().astype(np.float32)

                im_h = im_h*255.
                im_h[im_h<0] = 0
                im_h[im_h>255.] = 255.
                im_h = im_h.transpose(1,2,0)

                if out is None:
                    img_size_output = tuple(im_h.shape[:2][::-1])
                    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'HFYU'), fps, img_size_output)

                out.write(im_h)
                pbar.update(1)
            else:
                break
cap.release()
out.release()

