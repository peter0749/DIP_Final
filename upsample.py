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
parser.add_argument("--image", default="input.png", type=str, help="image name")
parser.add_argument("--output", default="output.png", type=str, help="image name")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

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

im_l = cv2.imread(opt.image, cv2.IMREAD_COLOR)[:,:,:3][:,:,::-1]

im_input = im_l.astype(np.float32).transpose(2,0,1)
im_input = im_input.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])
im_input = Variable(torch.from_numpy(im_input/255.).float())

if cuda:
    model = model.cuda()
    im_input = im_input.cuda()
else:
    model = model.cpu()

model = model.eval()
with torch.no_grad():
    start_time = time.time()
    out = model(im_input)
    elapsed_time = time.time() - start_time

out = out.cpu()

im_h = out[0].numpy().astype(np.float32)

im_h = im_h*255.
im_h[im_h<0] = 0
im_h[im_h>255.] = 255.
im_h = im_h.transpose(1,2,0)

print("It takes {}s for processing".format(elapsed_time))

cv2.imwrite(opt.output, im_h[:,:,::-1].astype(np.uint8))
