import torch

from avatar_net import AvatarNet
from utils import imload, imsave, maskload

def inference(cfg, args):

    ## Setting inference device
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.TRAINING.USE_CUDA else 'cpu')

    ## Load checkpoint
    check_point = torch.load(args.ckpt)

    ## Load network
    network = AvatarNet()
    network.load_state_dict(check_point['state_dict'])
    network = network.to(device)
    network.eval()

    ## Load target images
    content_img = imload(args.content_path, args.imsize, args.cropsize if args.cropsize>0 else None).to(device)
    # content_img = imload(args.content_path, args.imsize).to(device)
    style_imgs = [imload(style, args.imsize, args.cropsize if args.cropsize>0 else None, args.cencrop).to(device) for style in args.style_path]
    # style_imgs = [imload(style, args.imsize).to(device) for style in args.style_path]
    masks = None
    if args.mask_path:
        masks = [maskload(mask).to(device) for mask in args.mask_path]

     # stylize image
    with torch.no_grad():
        stylized_img =  network(content_img, style_imgs, args.style_strength, args.patch_size, args.patch_stride,
                masks, args.interpolation_weights, False)

    imsave(stylized_img, args.output_path)

