import argparse, json
from yacs.config import CfgNode as CN
from trainer import train
from inference import inference

def generate_default_configs():
    _C = CN()
    _C.DATASET = CN()
    _C.DATASET.DATA_ROOT = ''

    _C.LOSS = CN()
    _C.LOSS.FEATURE_WEIGHT = 0.0
    _C.LOSS.TV_WEIGHT = 0.0

    _C.TRAINING = CN()
    _C.TRAINING.EPOCH = 0
    _C.TRAINING.MAX_ITER = 0
    _C.TRAINING.BATCH_SIZE = 0
    _C.TRAINING.LEARNING_RATE = 0.0
    _C.TRAINING.USE_CUDA = True
    _C.TRAINING.CHECK_PER_ITER = 100

    _C.OUTPUT = CN()
    _C.OUTPUT.CHECKPOINT_ROOT = 'checkpoints/'
    _C.OUTPUT.OUTPUT_ROOT = 'output/'
    _C.OUTPUT.CHECKPOINT_PREFIX = ''

    return _C

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test a network')
    
    # Global arguments
    parser.add_argument('config_file', help='config file path')
    parser.add_argument('mode', help='train/test/inference')
    
    # Arguments for inference
    parser.add_argument('--imsize', type=int, help='Size for resize image during training', default=512)
    parser.add_argument('--cropsize', type=int, help='Size for crop image durning training', default=None)
    parser.add_argument('--cencrop', action='store_true', help='Flag for crop the center rigion of the image, default: randomly crop', default=False)
    parser.add_argument('--layers', type=int, nargs='+', help='Layer indices to extract features', default=[1, 6, 11, 20])
    parser.add_argument('--ckpt', type=str, help="Trained model load path")
    parser.add_argument('--content_path', type=str, help="Test content image path", default='')
    parser.add_argument('--style_path', type=str, nargs='+', help="Test style image path", default='')
    parser.add_argument('--mask_path', type=str, nargs='+', help="Mask image for masked stylization", default=None)
    parser.add_argument('--style-strength', type=float, help='Content vs style interpolation value: 1(style), 0(content)', default=1.0)
    parser.add_argument('--interpolation-weights', type=float, nargs='+', help='Multi-style interpolation weights', default=None)
    parser.add_argument('--patch-size', type=int, help='Size of patch for swap normalized content and style features',  default=3)
    parser.add_argument('--patch-stride', type=int, help='Size of patch stride for swap normalized content and style features',  default=1)

    args = parser.parse_args()
    config_file = args.config_file
    mode = args.mode
    
    cfg = generate_default_configs()

    # Configurations
    print("===== Load configuration =====")
    cfg.merge_from_file(config_file)
    cfg.freeze()
    print ('-----------Configs OK!-------------')
    print (cfg)
    print ('-----------------------------------')

    if mode == 'train':
        train(cfg)
    elif mode == 'inference':
        inference(cfg, args)
