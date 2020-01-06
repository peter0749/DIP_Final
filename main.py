import argparse, json
from yacs.config import CfgNode as CN
from trainer import train


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
    parser.add_argument('config_file', help='config file path')
    parser.add_argument('mode', help='train/test/inference')
    
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
    else:
        test()
