import torch
import os
from tqdm import tqdm
from tqdm import trange

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from auto_encoder import Encoder
from avatar_net import AvatarNet
from utils import ImageFolder, imsave, lastest_arverage_value

def calc_tv_loss(x):
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss

def checkpoint_file(cfg, iteration):
    return 'checkpoints/{}.mdl.checkpoint{}'.format(cfg.OUTPUT.CHECKPOINT_PREFIX, iteration)

def save_model(checkpt_prefix, params):
    print("=> saving '{}'".format(checkpt_prefix))
    torch.save(params, checkpt_prefix)

def train(cfg):

    ### Set the training device
    device = torch.device('cuda:{:d}'.format(cfg.MODEL.USABLE_GPUS[0]) if torch.cuda.is_available() and cfg.TRAINING.USE_CUDA else 'cpu')

    ### Tensorboard settings
    log_dir = cfg.OUTPUT.OUTPUT_ROOT + cfg.OUTPUT.CHECKPOINT_PREFIX
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    ### Get the training data
    print ('Getting training data ...')
    dataset = ImageFolder(cfg.DATASET.DATA_ROOT, cfg.IMG_PROCESSING.IMSIZE, cfg.IMG_PROCESSING.CROPSIZE, cfg.IMG_PROCESSING.CENCROP)

    ### Get models
    print ('Getting models ...')
    base_network = AvatarNet().to(device)

    ### Get optimizer
    ## Freeze the whole encode during training
    for param in base_network.encoder.parameters():
        param.requires_grad = False
    opt = torch.optim.Adam(base_network.decoder.parameters(), lr=cfg.TRAINING.LEARNING_RATE)

    network = torch.nn.DataParallel(base_network, device_ids=cfg.MODEL.USABLE_GPUS) if cfg.MODEL.USE_DATAPARALLEL else base_network

    ### Get losses
    print ('Setting training loss, optimizer, and loading loss network (VGG19)')
    loss_network = Encoder().to(device)
    if cfg.MODEL.USE_DATAPARALLEL:
        loss_network = torch.nn.DataParallel(loss_network, device_ids=cfg.MODEL.USABLE_GPUS)
    mse_loss = torch.nn.functional.mse_loss # torch.nn.MSELoss(reduction='mean').to(device)
    loss_seq = {'total':[], 'image':[], 'feature':[], 'tv':[]}

    ### Start training
    print ('Start training')
    dataloader = DataLoader(dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True)
    max_iter = cfg.TRAINING.MAX_ITER
    trange = tqdm(enumerate(range(max_iter)), total=max_iter, desc='Train')
    for i, iteration in trange:

        ## Get input images with batch
        input_image = next(iter(dataloader)).to(device)

        ## Get output images with batch
        output_image = network(input_image, [input_image], train=True)

        ## Calculate losses
        total_loss = 0

        ## Image reconstruction loss
        image_loss = mse_loss(output_image, input_image)
        loss_seq['image'].append(image_loss.item())
        total_loss += image_loss

        ## Feature reconstruction loss
        input_features = loss_network(input_image)
        output_features = loss_network(output_image)
        feature_loss = 0
        for output_feature, input_feature in zip(output_features, input_features):
            feature_loss += mse_loss(output_feature, input_feature)
        loss_seq['feature'].append(feature_loss.item())
        total_loss += feature_loss * cfg.LOSS.FEATURE_WEIGHT

        ## Total variation loss
        tv_loss = calc_tv_loss(output_image)
        loss_seq['tv'].append(tv_loss.item())
        total_loss += tv_loss * cfg.LOSS.TV_WEIGHT

        loss_seq['total'].append(total_loss.item())

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if (iteration + 1) % cfg.TRAINING.CHECK_PER_ITER == 0:
            # Show current lossese onto terminal
            avg_image_loss = lastest_arverage_value(loss_seq['image'])
            avg_feature_loss = lastest_arverage_value(loss_seq['feature'])
            avg_tv_loss = lastest_arverage_value(loss_seq['tv'])
            avg_total_loss = lastest_arverage_value(loss_seq['total'])
            trange.set_postfix(image_loss=avg_image_loss, feature_loss=avg_feature_loss, tv_loss=avg_tv_loss, total_loss=avg_total_loss)

            writer.add_scalar('Average training loss', avg_total_loss, global_step=iteration+1)

            # Save model checkpoint
            if not os.path.exists(cfg.OUTPUT.CHECKPOINT_ROOT):
                os.mkdir(cfg.OUTPUT.CHECKPOINT_ROOT)

            save_model(checkpoint_file(cfg, iteration+1),
                {'iteration': iteration+1,
                'state_dict': base_network.state_dict(),
                'loss_seq': loss_seq})

