import os
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
import shutil
from pathlib import Path
import time
import sys
from torch.utils.data import DataLoader

import cfg
from utils import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'sam2')))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import function
from dataloader import ShadowDataLoader, DataFiles, DataLoadPreprocess, preprocessing_transforms, random_click

class CustomDataFiles(DataFiles):
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        root_dataset = ''
        if args.dataset_name == 'sbu':
            root_dataset = args.root_sbu
            root_dataset = Path(root_dataset)
            
            train_dir = root_dataset.joinpath('SBUTrain4KRecoveredSmall', 'ShadowImages')
            if not train_dir.exists():
                print(f"Warning: Training directory {train_dir} does not exist!")
                print(f"Checking for alternative paths...")
                alt_train_dir = root_dataset.joinpath('SBUTrain4KRecoveredSmall')
                if alt_train_dir.exists():
                    subdirs = [d for d in alt_train_dir.iterdir() if d.is_dir()]
                    print(f"Found subdirectories: {[d.name for d in subdirs]}")
                    for subdir in subdirs:
                        if any(subdir.glob('*.jpg')) or any(subdir.glob('*.png')):
                            print(f"Found images in {subdir}")
                
            train_imgs = list(train_dir.rglob('*.jpg'))
            print(f"Found {len(train_imgs)} training images")
            
            mask_dir = root_dataset.joinpath('SBUTrain4KRecoveredSmall', 'ShadowMasks')
            if not mask_dir.exists():
                print(f"Warning: Mask directory {mask_dir} does not exist!")
                alt_mask_dirs = list(root_dataset.joinpath('SBUTrain4KRecoveredSmall').glob('*Mask*'))
                if alt_mask_dirs:
                    print(f"Found alternative mask directories: {[d.name for d in alt_mask_dirs]}")
                    mask_dir = alt_mask_dirs[0]
            
            self.train_jpgs = []
            for img_path in train_imgs:
                mask_path = mask_dir.joinpath(img_path.name[:-3] + 'png')
                if mask_path.exists():
                    self.train_jpgs.append((img_path, mask_path))
            
            test_dir = root_dataset.joinpath('SBU-Test', 'ShadowImages')
            if not test_dir.exists():
                print(f"Warning: Test directory {test_dir} does not exist!")
                alt_test_dir = root_dataset.joinpath('SBU-Test')
                if alt_test_dir.exists():
                    subdirs = [d for d in alt_test_dir.iterdir() if d.is_dir()]
                    print(f"Found test subdirectories: {[d.name for d in subdirs]}")
            
            test_imgs = list(test_dir.rglob('*.jpg'))
            print(f"Found {len(test_imgs)} test images")
            
            test_mask_dir = root_dataset.joinpath('SBU-Test', 'ShadowMasks')
            if not test_mask_dir.exists():
                print(f"Warning: Test mask directory {test_mask_dir} does not exist!")
                alt_test_mask_dirs = list(root_dataset.joinpath('SBU-Test').glob('*Mask*'))
                if alt_test_mask_dirs:
                    print(f"Found alternative test mask directories: {[d.name for d in alt_test_mask_dirs]}")
                    test_mask_dir = alt_test_mask_dirs[0]
            
            self.test_jpgs = []
            for img_path in test_imgs:
                mask_path = test_mask_dir.joinpath(img_path.name[:-3] + 'png')
                if mask_path.exists():
                    self.test_jpgs.append((img_path, mask_path))

            print('==> using SBU, train {%d}, test{%d}'%(len(self.train_jpgs), len(self.test_jpgs)))
        else:
            super().__init__(args)
            
    def get_pairs(self):
        if self.dataset_name == 'sbu':
            return self.train_jpgs, self.test_jpgs
        else:
            return super().get_pairs()

class CustomShadowDataLoader(ShadowDataLoader):
    def __init__(self, args):
        self.args = args
        self.data_files = CustomDataFiles(args)
        self.train_files, self.val_files = self.data_files.get_pairs()
        
    def get_dataloader(self, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(self.args, mode, self.train_files, transform=preprocessing_transforms(mode))
            data = DataLoader(self.training_samples, self.args.bs,
                               shuffle=True,
                               num_workers=self.args.bs,
                               pin_memory=True)
            return data
        elif mode == 'val' or mode == 'test':
            self.testing_samples = DataLoadPreprocess(self.args, mode, self.val_files, transform=preprocessing_transforms(mode))
            data = DataLoader(self.testing_samples, 1,
                               shuffle=False,
                               num_workers=0,
                               pin_memory=False)
            return data
        else:
            print('mode should be one of \'train, val, test\'. Got {}'.format(mode))
            
        return None

args = cfg.parse_args()

set_seed()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
sam2_model_registry = {
    'hiera_b+': {
        'config': 'sam2/sam2_hiera_b+.yaml',
        'checkpoint': os.path.join(os.path.dirname(__file__), 'sam2', 'checkpoints', 'sam2.1_hiera_base_plus.pt'),  
    },
    'hiera_l': {
        'config': 'sam2/sam2_hiera_l.yaml',
        'checkpoint': os.path.join(os.path.dirname(__file__), 'sam2', 'checkpoints', 'sam2.1_hiera_large.pt'),  
    },
    'hiera_s': {
        'config': 'sam2/sam2_hiera_s.yaml',
        'checkpoint': os.path.join(os.path.dirname(__file__), 'sam2', 'checkpoints', 'sam2.1_hiera_small.pt'),  
    },
    'hiera_t': {
        'config': 'sam2/sam2_hiera_t.yaml',
        'checkpoint': os.path.join(os.path.dirname(__file__), 'sam2', 'checkpoints', 'sam2.1_hiera_tiny.pt'),  
    },
}
model_key = 'hiera_b+'
entry = sam2_model_registry[model_key]
net = build_sam2(
    config_file=entry['config'],
    ckpt_path=entry['checkpoint'],
    device='cuda',
    mode='eval',
)

if args.pretrain:
    pass

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

shadow_dataset = CustomShadowDataLoader(args)
train_loader = shadow_dataset.get_dataloader('train')
val_loader = shadow_dataset.get_dataloader('val')

writer = SummaryWriter(log_dir=os.path.join(args.path_helper['log_path'], args.net, datetime.now().isoformat()))
checkpoint_path = os.path.join(args.path_helper['ckpt_path'], '{net}-{epoch}-{type}.pth')

best_acc = 0.0
best_tol = 1e4
for epoch in range(args.epochs):
    if True:
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, train_loader, epoch, writer, vis = args.vis)
        logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == args.epochs-1:
            tol, (eiou, edice) = function.validation_sam(args, val_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            sd = net.state_dict()

            if tol < best_tol:
                best_tol = tol
                is_best = True

                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
            else:
                is_best = False

writer.close() 