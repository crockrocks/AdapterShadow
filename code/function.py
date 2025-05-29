import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
from skimage import io
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
import time
import cfg
from tqdm import tqdm
from utils import *
from einops import rearrange
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'sam2')))
import sam2.utils.transforms as samtrans

import shutil
import tempfile

import matplotlib.pyplot as plt
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def train_sam(args, net: nn.Module, optimizer, train_loader, epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0

    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs = pack['image'].cuda()
            masks = pack['mask'].cuda()

            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']

            name = pack['image_meta_dict']['filename_or_obj']
            
            showp = pt

            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if isinstance(pt, tuple) and len(pt) == 2:
                coords_torch, labels_torch = pt
                if coords_torch.shape[-1] == 3:
                    labels_torch = coords_torch[..., 2].to(torch.int)
                    coords_torch = coords_torch[..., :2]
            else:
                coords_torch = torch.as_tensor(pt, dtype=torch.float).cuda()
                if coords_torch.shape[-1] == 3:
                    labels_torch = coords_torch[..., 2].to(torch.int)
                    coords_torch = coords_torch[..., :2]
                elif 'point_labels' in locals():
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int).cuda()
                else:
                    labels_torch = -torch.ones(coords_torch.shape[:-1], dtype=torch.int, device=coords_torch.device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)

            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
            imgs = imgs.float().cuda()
            
            for n, value in net.image_encoder.named_parameters():
                if "Adapter" not in n:
                    value.requires_grad = False

            encoder_output = net.image_encoder(imgs)
            
            if isinstance(encoder_output, dict) and "backbone_fpn" in encoder_output:
                imge = encoder_output["backbone_fpn"][-1]
            else:
                imge = encoder_output

            with torch.no_grad():
                se, de = net.sam_prompt_encoder(points=pt, boxes=None, masks=None, )
                
            print("DEBUG: se shape:", se.shape if hasattr(se, 'shape') else type(se))
            print("DEBUG: de shape:", de.shape if hasattr(de, 'shape') else type(de))
            print("DEBUG: imge shape:", imge.shape if hasattr(imge, 'shape') else type(imge))

            high_res_features = None
            if hasattr(net, 'use_high_res_features_in_sam') and net.use_high_res_features_in_sam:
                fpn = encoder_output["backbone_fpn"]
                high_res_features = [
                    net.sam_mask_decoder.conv_s0(fpn[0]),
                    net.sam_mask_decoder.conv_s1(fpn[1]),
                ]

            if de is not None and hasattr(de, 'shape'):
                if de.shape[-2:] != imge.shape[-2:]:
                    de = F.interpolate(de, size=imge.shape[-2:], mode='bilinear', align_corners=False)
                if de.shape[1] != imge.shape[1]:
                    if not hasattr(net, 'de_proj') or net.de_proj is None or \
                       net.de_proj.in_channels != de.shape[1] or net.de_proj.out_channels != imge.shape[1]:
                        net.de_proj = nn.Conv2d(de.shape[1], imge.shape[1], kernel_size=1).to(de.device)
                    de = net.de_proj(de)

            pred, iou_pred, sam_tokens_out, object_score_logits = net.sam_mask_decoder(
                image_embeddings=imge,
                image_pe=net.sam_prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de, 
                multimask_output=False,
                repeat_image=True,
                high_res_features=high_res_features,
            )

            if pred.shape[-2:] != masks.shape[-2:]:
                pred = F.interpolate(pred, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            loss = lossfunc(pred, masks)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                    vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()

    return loss

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)
    mix_res = (0,0,0,0)
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)

    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32).cuda()
            masksw = pack['mask'].to(dtype = torch.float32).cuda()
            orig_size = pack['orig_size']
            
            if 'pt' not in pack:
                imgsw, pt, masksw = generate_click_prompt(imgsw, masksw)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            imgs = imgsw
            masks = masksw
            showp = pt
            mask_type = torch.float32

            if isinstance(pt, tuple) and len(pt) == 2:
                coords_torch, labels_torch = pt
                if coords_torch.shape[-1] == 3:
                    labels_torch = coords_torch[..., 2].to(torch.int)
                    coords_torch = coords_torch[..., :2]
            else:
                coords_torch = torch.as_tensor(pt, dtype=torch.float).cuda()
                if coords_torch.shape[-1] == 3:
                    labels_torch = coords_torch[..., 2].to(torch.int)
                    coords_torch = coords_torch[..., :2]
                elif 'point_labels' in locals():
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int).cuda()
                else:
                    labels_torch = -torch.ones(coords_torch.shape[:-1], dtype=torch.int, device=coords_torch.device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)

            imgs = imgs.to(dtype = mask_type).cuda()
            with torch.no_grad():
                encoder_output = net.image_encoder(imgs)
                
                if isinstance(encoder_output, dict) and "backbone_fpn" in encoder_output:
                    imge = encoder_output["backbone_fpn"][-1]
                else:
                    imge = encoder_output
                    
                se, de = net.sam_prompt_encoder(points=pt, boxes=None, masks=None)

                print("DEBUG: se shape:", se.shape if hasattr(se, 'shape') else type(se))
                print("DEBUG: de shape:", de.shape if hasattr(de, 'shape') else type(de))
                print("DEBUG: imge shape:", imge.shape if hasattr(imge, 'shape') else type(imge))

                high_res_features = None
                if hasattr(net, 'use_high_res_features_in_sam') and net.use_high_res_features_in_sam:
                    fpn = encoder_output["backbone_fpn"]
                    high_res_features = [
                        net.sam_mask_decoder.conv_s0(fpn[0]),
                        net.sam_mask_decoder.conv_s1(fpn[1]),
                    ]

                if de is not None and hasattr(de, 'shape'):
                    if de.shape[-2:] != imge.shape[-2:]:
                        de = F.interpolate(de, size=imge.shape[-2:], mode='bilinear', align_corners=False)
                    if de.shape[1] != imge.shape[1]:
                        if not hasattr(net, 'de_proj') or net.de_proj is None or \
                           net.de_proj.in_channels != de.shape[1] or net.de_proj.out_channels != imge.shape[1]:
                            net.de_proj = nn.Conv2d(de.shape[1], imge.shape[1], kernel_size=1).to(de.device)
                        de = net.de_proj(de)

                pred, iou_pred, sam_tokens_out, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                    repeat_image=True,
                    high_res_features=high_res_features,
                )
                
                if pred.shape[-2:] != masks.shape[-2:]:
                    pred = F.interpolate(pred, size=masks.shape[-2:], mode='bilinear', align_corners=False)

                tot += lossfunc(pred, masks)

                if args.vis:
                    if ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'
                        vis_image(imgs, pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

                temp = eval_seg(pred, masks, threshold)
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    return tot/ n_val , tuple([a/n_val for a in mix_res])


def test_sam(args, val_loader, net: nn.Module):
    os.makedirs(args.out_dir, exist_ok=True)

    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32).cuda()
            masksw = pack['mask'].to(dtype = torch.float32).cuda()
            orig_size = pack['orig_size']
            
            if 'pt' not in pack:
                imgsw, pt, masksw = generate_click_prompt(imgsw, masksw)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']

            name = pack['image_meta_dict']['filename_or_obj']
            imgs = imgsw
            masks = masksw
            showp = pt
            mask_type = torch.float32

            if isinstance(pt, tuple) and len(pt) == 2:
                coords_torch, labels_torch = pt
                if coords_torch.shape[-1] == 3:
                    labels_torch = coords_torch[..., 2].to(torch.int)
                    coords_torch = coords_torch[..., :2]
            else:
                coords_torch = torch.as_tensor(pt, dtype=torch.float).cuda()
                if coords_torch.shape[-1] == 3:
                    labels_torch = coords_torch[..., 2].to(torch.int)
                    coords_torch = coords_torch[..., :2]
                elif 'point_labels' in locals():
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int).cuda()
                else:
                    labels_torch = -torch.ones(coords_torch.shape[:-1], dtype=torch.int, device=coords_torch.device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)

            imgs = imgs.to(dtype = mask_type).cuda()
            with torch.no_grad():
                encoder_output = net.image_encoder(imgs)
                
                if isinstance(encoder_output, dict) and "backbone_fpn" in encoder_output:
                    imge = encoder_output["backbone_fpn"][-1]
                else:
                    imge = encoder_output
                    
                se, de = net.sam_prompt_encoder(points=pt, boxes=None, masks=None)

                print("DEBUG: se shape:", se.shape if hasattr(se, 'shape') else type(se))
                print("DEBUG: de shape:", de.shape if hasattr(de, 'shape') else type(de))
                print("DEBUG: imge shape:", imge.shape if hasattr(imge, 'shape') else type(imge))

                high_res_features = None
                if hasattr(net, 'use_high_res_features_in_sam') and net.use_high_res_features_in_sam:
                    fpn = encoder_output["backbone_fpn"]
                    high_res_features = [
                        net.sam_mask_decoder.conv_s0(fpn[0]),
                        net.sam_mask_decoder.conv_s1(fpn[1]),
                    ]

                if de is not None and hasattr(de, 'shape'):
                    if de.shape[-2:] != imge.shape[-2:]:
                        de = F.interpolate(de, size=imge.shape[-2:], mode='bilinear', align_corners=False)
                    if de.shape[1] != imge.shape[1]:
                        if not hasattr(net, 'de_proj') or net.de_proj is None or \
                           net.de_proj.in_channels != de.shape[1] or net.de_proj.out_channels != imge.shape[1]:
                            net.de_proj = nn.Conv2d(de.shape[1], imge.shape[1], kernel_size=1).to(de.device)
                        de = net.de_proj(de)

                pred, iou_pred, sam_tokens_out, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )

            orig_size = orig_size.numpy().tolist()[0]
            pred = (F.sigmoid(pred)>0.5).float()
            file_name = pack['mask_path'][0]
            out_name = os.path.join(args.out_dir, file_name)

            prediction = np.uint8(pred.cpu().numpy().squeeze()*255)
            pred = np.array(transforms.Resize((orig_size))(Image.fromarray(prediction).convert('L')))
            Image.fromarray(pred).save(out_name)
            pbar.update()