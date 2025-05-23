import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from kornia.morphology import erosion

import cfg
# from models.sam import SamPredictor, sam_model_registry
# from models_eff.sam import SamPredictor, sam_model_registry
# from models_fuse.sam import SamPredictor, sam_model_registry
# from models_nf.sam import SamPredictor, sam_model_registry
# from models_df.sam import SamPredictor, sam_model_registry
from models_dense.sam import SamPredictor, sam_model_registry
from models_exp.sam import get_sam2_model
from models_exp.sam import SAM2ImagePredictor

from dataloader import ShadowDataLoader, DataFiles
from utils import *

from PIL import Image
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def test_sam(args, val_loader, net: nn.Module, out_dir=None):
    # create output directory
    if out_dir is None:
        out_dir = f'./logs_{args.exp_name}/version_0'
        # os.makedirs(args.out_dir, exist_ok=True)
    else:
        os.makedirs(out_dir, exist_ok=True)

    prefix = 'ucf_' if True else ''
    coarse_dir = os.path.join(out_dir, prefix+'cmasks')
    mask_dir = os.path.join(out_dir, prefix+'masks')
    cmp_dir = os.path.join(out_dir, prefix+'cmps')
    os.makedirs(coarse_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(cmp_dir, exist_ok=True)
    # eval mode
    net.eval()

    kernel = torch.ones(5,5).cuda()
    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch

    accuracy = 0.0
    _TP = 0.0
    _TN = 0.0
    _Np = 0.0
    _Nn = 0.0
    total = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32).cuda()
            masksw = pack['mask'].to(dtype = torch.float32).cuda()
            orig_img = pack['orig_image']

            orig_size = pack['orig_size']
            
            file_name = pack['mask_path'][0]
            mask_out_name = os.path.join(mask_dir, file_name)
            coarse_out_name = os.path.join(coarse_dir, file_name)
            # if os.path.exists(mask_out_name):
            #     continue
            
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

            imgs = imgs.to(dtype = mask_type).cuda()
            with torch.no_grad():
                if args.type=='gen_mask':
                    # mask from CNN
                    pred_mask, cnn_features = net.mask_generator(imgs)
                    pred_mask_erosion = erosion(pred_mask, kernel)
                    # upsample
                    mask_size = (args.mask_size, args.mask_size)
                    if pred_mask_erosion.shape[-2:]!=mask_size:
                        pred_mask_erosion = F.interpolate(pred_mask_erosion, mask_size, mode='bilinear', align_corners=True)
                        
                    se, de = net.prompt_encoder(points=None, boxes=None, masks=pred_mask_erosion, )

                elif args.type=='gt_pt':
                    cnn_features = None
                    if point_labels[0] != -1:
                        point_coords = pt
                        coords_torch = torch.as_tensor(point_coords, dtype=torch.float).cuda()
                        labels_torch = torch.as_tensor(point_labels, dtype=torch.int).cuda()
                        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                        pt = (coords_torch, labels_torch)

                    se, de = net.prompt_encoder(points=pt, boxes=None, masks=None, )

                elif args.type=='gen_pt':
                    # topk points
                    pred_mask, cnn_features = net.mask_generator(imgs)
                    m_b, m_c, m_h, m_w = pred_mask.shape
                    pred_mask_clone = pred_mask.clone().view(m_b, m_c, -1)
                    pts = []
                    _, indices = torch.topk(pred_mask_clone, args.npts, 2)
                    hs = torch.div(indices, m_w, rounding_mode='floor')
                    ws = indices % m_w
                    coords_torch = torch.cat((hs, ws), 1).permute(0, 2, 1)
                    labels_torch = torch.ones(m_b, args.npts)
                    pt = (coords_torch, labels_torch)

                    se, de = net.prompt_encoder(points=pt, boxes=None, masks=None, )

                elif args.type=='gen_mask_pt':
                    # topk points
                    pred_mask, cnn_features = net.mask_generator(imgs)

                    # mask
                    pred_mask_erosion = erosion(pred_mask, kernel)
                    mask_size = (args.mask_size, args.mask_size)
                    if pred_mask_erosion.shape[-2:]!=mask_size:
                        pred_mask_erosion = F.interpolate(pred_mask_erosion, mask_size, mode='bilinear', align_corners=True)

                    # points
                    m_b, m_c, m_h, m_w = pred_mask.shape
                    pred_mask_clone = pred_mask.clone().view(m_b, m_c, -1)
                    pts = []
                    _, indices = torch.topk(pred_mask_clone, args.npts, 2)
                    hs = torch.div(indices, m_w, rounding_mode='floor')
                    ws = indices % m_w
                    coords_torch = torch.cat((hs, ws), 1).permute(0, 2, 1)
                    labels_torch = torch.ones(m_b, args.npts)
                    pt = (coords_torch, labels_torch)

                    se, de = net.prompt_encoder(points=pt, boxes=None, masks=pred_mask_erosion, )
                
                imge= net.image_encoder(imgs, cnn_features)
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                )

            # save images
            orig_size = orig_size.numpy().tolist()[0]
            pred = (torch.sigmoid(pred)>0.5).float()
            out_name = os.path.join(cmp_dir, file_name)

            # save pred coarse images
            
            pred_mask_gen = (torch.sigmoid(pred_mask)>0.5).float()
            pred_mask_gen2 = np.uint8(pred_mask_gen.cpu().numpy().squeeze()*255)
            pred_mask_gen2 = np.array(transforms.Resize((orig_size))(Image.fromarray(pred_mask_gen2).convert('L')))
            Image.fromarray(pred_mask_gen2).save(coarse_out_name)

            prediction = np.uint8(pred.cpu().numpy().squeeze()*255)
            pred = np.array(transforms.Resize((orig_size))(Image.fromarray(prediction).convert('L')))
            
            Image.fromarray(pred).save(mask_out_name)

            # pbar.update()
            
            # rgb_img = Image.fromarray((imgs*255).cpu().squeeze().permute(1,2,0).numpy().astype(np.uint8))
            # rgb_img = np.array(transforms.Resize((orig_size))(rgb_img))
            orig_img = (orig_img.squeeze()*255).numpy().astype(np.uint8)
            masksw = np.array(transforms.Resize((orig_size))(Image.fromarray((masksw>0.5).cpu().squeeze().numpy()).convert('L')))

            h, w = orig_size
            n_tp, n_tn, n_p, n_n, _ = cal_acc(pred, masksw)
            _TP += n_tp
            _TN += n_tn
            _Np += n_p
            _Nn += n_n
            total += h*w

            '''
            fig = plt.figure()
            ax = fig.add_subplot(221)
            ax.title.set_text('rgb')
            # ax.imshow(imgsw.squeeze().cpu().permute(1,2,0).numpy())
                    
            ax.imshow(orig_img)
            if args.type=='gen_pt':
                # plot selected points
                pts = pt[0].cpu().squeeze()
                for pt in pts:
                    # print(pt[0], args.image_size, orig_size[0])
                    ax.plot(int(pt[0]/args.image_size*orig_size[0]), int(pt[1]/args.image_size*orig_size[1]), label='x', color='r')

            ax = fig.add_subplot(222)
            ax.title.set_text('gt')
            ax.imshow(masksw, cmap='gray')
            
            ax = fig.add_subplot(223)
            ax.title.set_text('mask_gen')
            ax.imshow(pred_mask_gen2, cmap='gray')


            ax = fig.add_subplot(224)
            ax.title.set_text('final')
            ax.imshow(pred, cmap='gray')
            plt.savefig(out_name)
            plt.close()
            '''

        
            pbar.update()

        TP, TN, Np, Nn = _TP, _TN, _Np, _Nn
        accuracy = (TP+TN) * 1.0 / total
        ber = (1 - 0.5*(TP/Np+TN/Nn))*100
        shadow_ber = (1 - TP/Np)*100
        noshadow_ber = (1 - TN/Nn)*100
        str_result = 'Accuracy: %f, BER: %f, Shadow Ber: %f, Non-Shadow Ber: %f'%(accuracy, ber, shadow_ber, noshadow_ber)
        print(str_result)

    return mask_dir

args = cfg.parse_args('test')

log_dir = f'./logs_{args.exp_name}/version_0'

args.weights = os.path.join(f'{log_dir}/checkpoints', 'sdnet-epoch=23-ber=2.702.ckpt')
print('weights: ', args.weights)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
net = sam_model_registry['vit_b'](args, checkpoint=args.sam_ckpt)

# load pretrained model
assert args.weights != 0
print(f'=> resuming from {args.weights}')
assert os.path.exists(args.weights)
checkpoint_file = os.path.join(args.weights)
assert os.path.exists(checkpoint_file)
checkpoint = torch.load(checkpoint_file, map_location='cpu')
start_epoch = checkpoint['epoch']

state_dict = checkpoint['state_dict']
checkpoint_ = {}
for k, v in state_dict.items():
    if not k.startswith('model.'):
        continue

    k = k[6:] # remove 'model.'
    checkpoint_[k] = v

net.load_state_dict(checkpoint_)

net = net.cuda()
###################################
# dataset
###################################
shadow_dataset = ShadowDataLoader(args)
test_loader = shadow_dataset.get_dataloader('test')

# begain valuation
mask_out_name = test_sam(args, test_loader, net)

# assert 1==0

###################################
# start calculating ber
###################################
# print('>>> start calculating ber ...')
# accuracy = 0.0
# _TP = 0.0
# _TN = 0.0
# _Np = 0.0
# _Nn = 0.0
# total = 0

# df_obj = DataFiles(args)
# test_files = df_obj.get_test()

# mask_out_path = os.path.join(f'./logs_{args.exp_name}/version_0', 'masks')

# for _, mask_path in tqdm(test_files):
#     gt_mask = np.array(Image.open(mask_path).convert('L'))
#     h, w = gt_mask.shape

#     pred_path = os.path.join(mask_out_path, os.path.basename(mask_path))
#     pred_img = np.array(Image.open(pred_path).convert('L'))

#     n_tp, n_tn, n_p, n_n, _ = cal_acc(pred_img, gt_mask)
#     _TP += n_tp
#     _TN += n_tn
#     _Np += n_p
#     _Nn += n_n
#     total += h*w

# TP, TN, Np, Nn = _TP, _TN, _Np, _Nn
# accuracy = (TP+TN) * 1.0 / total
# ber = (1 - 0.5*(TP/Np+TN/Nn))*100
# shadow_ber = (1 - TP/Np)*100
# noshadow_ber = (1 - TN/Nn)*100
# str_result = 'Accuracy: %f, BER: %f, Shadow Ber: %f, Non-Shadow Ber: %f'%(accuracy, ber, shadow_ber, noshadow_ber)
# print(str_result)


    