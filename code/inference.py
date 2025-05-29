import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def generate_click_prompt(image, mask=None):
    h, w = image.shape[-2:]
    if mask is not None:
        mask_np = mask.cpu().numpy().squeeze()
        y_fg, x_fg = np.where(mask_np > 0.5)
        y_bg, x_bg = np.where(mask_np <= 0.5)
        
        if len(y_fg) > 0 and len(y_bg) > 0:
            fg_idx = len(y_fg) // 2
            bg_idx = len(y_bg) // 2
            points = np.array([
                [x_fg[fg_idx], y_fg[fg_idx], 1],
                [x_bg[bg_idx], y_bg[bg_idx], 0],
            ])
            return points
    
    return np.array([[w//2, h//2, 1]])

def prepare_image(image_path, size=1024):
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size
    img = img.resize((size, size), Image.LANCZOS)
    img_np = np.array(img)
    return img_np, orig_size

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)  
    model_config = os.path.join(os.path.dirname(__file__), '/home/harsh/shadow/AdapterShadow/code/sam2/sam2/sam2_hiera_b+.yaml')
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found at {args.checkpoint}")    
    print(f"Loading model from {args.checkpoint}...")
    model = build_sam2(
        config_file=model_config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mode='eval',
    )
    checkpoint = torch.load(args.checkpoint, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    predictor = SAM2ImagePredictor(model)
    
    if os.path.isdir(args.input):
        image_paths = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    else:
        image_paths = [args.input]
    
    print(f"Processing {len(image_paths)} images...")
    
    for image_path in tqdm(image_paths):
        img_np, orig_size = prepare_image(image_path, size=args.image_size)
        
        predictor.set_image(img_np)
        
        point_coords = np.array([[img_np.shape[1]//2, img_np.shape[0]//2]])
        point_labels = np.array([1])
        
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        
        mask = masks[0]
        
        input_filename = os.path.basename(image_path)
        output_name = os.path.splitext(input_filename)[0] + '_shadow_mask.png'
        output_path = os.path.join(args.output_dir, output_name)
        
        if args.image_size != orig_size[0] or args.image_size != orig_size[1]:
            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_pil = mask_pil.resize(orig_size, Image.NEAREST)
            mask = np.array(mask_pil) > 127
        
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_image.save(output_path)
        
        print(f"Saved mask to {output_path}")
        
        if args.visualize:
            input_img = Image.open(image_path).convert('RGB')
            input_img = input_img.resize(orig_size)
            
            overlay = np.zeros((orig_size[1], orig_size[0], 3), dtype=np.uint8)
            overlay[..., 0] = mask.astype(np.uint8) * 255
            
            input_array = np.array(input_img)
            blend = (input_array * 0.7 + overlay * 0.3).astype(np.uint8)
            
            vis_path = os.path.join(args.output_dir, os.path.splitext(input_filename)[0] + '_visualization.jpg')
            Image.fromarray(blend).save(vis_path)
            print(f"Saved visualization to {vis_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shadow Detection Inference")
    parser.add_argument("--input", help="Input image path or directory")
    parser.add_argument("input_path", nargs="?", help="Input image path (positional argument)")
    parser.add_argument("--output_dir", default="results", help="Output directory for masks")
    parser.add_argument("--checkpoint", default="logs/sbu_training_2025_05_12_05_53_15/checkpoints/checkpoint_best.pth", 
                        help="Path to model checkpoint")
    parser.add_argument("--image_size", type=int, default=1024, help="Size to resize input images to")
    parser.add_argument("--visualize", action="store_true", help="Create visualization overlays")
    
    args = parser.parse_args()
    
    if args.input_path and not args.input:
        args.input = args.input_path
    
    if not args.input:
        parser.error("Please provide an input image path either with --input or as a positional argument")
    
    main(args) 