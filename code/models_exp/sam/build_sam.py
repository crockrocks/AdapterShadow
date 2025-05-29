import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../sam2')))
from sam2.build_sam import build_sam2


sam2_model_registry = {
    'hiera_b+': {
        'config': 'sam2/sam2.1_hiera_base_plus.yaml',
        'checkpoint': 'checkpoints/sam2.1_hiera_base_plus.pt',  
    },
    'hiera_l': {
        'config': 'sam2/sam2.1_hiera_large.yaml',
        'checkpoint': 'checkpoints/sam2.1_hiera_large.pt',  
    },
    'hiera_s': {
        'config': 'sam2/sam2.1_hiera_small.yaml',
        'checkpoint': 'checkpoints/sam2.1_hiera_small.pt',  
    },
    'hiera_t': {
        'config': 'sam2/sam2.1_hiera_tiny.yaml',
        'checkpoint': 'checkpoints/sam2.1_hiera_tiny.pt',  
    },
}

def get_sam2_model(model_key='hiera_b+', device='cuda', mode='eval'):
    entry = sam2_model_registry[model_key]
    model = build_sam2(
        config_file=entry['config'],
        ckpt_path=entry['checkpoint'],
        device=device,
        mode=mode,
    )
    return model
