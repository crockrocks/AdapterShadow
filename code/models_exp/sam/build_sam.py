# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../sam2')))
from sam2.build_sam import build_sam2

# Registry for SAM2 configs (add more as needed)
sam2_model_registry = {
    'hiera_b+': {
        'config': 'sam2/sam2_hiera_b+.yaml',
        'checkpoint': 'checkpoints/sam2_hiera_base_plus.pt',
    },
    'hiera_l': {
        'config': 'sam2/sam2_hiera_l.yaml',
        'checkpoint': 'checkpoints/sam2_hiera_large.pt',
    },
    'hiera_s': {
        'config': 'sam2/sam2_hiera_s.yaml',
        'checkpoint': 'checkpoints/sam2_hiera_small.pt',
    },
    'hiera_t': {
        'config': 'sam2/sam2_hiera_t.yaml',
        'checkpoint': 'checkpoints/sam2_hiera_tiny.pt',
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
