"""Measure params, FLOPs and peak GPU memory for selected trainers.
Usage: python tools/measure_model_stats.py --trainer <TrainerName> --plans /path/to/nnUNetData_plans_v2.1 --device cuda
"""
import argparse
import importlib
import os
import sys
from batchgenerators.utilities.file_and_folder_operations import load_pickle
import torch

# try optional libs
try:
    from thop import profile
except Exception:
    profile = None
try:
    from ptflops import get_model_complexity_info
except Exception:
    get_model_complexity_info = None


def load_plans(plans_path):
    p = os.path.join(plans_path, 'plans_3D.pkl')
    if not os.path.exists(p):
        # fallback to generic plans.pkl inside plans_path
        p = os.path.join(plans_path, 'plans.pkl')
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return load_pickle(p)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure(model, C, patch, device='cuda'):
    dev = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model.to(dev)
    model.eval()

    with torch.no_grad():
        x = torch.randn(1, C, *patch, device=dev)
        # reset and warmup
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize()
        _ = model(x)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        # FLOPs
        flops = None
        if profile is not None and dev.type == 'cuda':
            try:
                flops, _ = profile(model, inputs=(x,), verbose=False)
            except Exception:
                flops = None
        if flops is None and get_model_complexity_info is not None:
            try:
                # ptflops expects module on cpu and will create a new input
                model_cpu = model.to('cpu')
                macs, params = get_model_complexity_info(model_cpu, (C,)+tuple(patch), as_strings=False, print_per_layer_stat=False)
                flops = macs * 2
                model.to(dev)
            except Exception:
                flops = None

        peak = None
        if dev.type == 'cuda':
            peak = torch.cuda.max_memory_allocated(dev)
        return flops, peak


# Trainer -> network instantiation map
TRAINER_MAP = {
    'UNETRTrainer': {
        'module': 'nnunet.training.network_training.UNETRTrainer',
        'class': 'UNETRTrainer',
        'instantiate_net': 'unetr'
    },
    'UNet3DTrainer': {
        'module': 'nnunet.training.network_training.UNet3DTrainer',
        'class': 'UNet3DTrainer',
        'instantiate_net': 'unet3d'
    },
    'nnUNetTrainerV2': {
        'module': 'nnunet.training.network_training.nnUNetTrainerV2',
        'class': 'nnUNetTrainerV2',
        'instantiate_net': 'generic_unet'
    },
    'BANetTrainerV2': {
        'module': 'nnunet.training.network_training.BANetTrainerV2',
        'class': 'BANetTrainerV2',
        'instantiate_net': 'banet'
    },
    'MedNeXtTrainerV2': {
        'module': 'nnunet.training.network_training.MedNeXt.MedNeXtTrainerV2',
        'class': 'MedNeXtTrainerV2',
        'instantiate_net': 'mednext'
    }
}


def build_and_measure(trainer_name, plans_path=None, device='cuda'):
    if trainer_name not in TRAINER_MAP:
        raise ValueError(trainer_name)
    tinfo = TRAINER_MAP[trainer_name]

    # load plans if provided
    if plans_path is not None:
        plans = load_plans(plans_path)
        stage = list(plans['plans_per_stage'].keys())[0]
        stage_plans = plans['plans_per_stage'][stage]
        C = int(plans['num_modalities'])
        K = int(plans['num_classes']) + 1
        patch = tuple(map(int, stage_plans['patch_size']))
        base_nf = int(plans.get('base_num_features', 32))
    else:
        C = 1; K = 2; patch = (96,96,96); base_nf = 32

    # import the trainer module to see initialization patterns
    mod = importlib.import_module(tinfo['module'])

    model = None
    instype = tinfo['instantiate_net']
    if instype == 'unetr':
        from nnunet.network_architecture.unetr import UNETR
        model = UNETR(in_channels=C, out_channels=K, img_size=patch, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12)
    elif instype == 'unet3d':
        from nnunet.network_architecture.UNet3D import UNet3D
        model = UNet3D(in_channels=C, out_channels=K, init_features=32)
    elif instype == 'generic_unet':
        from nnunet.network_architecture.generic_UNet import Generic_UNet
        # try to reconstruct args used in trainer
        # default conv_per_stage: trainer.process_plans sets conv_per_stage in trainer object, but fallback to 2
        net_num_pool = len(stage_plans['pool_op_kernel_sizes']) if 'pool_op_kernel_sizes' in stage_plans else len(stage_plans.get('num_pool_per_axis', []))
        conv_per_stage = plans.get('conv_per_stage', 2)
        model = Generic_UNet(C, base_nf, K, net_num_pool, conv_per_stage, 2, None, None, None, None, None, None, True, False, lambda x: x, None, stage_plans.get('pool_op_kernel_sizes', None), stage_plans.get('conv_kernel_sizes', None), False, True, True)
    elif instype == 'banet':
        from nnunet.network_architecture.generic_BANetV2 import generic_BANetV2
        net_num_pool = len(stage_plans['pool_op_kernel_sizes']) if 'pool_op_kernel_sizes' in stage_plans else len(stage_plans.get('num_pool_per_axis', []))
        conv_per_stage = plans.get('conv_per_stage', 2)
        model = generic_BANetV2(C, base_nf, K, net_num_pool, conv_per_stage, 2, None, None, None, None, None, None, True, False, lambda x: x, None, stage_plans.get('pool_op_kernel_sizes', None), stage_plans.get('conv_kernel_sizes', None), False, True, True)
    elif instype == 'mednext':
        # MedNeXt trainer uses MedNeXt wrapper -> class MedNeXt in file
        from nnunet.network_architecture.mednextv1.MedNextV1 import MedNeXt as MedNeXt_Orig
        from nnunet.training.network_training.MedNeXt.MedNeXtTrainerV2 import MedNeXt
        model = MedNeXt(in_channels=C, n_channels=16, n_classes=K, exp_r=2, kernel_size=3, deep_supervision=False, do_res=True, do_res_up_down=True, block_counts=[2]*9, checkpoint_style=None)
    else:
        raise NotImplementedError(instype)

    params = count_params(model)
    flops, peak = measure(model, C, patch, device=device)
    return {
        'trainer': trainer_name,
        'params': params,
        'flops': flops,
        'peak_bytes': peak,
        'C': C,
        'K': K,
        'patch': patch,
        'base_nf': base_nf
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer', required=True)
    parser.add_argument('--plans', required=False, default=None)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    out = build_and_measure(args.trainer, args.plans, device=args.device)
    print(out)
