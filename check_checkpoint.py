import sys
sys.path.insert(0,'/workspace/source/qd-3dt/qd3dt')
import torch
from mmcv.runner import load_checkpoint
from qd3dt.models import build_detector
import mmcv
checkpoint_path = "work_dirs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter/latest.pth"
config_path = "./configs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter.py"
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = mmcv.Config.fromfile(config_path)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    print(model)
    print("Checkpoint loaded successfully!")
    print("Keys in checkpoint:", checkpoint.keys())
    print("Keys in checkpoint:", checkpoint['meta'].keys())
    print("Keys in checkpoint:", checkpoint['state_dict'].keys())
    print("Keys in checkpoint:", checkpoint['optimizer'].keys())

    
except Exception as e:
    print(f"Error loading checkpoint: {e}")

# import pkgutil
# import mmcv.mmcv as a

# print("🔍 Available modules in qd3dt:")
# for module in pkgutil.walk_packages(a.__path__):
#     print(f"📦 {module.name}")


