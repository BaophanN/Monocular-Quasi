#!/usr/bin/env bash
set -x
# ----configs----
# work_dir=work_dirs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter/output_val_box3d_deep_depth_motion_lstm_3dcen
# config_path=configs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter.py
# gpu_ids=0
# gpu_nums=1
# PY_ARGS=--data_split_prefix val --full_frames
work_dir=$1
config_path=$2
gpu_ids=$3
gpu_nums=$4
PY_ARGS=${@:5}
# --------------
folder='work_dirs/'$(dirname ${config_path#*/})
config=$(basename -s .py ${config_path})
# -----------------

# $ ./scripts/run_eval_nusc.sh ${WORK_DIR} ${CONFIG} ${gpu_id} 1 --data_split_prefix ${EXP_NAME} --full_frames

# # ${WORK_DIR} is the path to place the model output.
# # ${CONFIG} is the corresponding config file you use.
# # ${EXP_NAME} is the experiment name you want to specify.
# $ ./scripts/run_eval_nusc.sh work_dirs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter/output_val_box3d_deep_depth_motion_lstm_3dcen configs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter.py 0 1 --data_split_prefix val --full_frames



# get the results of each camera
./scripts/test_eval_exp.sh nuscenes ${config_path} ${gpu_ids} ${gpu_nums} --add_test_set ${PY_ARGS}

# 3D Detection generation
python scripts/eval_nusc_det.py \
--version=v1.0-mini \
--root=data/nuscenes/ \
--work_dir=$work_dir \
--gt_anns=data/nuscenes/anns/tracking_test.json

# 3D Tracking generation
python scripts/eval_nusc_mot.py \
--version=v1.0-mini \
--root=data/nuscenes/ \
--work_dir=$work_dir \
--gt_anns=data/nuscenes/anns/tracking_test.json
