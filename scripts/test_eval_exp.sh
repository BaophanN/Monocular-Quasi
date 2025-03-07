#!/usr/bin/env bash
set -x
# ----configs----
dataset=$1
config_path=$2
gpu_ids=$3
gpu_nums=$4
PY_ARGS=${@:5}
# --------------
root=.
folder='work_dirs/'$(dirname ${config_path#*/})
config=$(basename -s .py ${config_path})
# -----------------

cd mmcv
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ..

# test
CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=${gpu_ids} python3 -u ${root}/tools/test_eval_video_exp.py \
${dataset} \
${config_path} \
./${folder}/${config}/latest.pth \
./${folder}/${config}/output/output.pkl \
${PY_ARGS}


# $ ./scripts/test_eval_exp.sh nuscenes configs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter.py 0 1 --data_split_prefix test_r101_dcn_nusc --add_test_set

# quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter.py
# quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter.py