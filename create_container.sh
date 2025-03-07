docker run --name qd-3dt-again\
                --gpus all\
                --mount type=bind,source="/home/baogp4/Bao",target="/workspace/source"\
                --mount type=bind,source="/home/baogp4/datasets",target="/workspace/datasets"\
                --shm-size=16GB\
                -it cmcandy2021/redet-torch1.3-cuda101-py3.7-mmcv0.2.3-ch:latest
                # -it pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
                # -it pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel