#!/bin/bash
DATAROOT=${1:-'/home/yuliu/Dataset/uorf/clevr_567/clevr_567_test'}
CHECKPOINT=${2:-'/home/yuliu/Dataset/uorf/ckpt/clevr_567_models/latest'}
PORT=8077
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 500 --n_img_each_scene 4 \
    --checkpoints_dir $CHECKPOINT --name clevr_567 --exp_id latest --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 64 --render_size 8 --frustum_size 128 \
    --n_samp 256 --z_dim 40 --num_slots 8 \
    --model 'uorf_eval' --gpu_ids '6'
echo "Done"
