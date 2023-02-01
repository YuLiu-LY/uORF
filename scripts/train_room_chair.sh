#!/bin/bash
DATAROOT=${1:-'/home/yuliu/Dataset/uorf/room_chair/room_chair_train'}
PORT=${2:-8011}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_chair_120' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 60 --z_dim 64 --num_slots 5 \
    --model 'uorf_nogan' --gpu_ids '5' --init_method 'embedding' --niter 120 --percept_in 10 --no_locality_epoch 30 \
# done
echo "Done"
