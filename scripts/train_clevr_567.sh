#!/bin/bash
DATAROOT=${1:-'/home/yuliu/Dataset/uorf/clevr_567/clevr_567_train'}
PORT=${2:-8077}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'clevr_boqsa_300' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 150 --z_dim 40 --num_slots 8 \
    --model 'uorf_nogan' --gpu_ids '4' --init_method 'embedding' --niter 300 --percept_in 25 --no_locality_epoch 75 \
# done
echo "Done"
