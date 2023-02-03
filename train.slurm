#!/bin/bash
#SBATCH --job-name=uorf-1200 # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1      # number of gpus per node
#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)

DATAROOT=${1:-'/scratch/generalvision/uORF/clevr_567/clevr_567_train'}
PORT=${2:-8077}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'clevr_boqsa_s_1200' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --z_dim 40 --num_slots 8 \
    --model 'uorf_nogan' --init_method 'embedding' \
    --coarse_epoch 600 \
    --project 'uorf-clevr' \
    --exp_id 'latest' \
    --sigma_step 30000 \
    # --niter 600 --percept_in 50 --no_locality_epoch 150 \
    # --continue_train --epoch_count 133 \