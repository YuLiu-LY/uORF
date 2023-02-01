#!/bin/bash
#SBATCH --job-name=uorf-test # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1      # number of gpus per node
#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)

DATAROOT=${1:-'/scratch/generalvision/uORF/room_chair/room_chair_train'}
PORT=${2:-8011}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_chair_400' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 200 --z_dim 64 --num_slots 5 \
    --model 'uorf_nogan' --gpu_ids '0' --init_method 'embedding' --niter 400 --percept_in 33 --no_locality_epoch 100 
# done
echo "Done"