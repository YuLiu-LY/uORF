#!/bin/bash
#SBATCH --job-name=uorf-test # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1      # number of gpus per node
#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)

DATAROOT=${1:-'/scratch/generalvision/CLEVR_567/clevr_567_test'}
CHECKPOINT=${2:-'/home/liuyu/scratch/uORF/checkpoints/clevr_567/run-2023-01-29-11-55-11'}
PORT=8077
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 500 --n_img_each_scene 4 \
    --checkpoints_dir $CHECKPOINT --name clevr_sa --exp_id latest --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 64 --render_size 8 --frustum_size 128 \
    --n_samp 256 --z_dim 40 --num_slots 8 \
    --model 'uorf_eval' 
echo "Done"