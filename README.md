# uORF with BO-QSA
This repo is forked from [uORF](https://github.com/KovenYu/uORF) and modified by [Yu Liu](https://yuliu-ly.github.io). We adapt [BO-QSA](https://github.com/YuLiu-LY/BO-QSA) to uORF to further investigate the effectiveness and generality of BO-QSA.
uORF: [ICLR22] [Unsupervised Discovery of Object Radiance Fields](https://arxiv.org/abs/2107.07905) by [Hong-Xing Yu](https://kovenyu.com), [Leonidas J. Guibas](https://geometry.stanford.edu/member/guibas/), [Jiajun Wu](https://jiajunwu.com/) 
BO-QSA: [ICLR2023] [Improving Object-centric Learning With Query Optimization](http://arxiv.org/abs/2210.08990) by [Baoxiong Jia](https://buzz-beater.github.io/)\*, [YuLiu](https://yuliu-ly.github.io)\*, [Siyuan Huang](https://siyuanhuang.com/)

![teaser](teaser.gif)

Project website: [uORF](https://kovenyu.com/uorf), [BO-QSA](https://bo-qsa.github.io)

# Main modifications
- change the `model.py, uorf_gan_model, uorf_nogan_model` in `models` and `*.sh` in `scripts` folder to adapt BO-QSA to uORF. We only modify the initialization and optimization method of the Slot-Attention module in uORF, leaving all other hyperparameters unchanged.
- add `vis.py`, `vis_utils.py` in `utils`, `uorf_vis_model.py` in `models`, and `vis_*.sh` in `scripts` to visualize the results of uORF and BO-QSA.
- add `generate_video.ipynb` to generate video and gif of the results.

## Environment
We recommend using Conda:
```sh
conda env create -f environment.yml
conda activate uorf-3090
```
or install the packages listed therein. Please make sure you have NVIDIA drivers supporting CUDA 11.0, or modify the version specifictions in `environment.yml`.

## Data and model
Please download datasets and models [here](https://office365stanford-my.sharepoint.com/:f:/g/personal/koven_stanford_edu/Et9SOVcOxOdHilaqfq4Y3PsBsiPGW6NGdbMd2i3tRSB5Dg?e=WRrXIh).
If you want to train on your own dataset or generate your own dataset similar to our used ones, please refer to this [README](data/README.md).

## Evaluation
We assume you have a GPU.
If you have already downloaded and unzipped the datasets and models into the root directory,
simply run
```sh
bash scripts/eval_nvs_seg_chair.sh
```
from the root directory. Replace the script filename with `eval_nvs_seg_clevr.sh`, `eval_nvs_seg_diverse.sh`,
and `eval_scene_manip.sh` for different evaluations. Results will be saved into `./results/`.
During evaluation, the results on-the-fly will also be sent to visdom in a nicer form, which can be accessed from
[localhost:8077](http://localhost:8077).

## Training
We assume you have a GPU with no less than 24GB memory (evaluation does not require this as rendering can be done ray-wise but some losses are defined on the image space),
e.g., 3090. Then run
```shell
bash scripts/train_clevr_567.sh
```
or other training scripts. If you unzip datasets on some other place, add the location as the first parameter:
```shell
bash scripts/train_clevr_567.sh PATH_TO_DATASET
```
Training takes ~6 days on a 3090 for CLEVR-567 and Room-Chair, and ~9 days for Room-Diverse.
It can take even longer for less powerful GPUs (e.g., ~10 days on a titan RTX for CLEVR-567 and Room-Chair).
During training, visualization will be sent to [localhost:8077](http://localhost:8077).

## Bibtex
```
@inproceedings{yu2022unsupervised,
  author    = {Yu, Hong-Xing and Guibas, Leonidas J. and Wu, Jiajun},
  title     = {Unsupervised Discovery of Object Radiance Fields},
  booktitle = {International Conference on Learning Representations},
  year      = {2022},
}
```

## Acknowledgement
Our code framework is adapted from Jun-Yan Zhu's [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
Some code related to adversarial loss is adapted from [a pytorch implementation of StyleGAN2](https://github.com/rosinality/stylegan2-pytorch).
Some snippets are adapted from pytorch [slot attention](https://github.com/lucidrains/slot-attention) and [NeRF](https://github.com/yenchenlin/nerf-pytorch).
If you find any problem please don't hesitate to email me at koven@stanford.edu or open an issue.
