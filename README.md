# 3D Scene Reconstruction with Novel View Synthesis from Single Image Based on Ray-Transformer Neural Radiance Fields

Tsung-Min Yu <br>
National Taiwan University of Science and Technology (NTUST)


This is the official repository for our paper: [Ray-Transformer NeRF](https://etheses.lib.ntust.edu.tw/cgi-bin/gs32/gsweb.cgi?o=dstdcdr&s=id=%22G0M11015032%22.&searchmode=basic).<br>
This repository based on pixelNeRF.
<https://github.com/sxyu/pixel-nerf>

# Environment setup

To start, we prefer creating the environment using conda:
```sh
conda env create -f environment.yml
conda activate rtnerf
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
Please make sure you have up-to-date NVIDIA drivers supporting CUDA 11.3 at least.


# Getting the data

- For the main ShapeNet experiments, we use the ShapeNet 64x64 dataset from NMR
https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip
(Hosted by DVR authors)
    - For novel-category generalization experiment, a custom split is needed.
      Download the following script:
      https://drive.google.com/file/d/1Uxf0GguAUTSFIDD_7zuPbxk1C9WgXjce/view?usp=sharing
      place the said file under `NMR_Dataset` and run `python genlist.py` in the said directory.
      This generates train/val/test lists for the experiment. Note for evaluation performance reasons,
      test is only 1/4 of the unseen categories.
      (Hosted by pixelNeRF authors)

- The remaining datasets may be found in
https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR?usp=sharing
(Hosted by pixelNeRF authors)
    - DTU (4x downsampled, rescaled) in DVR's DTU format `dtu_dataset.zip`

Data adapters are built into the code.

# Running the model (video generation)

The main implementation is in the `src/` directory,
while evalutation scripts are in `eval/`.

First, download all pretrained weight files from
<https://drive.google.com/file/d/1H-5-wIV1FvhHtEBgZ7meDbsiwr_GcrZA/view?usp=share_link>.
Extract this to `<project dir>/checkpoints/`, so that `<project dir>/checkpoints/dtu/pixel_nerf_latest` exists.


# Overview of flags

Generally, all scripts in the project take the following flags
- `-n <expname>`: experiment name, matching checkpoint directory name
- `-D <datadir>`: dataset directory. 
    To save typing, you can set a default data directory for each expname in `expconf.conf` under `datadir`.
    For SRN/multi_obj datasets with
    separate directories e.g. `path/cars_train`, `path/cars_val`,
    put `-D path/cars`.
- `--split <train | val | test>`: data set split
- `-S <subset_id>`: scene or object id to render
- `--gpu_id <GPU(s)>`: GPU id(s) to use, space delimited. All scripts except `calc_metrics.py`
are parallelized. If not specified, uses GPU 0.
Examples: `--gpu_id=0` or `--gpu_id='0 1 3'`.
- `-R <sz>`: Batch size of rendered rays per object. Default is 50000 (eval) and 128 (train); make it smaller if you run out of memory.  On large-memory GPUs, you can set it to 100000 for eval.
- `-c <conf/*.conf>`: config file. *Automatically inferred* for the provided experiments from the expname. Thus the flag is only required when working with your own expnames.
                    You can associate a config file with any additional expnames in the `config` section of `<project root>/expconf.conf`.

Please refer the the following table for a list of provided experiments with associated config and data files:

| Name                       | expname -n      | config -c (automatic from expconf.conf)   | Data file                               | data dir -D       |
|----------------------------|-----------------|-------------------------------------------|-----------------------------------------|-------------------|
| ShapeNet all category | ShapeNet_All    | conf/exp/sn64_all_BNv2+ResXt+RTLN1.conf                 | NMR_Dataset.zip (from AWS)              | path/NMR_Dataset  |
| ShapeNet unseen category   | ShapeNet_Unseen | conf/exp/sn64_unseen_BNv2+ResXt+RTLN1.conf          | NMR_Dataset.zip (from AWS) + genlist.py | path/NMR_Dataset  |
| DTU                        | DTU_MVS             | conf/exp/dtu_BNv2+ResXt+RTLN1.conf                  | dtu_dataset.zip                         | path/rs_dtu_4     |


# Quantitative evaluation instructions

All evaluation code is in `eval/` directory.
The full, parallelized evaluation code is in `eval/eval.py`.


## Full Evaluation

Here we provide commands for full evaluation with `eval/eval.py`.
After running this you should also use `eval/calc_metrics.py`, described in the section below,
to obtain final metrics.

Append `--gpu_id=<GPUs>` to specify GPUs, for example `--gpu_id=0` or `--gpu_id='0 1 3'`.
**It is highly recommended to use multiple GPUs if possible to finish in reasonable time.**
We use 4-10 for evaluations as available.
Resume-capability is built-in, and you can simply run the command again to resume if the process is terminated.

In all cases, a source-view specification is required. This can be either `-P` or `-L`. `-P 'view1 view2..'` specifies
a set of fixed input views. In contrast, `-L` should point to a viewlist file (viewlist/src_*.txt) which specifies views to use for each object.

Renderings and progress will be saved to the output directory, specified by `-O <dirname>`.

### ShapeNet Multiple Categories (NMR)

- All category eval `python eval/eval.py -D <path>/NMR_Dataset -n ShapeNet_All -L viewlist/src_dvr.txt --multicat -O eval_out/ShapeNet_All -c conf/exp/sn64_all_BNv2+ResXt+RTLN1.conf`

- Unseen category eval `python eval/eval.py -D <path>/NMR_Dataset -n ShapeNet_Unseen -L viewlist/src_gen.txt --multicat -O eval_out/ShapeNet_Unseen -c conf/exp/sn64_unseen_BNv2+ResXt+RTLN1.conf`


### DTU
- 1-view `python eval/eval.py -D <path>/rs_dtu_4 --split val -n DTU_MVS -P '25' -O DTU_MVS -c conf/exp/dtu_BNv2+ResXt+RTLN1.conf`

## Final Metric Computation

The above computes PSNR and SSIM without quantization. The final metrics we report in the paper
use the rendered images saved to disk, and also includes LPIPS + category breakdown.
To do so run the `eval/calc_metrics.py`, as in the following examples

### ShapeNet Multiple Categories (NMR)

- All category `python eval/calc_metrics.py -D <path>/NMR_Dataset -O eval_out/ShapeNet_All -F dvr --list_name 'softras_test' --multicat --gpu_id=0`

- Unseen category `python eval/calc_metrics.py -D <path>/NMR_Dataset -O eval_out/ShapeNet_Unseen -F dvr --list_name 'softras_test' --multicat --gpu_id=0 `


### DTU
- 1-view `python eval/calc_metrics.py -D <path>/rs_dtu_4/DTU -O eval_out/DTU_MVS -F dvr --list_name 'new_val' --exclude_dtu_bad --dtu_sort`


Adjust -O according to the -O flag of the eval.py command.
(Note: Currently this script has an ugly standalone argument parser.)
This should print a metric summary like the following
```
airplane     psnr: 30.267611 ssim: 0.951685 lpips: 0.077900 n_inst: 809
bench        psnr: 26.968754 ssim: 0.920448 lpips: 0.102839 n_inst: 364
cabinet      psnr: 28.330564 ssim: 0.916191 lpips: 0.093351 n_inst: 315
car          psnr: 27.922617 ssim: 0.945922 lpips: 0.087204 n_inst: 1500
chair        psnr: 24.422404 ssim: 0.869815 lpips: 0.131552 n_inst: 1356
display      psnr: 24.596372 ssim: 0.874220 lpips: 0.119729 n_inst: 219
lamp         psnr: 29.298646 ssim: 0.920599 lpips: 0.101070 n_inst: 464
loudspeaker  psnr: 24.968117 ssim: 0.865188 lpips: 0.128639 n_inst: 324
rifle        psnr: 30.709510 ssim: 0.969676 lpips: 0.062767 n_inst: 475
sofa         psnr: 27.411612 ssim: 0.914487 lpips: 0.104755 n_inst: 635
table        psnr: 26.377833 ssim: 0.911050 lpips: 0.082509 n_inst: 1702
telephone    psnr: 27.347570 ssim: 0.925163 lpips: 0.090046 n_inst: 211
vessel       psnr: 29.727893 ssim: 0.944743 lpips: 0.102059 n_inst: 388
---
total        psnr: 27.333032 ssim: 0.918114 lpips: 0.096920

eval_out/ShapeNet_All
```

# Training instructions

Training code is in `train/` directory, specifically `train/train.py`.

- Example for training to DTU: `python train/train.py -n DTU_MVS_exp -c conf/exp/dtu_BNv2+ResXt+RTLN1.conf -D <path>/rs_dtu_4 -V 1 --gpu_id=0 --resume --epochs 1000 -B 4`

- Example for training to ShapeNet_All: `python train/train.py -n ShapeNet_All_exp -c conf/exp/sn64_all_BNv2+ResXt+RTLN1.conf -D <path>/NMR_Dataset --gpu_id=0 --resume -V 1 --no_bbox_step 400000 --epochs 1000 -B 8`

- Example for training to ShapeNet_Unseen: `python train/train.py -n ShapeNet_Unseen_exp -c conf/exp/sn64_unseen_BNv2+ResXt+RTLN1.conf -D <path>/NMR_Dataset --gpu_id=0 --resume -V 1 --no_bbox_step 400000 --epochs 1000 -B 8`

Additional flags
- `--resume` to resume from checkpoint, if available. Usually just pass this to be safe.
- `-V <number>` to specify number of input views to train with. Default is 1.
    - `-V 'numbers separated by space'` to use random number of views per batch. This does not work so well in our experience but we use it for SRN experiment.
- `-B <number>` batch size of objects, default 4
- `--lr <learning rate>`, `--epochs <number of epochs>`
- `--no_bbox_step <number>` to specify iteration after which to stop using bounding-box sampling.
Set to 0 to disable.

If the checkpoint becomes corrupted for some reason (e.g. if process crashes when saving), a backup is saved to `checkpoints/<expname>/pixel_nerf_backup`.
To avoid having to specify -c, -D each time, edit `<project root>/expconf.conf` and add rows for your expname in the config and datadir sections.

## Log files and visualizations
View logfiles with `tensorboard --logdir <project dir>/logs/<expname>`.
Visualizations are written to  `<project dir>/visuals/<expname>/<epoch>_<batch>_vis.png`.
They are of the form
- Top coarse, bottom fine (1 row if fine sample disabled)
- Left-to-right: input-views, depth, output, alpha.

