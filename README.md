<div align="center">
<h2 align="center">PolyDiffuse: Polygonal Shape Reconstruction via <br/> Guided Set Diffusion Models</h2>

[Jiacheng Chen](https://jcchen.me) , [Ruizhi Deng](https://ruizhid.me) , [Yasutaka Furukawa](https://www2.cs.sfu.ca/~furukawa/)

Simon Fraser University

ArXiv preprint ([arXiv 2306.01461](https://arxiv.org/abs/2306.01461)), [Project page](https://poly-diffuse.github.io/)

<img src="./assets/imgs/teaser.jpg" width=85% height=85%>
</div>

This repository provides the official implementation of the paper [PolyDiffuse: Polygonal Shape Reconstruction via Guided Set Diffusion Models](https://arxiv.org/abs/2306.01461). This branch contains the code of the **HD mapping** task. The code of the floorplan reconstruction task is in the ``main`` branch.

The implementation of PolyDiffuse for the HD mapping task refers to the open-source works [EDM](https://github.com/NVlabs/edm) and [MapTR](https://github.com/hustvl/MapTR). The overall training and sampling framework follows EDM. The folder `projects` is borrowed and adapted from MapTR (for data pipeline, denoising network architecture, and evaluation). The folder `mmdetection3d` is the source code of `mmdet3d` copied from the original MapTR codebase, which will be compiled during installation. We thank the authors for releasing their source code.

### Introduction video
https://github.com/woodfrog/poly-diffuse/assets/13405255/6dcd4bb6-9bd2-4fc8-aa65-c32b6ae0e92b


## Preparation
The denoising network and the data pipeline are the same as [MapTR](https://github.com/hustvl/MapTR), so both the environment installation and dataset downloads follow MapTR.


#### Installation
Please first follow the [instructions provided by the MapTR repo](https://github.com/hustvl/MapTR/blob/main/docs/install.md) to install the conda environment. Then, activate the MapTR conda environment and install the remaining dependencies by `pip install -r requirements.txt`.


#### Data
Since we follow the RGB input setting as in the original [MapTR paper](https://arxiv.org/abs/2208.14437), we only need a subset of the nuScenes dataset (i.e., RGB captures from six surrounding cameras, map annotations). To simplify the data preparation process and save the disk space, we provide a zipped file for all necessary data (~40GB) via this [Dropbox link](https://www.dropbox.com/s/p9vcd94c3wm0qmy/nuscenes_maptr_processed.zip?dl=0). Please download it into ```./data``` and unzip. Or you can also follow the [guidelines provided by MapTR](https://github.com/hustvl/MapTR/blob/main/docs/prepare_dataset.md) to download the entire dataset and run the preprocessing. 

We run the official MapTR to serve as one of our proposal generators. The saved results are provided in [this link](https://www.dropbox.com/s/g44qkmopqvmgii4/maptr_test.json?dl=0). Please download and put it under `./init_results`. Or you can also run the MapTR to get the outputs.



#### Model checkpoints
Please download the following two checkpoints:
(1). [Our pretrained models](https://www.dropbox.com/s/l63jwspd8h7416d/nuscenes_pretrained_ckpts.zip?dl=0), put it into `./training-runs` and unzip.
(2). [Pretrained MapTR and ResNet](https://www.dropbox.com/s/5fzcvn8r5rjl5py/ckpts.zip?dl=0) (needed for denoising training), unzip it as `./ckpts`.

The final file structure should be as follows:

```
data
  ├── can_bus  
  ├── nuscenes 
          │── samples
          │── maps
          └── ...  
ckpts   # checkpoints for initialize the denoising network
  ├── resnet50-19c8e357.pth 
  ├── maptr_tiny_r50_110e.pth

init_results
  ├── maptr_test.json  # MapTR test results (serving as initial proposals)

training-runs
  ├── nuscenes_pretrained_ckpts
          │── guide/...  # checkpoints of guidance network
          └── denoise/...  # checkpoints of denoising network
```


## Testing 
The testing pipeline consists of sampling, visualization, and quantitative evaluation. 

#### Sampling & visualization
First, run the sampling for all the test examples by:
```
CUDA_VISIBLE_DEVICES=0 bash scripts/sample.sh
```
The default setting uses MapTR results as the initial proposal, set up the argument `--proposal_type=rough_annot`  to init with mimic rough annotations. Set `--viz_results=True` to visualize the predictions, and set `--viz_gif=True` to also show the per-step GIF animation for each test sample.

Note that the default parameters in the script assume the use of pretrained checkpoints. If you re-train the models, remember to set up the parameters accordingly.


#### Evaluation
Evaluate the results via:
```
bash scripts/eval_map.sh
```
The argument `--results_path` should be set up properly to point to the output of the sampling script. The arguement `--consider_angle` is turned on to consider the angle-level correctness as discussed in our papaer. Without this argument, the evaluation becomes the same as the original mAP with Chamfer-only matching criterion.



## Training
The training of PolyDiffuse consists of two separate stages: 1) guidance training and 2) denoising training.

#### Guidance training
Train the guidance network by:
```
bash scripts/train_guide.sh
```
The training of the guidance network takes around an hour on a single NVIDIA RTX A5000 GPU.

#### Denoising training
Then train the denoising network by:
```
bash scripts/train.sh
```
Note that the path to the guidance network trained in the first stage needs to be set up properly with the argument `--guide_ckpt`. On our machine, the training takes around 45 hours with 8 NVIDIA RTX A5000, or around 65 hours with 4 NVIDIA RTX A5000.


## Acknowledgements
This research is partially supported by NSERC Discovery Grants with Accelerator Supplements and DND/NSERC Discovery Grant Supplement, NSERC Alliance Grants, and John R. Evans Leaders Fund (JELF). We thank the Digital Research Alliance of Canada and BC DRI Group for providing computational resources.


## Citation
If you find PolyDiffuse is helpful in your work, please consider starring 🌟 the repo and citing it by:
```bibtex
@article{Chen2023PolyDiffuse,
  title={PolyDiffuse: Polygonal Shape Reconstruction via Guided Set Diffusion Models},
  author={Jiacheng Chen and Ruizhi Deng and Yasutaka Furukawa},
  journal={ArXiv},
  year={2023},
  volume={abs/2306.01461}
}
```
