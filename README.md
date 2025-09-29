<div align="center">
<h2 align="center">PolyDiffuse: Polygonal Shape Reconstruction via <br/> Guided Set Diffusion Models</h2>

<h4 align="center"> NeurIPS 2023 </h4>

[Jiacheng Chen](https://jcchen.me) , [Ruizhi Deng](https://ruizhid.me) , [Yasutaka Furukawa](https://www2.cs.sfu.ca/~furukawa/)

Simon Fraser University 

[ArXiv](https://arxiv.org/abs/2306.01461), [Project page](https://poly-diffuse.github.io/)

<img src="./assets/imgs/teaser.jpg" width=85% height=85%>
</div>

This repository provides the official implementation of the paper [PolyDiffuse: Polygonal Shape Reconstruction via Guided Set Diffusion Models](https://arxiv.org/abs/2306.01461). This branch contains the code of the floorplan reconstruction task. The code of the HD mapping task will be ready in a separate branch.

The implementation of PolyDiffuse refers to the following open-source works: [EDM](https://github.com/NVlabs/edm), [RoomFormer](https://github.com/ywyue/RoomFormer), [MapTR](https://github.com/hustvl/MapTR), [HEAT](https://github.com/woodfrog/heat). We thank the authors for releasing their source code.

### Introduction video
https://github.com/woodfrog/poly-diffuse/assets/13405255/6dcd4bb6-9bd2-4fc8-aa65-c32b6ae0e92b


## Catalog
#### 1. Floorplan reconstruction
- [x] PolyDiffuse code and model checkpoints (in this `main` branch)
- [x] code and models of the adapted RoomFormer, which serves as our proposal generator (in the folder `./roomformer_adapted`)

#### 2. HD mapping
- [x] PolyDiffuse code and model checkpoints (in the [`hd-mapping` branch](https://github.com/woodfrog/poly-diffuse/tree/hd-mapping))


## Preparation
Please follow the instructions below to install the environment, as well as download the data and pretrained models.

A copy of all the resources on BaiduNetDisk: 
- Link: https://pan.baidu.com/s/1WpO3rrgnJQqG_Z2YQzb3Kg   
- Code: caam 

#### Installation
First, install the environment via conda:
```
conda env create -f environment.yml -n polydiffuse
conda activate polydiffuse
```

Then install the Deformable attention module used in the denoising network:
```
cd src/models/polygon_models/ops
sh make.sh
cd ../../../..
```

#### Data
Download the data file from this [link](https://www.dropbox.com/scl/fi/2lvtqjyorp4bc92dx4aea/processed_s3d.zip?rlkey=womgwak63agctkf1eyvzajrfp&st=wcwzmuwh&dl=0), put it into `./data` and unzip.

Unzip the results of RoomFormer, which serves as one of our proposal generators:
```
cd init_results
unzip npy_roomformer_s3d_256.zip
cd ..
```

#### Model checkpoints
Download the pretrained models from this [link](https://www.dropbox.com/scl/fi/f0z0x1xfy6qi0bbmxwgh4/s3d_pretrained_ckpts.zip?rlkey=w1wmjrrwvlwxf7jyhvj2vj8bj&st=0yz1p8en&dl=0), put it into `./training-runs` and unzip.

The final file structure should be as follows:

```
data
  ├── processed_s3d # the preprocessed structured3d dataset
          │── train
          │── test
          └── val   
init_results
  ├── npy_roomformer_s3d_256  # RoomFormer results of the test scenes in npy format

training-runs
  ├── s3d_pretrained_ckpts  # pretrained checkpoints
          │── guide  # checkpoints of guidance network
          └── denoise  # checkpoints of denoising network
```


## Testing 
We first introduce the testing pipeline, which consists of sampling, quantitative evaluation, and qualitative visualization. 

#### Sampling
First, run the sampling for all the test examples by:
```
CUDA_VISIBLE_DEVICES=0 bash scripts/sample.sh
```
Set up the `--proposal_type` argument to init with mimic rough annotations. Set `--viz_results=True` to show the visualization results including per-step GIF animation. Note that the default parameters in the above testing script assume the use of pretrained checkpoints. If you re-train the models, remember to set up the parameters accordingly.


#### Evaluation
Evaluate the results via:
```
python eval/evaluate_solution.py --dataset_path ./data/processed_s3d --dataset_type s3d --scene_id val --results_dir outputs/s3d_polydiffuse_step10/npy
```
The argument `results_dir` should be set up properly, which points to the sampling outputs.

This evaluation pipeline follows [MonteFloor](https://arxiv.org/abs/2103.11161), [HEAT](https://github.com/woodfrog/heat), and [RoomFormer](https://github.com/ywyue/RoomFormer).

#### Visualization
Get the qualitative visualization results used in the paper with the script `eval/visualize_npy.py`, including the GIF animation of the denoising process. Set up the paths in the script first, and then simply run:
```
python eval/visualize_npy.py
```

## Training
The training of PolyDiffuse consists of two separate stages: 1) guidance training and 2) denoising training.

#### Guidance training
Train the guidance network by:
```
bash scripts/train_guide.sh
```
The training of the guidance network is very fast as there is no image information involved, and usually takes less than an hour. 

#### Denoising training
Then train the denoising network by:
```
bash scripts/train.sh
```
Note that the path to the guidance network trained in the first stage needs to be set up properly with the argument `guide_ckpt`. The training takes around 30 hours with 4 NVIDIA RTX A5000 GPUs on our machine. 


## Adapted RoomFormer as the proposal generator

The folder `./roomformer_adapted` contains our modified version of RoomFormer (see details in the paper). The same environment as PolyDiffuse works for this codebase, please check the [folder README](https://github.com/woodfrog/poly-diffuse/tree/main/roomformer_adapted) for pretrained weights and training/testing instructions.


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
