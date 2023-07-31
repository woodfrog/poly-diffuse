## Adapted RoomFormer used in PolyDiffuse

The codebase is based on the overall framework of [HEAT](https://github.com/woodfrog/heat) and refers to the [official RoomFormer implementation](https://github.com/ywyue/RoomFormer). The major modifications are as follows:

- Employ more aggressive data augmentations (random rotation and flipping) while training the model for a longer schedule.
- Remove the Dice loss to simplify the training loss.

With the above modifications, the room-level performance becomes slightly worse while the corner and edge-level performance improve. PolyDiffuse uses the network from this cobebase as the denoising network.


### Pre-trained models

Please download the checkpoint from this [Dropbox link](https://www.dropbox.com/s/h8rc5nvlki9p5m0/roomformer_adapted_pretrained.zip?dl=0) and unzip it into `./checkpoints`.


### Testing

First run the inferenec by:
```
bash scripts/infer.sh
```
Then use [the same evaluation script as PolyDiffuse](https://github.com/woodfrog/poly-diffuse#evaluation) for quantitative evaluation. Remember to update the `--results_dir` accordingly.


### Training
Run the training by:
```
bash scripts/train.sh
```
We use 4 NVIDIA RTX A5000 GPUs and please adjust the corresponding parameters (i.e., number of nodes, batch size) if you have different computing resources.




