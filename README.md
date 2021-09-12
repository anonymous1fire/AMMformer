**AMMformer** for Object Detection on COCO and Image Classification on ImageNet
========
Our codes are based on [DETR](https://github.com/facebookresearch/detr) and [T2T-ViT](https://github.com/yitu-opensource/T2T-ViT).

Our codes are tested succesfully on **4** Nvidia 2080Ti, each with **11GB** GPU memory.
## Data preparation

### COCO
Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```
### ImageNet
Download ImageNet train and val images from [https://www.image-net.org](https://www.image-net.org/download.php).
Extract ImageNet by the [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).
We expect the directory structure to be the following:
```
path/to/imagenet/
  train2017/    # train images
    n01440764
  val2017/      # val images
    n01440764
```

## Package dependencies
We provide instructions how to install dependencies via [conda](https://docs.anaconda.com/anaconda/install/). 
```
git clone https://github.com/anonymous1fire/AMMformer.git
cd AMMformer
conda env create -f environment.yml
conda activate detr
conda install -c conda-forge tqdm
```

## AMMformer on COCO
### Direct deployment
1 GPU is required.
```
cd COCO
```
Download pretrained DETR-ResNet50 at [here](https://www.dropbox.com/s/ir0boozs2nba9rf/detr-resnet50.pth) and put it under pretrained folder.
```
mv detr-resnet50.pth ./pretrained
```
Set the coco dataset path (/path/to/coco) to the path you extracted in run_ammformer_eval.sh, run_linformer_eval.sh, run_nystrom_eval.sh, and run_performer_eval.sh.
Run the following scripts to obtain the deployment results.
```
sh run_ammformer_eval.sh
sh run_linformer_eval.sh
sh run_nystrom_eval.sh
sh run_performer_eval.sh
```
### Finetune each model with 50 epochs
4 GPUs with 11GB are required.

Set the coco dataset path (/path/to/coco) to the path you extracted in run_ammformer_finetune.sh, run_linformer_finetune.sh, run_nystrom_finetune.sh, and run_performer_finetune.sh.
Run the following scripts to finetune each model with 50 epochs.
```
sh run_ammformer_finetune.sh
sh run_linformer_finetune.sh
sh run_nystrom_finetune.sh
sh run_performer_finetune.sh
```

## Ammformer on ImageNet
```
cd ../ImageNet
```
### Validation
One GPU is required.

Download pretrained AMM-T2T-VIT-39 at [here](https://www.dropbox.com/s/9uzu44gblz6i8n3/amm-t2t-vit-39.pth.tar) and put it under pretrained folder for obtain validation accuracy.
```
mv amm-t2t-vit-39.pth ./pretrained
```
Set the imagenet data path (/path/to/imagenet) to the path you extracted in the following scripts and run it.
```
CUDA_VISIBLE_DEVICES=0 python main.py path/to/imagenet --model amm_t2t_vit_39 -b 64 --eval_checkpoint ./pretrained/att-t2t-vit-39.pth.tar
```
### Training
4 GPUs with 11GB are required to train AMM-T2T-VIT-39 for achieving  **82.2% top-1 accuray** on ImageNet, which only uses **41% parameters** and **50% FLOPs** of T2T-ViT\_t-24!
```
CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 path/to/imagenet --model amm_t2t_vit_39 -b 64 --lr 5e-4 --weight-decay .05 --amp --img-size 224
```
