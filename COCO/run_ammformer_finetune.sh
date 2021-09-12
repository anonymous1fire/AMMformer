#! /bin/bash

python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_addr 127.0.0.3 --master_port 29502 main.py --coco_path "/path/to/coco" \
	--output_dir "./experiments/ammformer_finetune" \
	--batch_size 8 --epoch 50 --resume ./pretrained/detr-resnet50.pth \
	--lr 0.00001 --lr_backbone 0 \
	--attn_type "ammformer" 
