#! /bin/bash

python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_addr 127.0.0.3 --master_port 29502 main.py --coco_path "/path/to/coco" \
	--output_dir "./experiments/nystrom_eval" \
	--batch_size 4 --resume "./pretrained/detr-resnet50.pth" \
	--eval \
	--attn_type "nystrom" --landmarks 64 --max_seq_len 1764
