#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_simple.py dual &
CUDA_VISIBLE_DEVICES=1 python train_simple.py sim &
CUDA_VISIBLE_DEVICES=0 python train_simple.py random &
CUDA_VISIBLE_DEVICES=1 python train_simple.py rb 