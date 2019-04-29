#!/bin/sh


# python test_fashion.py --cfg cfg/DAMSM/CelebA.yml --gpu 0 
CUDA_LAUNCH_BLOCKING=1 python pretrain_DAMSM_test.py --cfg cfg/DAMSM/Fashion.yml --gpu 0 
