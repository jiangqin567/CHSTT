#conda activate torch

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=2314 basicsr/test.py  \
-opt options/test/test_chstt_x4_Vid4.yml --launcher pytorch
