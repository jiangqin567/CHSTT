# CHSTT: Cross-scale Hierarchical Spatio-Temporal Transformer for Video Enhancement 

By [Qin Jiang], [Qinglin Wang], [Lihua Chi], [Jie Liu]

## Dependencies and Installation

- Python >= 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.11](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

1. Clone repository

    ```bash
    git clone https://github.com/jiangqin/CHSTT.git
    ```

2. Install dependent packages

    ```bash
    cd CHSTT
    pip install -r requirements.txt
    ```


## Dataset Preparation

## Video Super-Resolution
1. Training set
	* [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset. We regroup the training and validation dataset into one folder. The original training dataset has 240 clips from 000 to 239. The original validation dataset were renamed from 240 to 269.
		- Make REDS structure be:
	    ```
			├────REDS
				├────train
					├────train_sharp
						├────000
						├────...
						├────269
					├────train_sharp_bicubic
						├────X4
							├────000
							├────...
							├────269
        ```
	* [Viemo-90K](https://github.com/anchen1011/toflow) dataset. Download the [original training + test set](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip) and use the script 'degradation/BD_degradation.m' (run in MATLAB) to generate the low-resolution images. The `sep_trainlist.txt` file listing the training samples in the download zip file.
		- Make Vimeo-90K structure be:
		```
			├────vimeo_septuplet
				├────sequences
					├────00001
					├────...
					├────00096
				├────sequences_BD
					├────00001
					├────...
					├────00096
				├────sep_trainlist.txt
				├────sep_testlist.txt
        ```

2. Testing set
	* [REDS4](https://seungjunnah.github.io/Datasets/reds.html) dataset. The 000, 011, 015, 020 clips from the original training dataset of REDS.
    * [Viemo-90K](https://github.com/anchen1011/toflow) dataset. The `sep_testlist.txt` file listing the testing samples in the download zip file.
    * [Vid4 and UDM10](https://www.terabox.com/web/share/link?surl=LMuQCVntRegfZSxn7s3hXw&path=%2Fproject%2Fpfnl) dataset. Use the script 'degradation/BD_degradation.m' (run in MATLAB) to generate the low-resolution images.
		- Make Vid4 and UDM10 structure be:
		```
			├────VID4
				├────BD
					├────calendar
					├────...
				├────HR
					├────calendar
					├────...
			├────UDM10
				├────BD
					├────archpeople
					├────...
				├────HR
					├────archpeople
					├────...
        ```
## Video Deblurring
1.  Training set
    * DVD_train
    
2. Testing set
    * DVD_test
   
## Video Denoising

1. Training set
    * DAVIS_train
   
2. Testing set
    * DAVIS_test
    * Set8

## Testing

- Please refer to **[configuration of testing](options/test/)** for more details.

    ```bash
    # Test on REDS
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/test_chstt_x4_REDS.yml --launcher pytorch

    # Test on Vimeo-90K
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/test_chstt_x4_Vimeo.yml --launcher pytorch

    # Test on Vid4
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/test_chstt_x4_Vid4.yml --launcher pytorch
    ```

## Citation

If you use this code of our paper please cite:


## Acknowledgments

This repository is implemented based on [BasicSR](https://github.com/xinntao/BasicSR). If you use the repository, please consider citing BasicSR.