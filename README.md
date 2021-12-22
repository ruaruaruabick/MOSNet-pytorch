# MOSNet
pytorch implementation of  "MOSNet: Deep Learning based Objective Assessment for Voice Conversion"
https://arxiv.org/abs/1904.08352

## Dependency
Linux Ubuntu 20.04
- GPU: GeForce RTX 2080 Ti
- CUDA version: 10.0

Python 3.7
- pytorch==1.4.0
- numpy==1.19.5
- tqdm
- scipy==1.6.2
- pandas==1.2.4
- matplotlib
- librosa==0.6.0



## Usage

### Reproducing results in the paper

1. `cd ./data` and run `bash download.sh` to download the VCC2018 evaluation results and submitted speech. (downsample the submitted speech might take some times)
2. Run `python mos_results_preprocess.py` to prepare the evaluation results. (Run `python bootsrap_estimation.py` to do the bootstrap experiment for intrinsic MOS calculation)
3. Run `python utils.py` to extract .wav to .h5
4. Run `python train.py -c config.json` to train a CNN-BLSTM version of MOSNet. 
4. Run `python test.py -c config.json --epoch BEST_EPOCH --is_fp16` to test a CNN-BLSTM version of MOSNet. 


#### Note
Thanks to the authors of the paper MOSNet and the code is based on their tensorflow implementation https://github.com/lochenchou/MOSNet.
However, my workstation will show OOM errors even with BATCH_SIZE=4 under tensorflow2.0 and RTX 2080 Ti. Therefore I implement the code with pytorch. Currently only 7700MiB memory is used when BATCH_SIZE=64.
If you find any problem with my code, you can write a issue.


## Citation

If you find this work useful in your research, please consider citing:
```
@inproceedings{mosnet,
  author={Lo, Chen-Chou and Fu, Szu-Wei and Huang, Wen-Chin and Wang, Xin and Yamagishi, Junichi and Tsao, Yu and Wang, Hsin-Min},
  title={MOSNet: Deep Learning based Objective Assessment for Voice Conversion},
  year=2019,
  booktitle={Proc. Interspeech 2019},
}
```
 
 
## License

This work is released under MIT License (see LICENSE file for details).


## VCC2018 Database & Results

The model is trained on the large listening evaluation results released by the Voice Conversion Challenge 2018.<br>
The listening test results can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3257)<br>
The databases and results (submitted speech) can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3061)<br>
