# WADEPre Official Implementation



WADEPre: **WA**velet-based **D**ecomposition Model for **E**xtreme **Pre**cipitation Nowcasting with Multi-Scale Learning



*Authors*: Baitian Liu [1], Haiping Zhang [1], Huiling Yuan [2, 3], Dongjing Wang [1], Ying Li [4], Feng Chen [4], Hao Wu [1, *]



> 1. Department of Computer Science and Technology, Hangzhou Dianzi University, Hangzhou, Zhejiang  Province, China
>2. State Key Laboratory of Severe Weather Meteorological Science and Technology, Nanjing University, Nanjing, China
> 3. Key Laboratory of Mesoscale Severe Weather, Ministry of Education, and School of Atmospheric Sciences, Nanjing University, Nanjing, China
>4. Zhejiang Institute of Meteorological Sciences, Hangzhou, Zhejiang Province, China
> 
>*Corresponding author: Hao Wu



**The paper has been submitted to KDD 2026 and is currently under review.**




> Last updated: January 28. 2026



## Introduction

**WADEPre** is a wavelet-based deep learning framework designed to address <u>smoothing effects</u> in extreme precipitation nowcasting. By explicitly decomposing radar imagery into stable large-scale advection (approximation coefficients) and volatile local intensity (detail coefficients), the model overcomes the regression-to-the-mean dilemma inherent in standard pixel-wise optimization. Powered by a `multi-task curriculum learning strategy`, WADEPre achieves state-of-the-art performance on the SEVIR and Shanghai Radar benchmarks, significantly improving forecast accuracy and structural fidelity for high-impact weather events compared with Fourier-based and deterministic baselines.



This repository contains the training and inference code for running WADE-Pre to make predictions (6 --> 6) on two datasets.



## Dataset

- *SEVIR*:  We use Vertically Integrated Liquid (VIL) mosaics in SEVIR for benchmarking precipitation nowcasting, predicting the future VIL up to 6\*10 minutes given 6\*10 minutes of context VIL, and resizing the spatial resolution to 128. The resolution is thus `6×128×128 → 6×128×128`.

- *Shanghai Radar*: The raw data spans a 460 × 460 grid covering a physical region of `460km × 398km`, with reflectivity values ranging from 0 to 70 dBZ. We resize the spatial resolution to 128. The resolution is thus `6×128×128 → 6×128×128`.


We thank AWS for providing online download service, for more details please refer to [AWS - Storm EVent ImageRy (SEVIR)](https://registry.opendata.aws/sevir/)

The Shanghai Radar dataset can be downloaded from [here](https://zenodo.org/records/7251972).


## Code



### Environment

```bash
conda env create -f env.yaml
conda activate wadepre
```



### Evaluation

Open the `eval.py` file, and replace the weight file path with your path. Then run the command:

```bash
python eval.py 
```



### Training

```python
python train.py
```



When you start training, these folders may offer you useful information:

- `logs` : All training metrics, including hyperparameters, are recorded in this file.
- `checkpoints` : Model weight file. 



## Reproduction

The model’s weights **will be released upon acceptance**.




## Credit

Our implementation is heavily inspired by the following excellent works. We extend our thanks to the original authors.



Third-party libraries and tools:

- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Muon](https://github.com/KellerJordan/Muon)
- [Draw.io](https://www.drawio.com/)



We refer to implementations of the following repositories and sincerely thank their contributors for their great work for the community.

- [Dilated ResNet](https://github.com/fyu/drn)
- [FPN](https://github.com/kuangliu/pytorch-fpn)
- [ConvLSTM](https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py)
- [MAU](https://github.com/ZhengChang467/MAU)
- [EarthFarseer](https://github.com/Alexander-wu/EarthFarseer)
- [SimVP](https://github.com/A4Bio/SimVP)
- [AlphaPre](https://github.com/linkenghong/AlphaPre)
