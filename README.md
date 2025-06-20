## Enhanced Swin2SR: RGB SUPER-RESOLUTION FOR MULTISPECTRAL REMOTE SENSING IMAGES

Official PyTorch implementation of **Enhanced Swin2SR**.

In this paper, we propose **Enhanced Swin2SR** model, an enhanced version of Swin2SR for
Single-Image Super-Resolution for Remote Sensing.

## Usage

### Installation

```bash
$ git clone https://github.com/Ali12114/Remote-Sensing-SR.git
$ conda create -n enhance_swin2 python=3.10  # or your desired version
$ conda activate enhance_swin2
$ pip install -r requirements.txt
$ conda activate enhanced_swin2
```

### DOP20 and S2 Dataset Preparation

This repository contains all the necessary scripts and instructions for downloading, preprocessing, and preparing the dataset. [Dataset Repository](https://github.com/fabianstahl/SR_Double_Dataset)
After downloading the DOP20 and S2 dataset, split it into train and test set. In paper authors have 80/20 split for train/test.

If everything went well, you will have the following files structure:

```
data/oli2msi
├── test_hr
│   ├── dop20_32_451_5627_1_he.jpg
│   ├── dop20_32_460_5493_1_he.jpg
|   ...
├── test_lr
│   ├── dop20_32_451_5627_1_he.npy
│   ├── dop20_32_460_5493_1_he.npy
|   ...
├── train_hr
│   ├── dop20_32_451_5556_1_he.jpg
│   ├── dop20_32_459_5566_1_he.jpg
|   ...
└── train_lr
    ├── dop20_32_451_5556_1_he.npy
    ├── dop20_32_459_5566_1_he.npy
    ...
```

In paper authors have performed 4x super resolution. For this you would need to resize DOP20 images before training or inference. For that resize all of the DOP20 images to 400x400 using bicubic interpolation.

### Download pretrained

Open
[Pretrained Model](https://drive.google.com/file/d/1CLGX83VMxGHINp__08E1kWVXGd6K3WDi/view?usp=sharing)
page and download LR-HR-OUR21K-13B4xv1.zip file.

Unzip them inside the output directory, obtaining the following directories
structure:


```
output/LR-HR-OUR21K-13B4xv1/
├── checkpoints
│   └── model-100.pt
└── eval
    └── results-100.pt
```

### Best configuration

```bash
# S2 4x
CONFIG_FILE=cfgs/swin2_mose/super_res_ms.yml
```

### Train

```bash
CUDA_VISIBLE_DEVICES=$DEVICE_ID python src/main.py --phase train --config $CONFIG_FILE --output $OUT_DIR --epochs ${EPOCH} --epoch -1
```

### Validate

```bash
python src/main.py --phase test --config $CONFIG_FILE --output $OUT_DIR --batch_size 32 --epoch ${EPOCH}
```

### Show results

```
python src/main.py --phase vis --config $CONFIG_FILE --output $OUT_DIR --num_images 3 --epoch ${EPOCH}
```

[Swin2SR](https://link.springer.com/chapter/10.1007/978-3-031-25063-7_42)  

[Transformer for SISR](https://github.com/luissen/ESRT)

[WAT](https://github.com/mandalinadagi/Wavelettention)

[SWIN2-MOSE](https://github.com/IMPLabUniPr/swin2-mose)
## License

See [GPL v2](./LICENSE) License.

## Citation

If you find our work useful in your research, please cite:

```

```
