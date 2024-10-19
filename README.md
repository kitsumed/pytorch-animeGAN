# AnimeGAN Pytorch
This is a Fork of [ptran1203 Pytorch implementation of AnimeGAN](https://github.com/ptran1203/pytorch-animeGAN) for fast photo animation.
This fork use more recents requirements.

* Paper: *AnimeGAN: a novel lightweight GAN for photo animation* - [Semantic scholar](https://www.semanticscholar.org/paper/AnimeGAN%3A-A-Novel-Lightweight-GAN-for-Photo-Chen-Liu/10a9c5d183e7e7df51db8bfa366bc862262b37d7#citing-papers) or from [Yoshino repo](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/doc/Chen2020_Chapter_AnimeGAN.pdf)
* Original implementation in [Tensorflow](https://github.com/TachibanaYoshino/AnimeGAN) by [Tachibana Yoshino](https://github.com/TachibanaYoshino)

| Input | Animation |
|--|--|
|![c2](./example/gif/giphy.gif)|![g2](./example/gif/giphy_anime.gif)|


---
## Tested On
This fork of PyTorch-AnimeGAN successfully worked on :

* **OS** : `Windows 10/11`
* **Python** : `3.10.11`

## Quick start

```bash
git clone https://github.com/kitsumed/pytorch-animeGAN
cd pytorch-animeGAN

# Create a environment (Optional) 
py -3.10.11 -m venv venv
# Go into the environment
.\venv\Scripts\activate

# Install cpu or gpu requirments
pip install -r requirements_cpu.txt
pip install -r requirements_gpu.txt
```
### Inference
**On your local machine**
> --src can be a directory or image file

```
python3 inference.py --weight /your/path/to/weight.pth --src /your/path/to/image_dir --out /path/to/output_dir
```

**From python code**

```python
from inference import Predictor

predictor= Predictor(
    '/your/path/to/weight.pth',
    # if set True, generated image will retain original color as input image
    retain_color=True
)

url = 'https://github.com/ptran1203/pytorch-animeGAN/blob/master/example/result/real/1%20(20).jpg?raw=true'

predictor.transform_file(url, "anime.jpg")
```

## Pretrained weight
Some weight where made available by ptran1203 [here](https://github.com/ptran1203/pytorch-animeGAN/releases).

### 1. Prepare dataset

#### 1.1 To download dataset from the paper, run below command

```bash
wget -O anime-gan.zip https://github.com/ptran1203/pytorch-animeGAN/releases/download/v1.0/dataset_v1.zip
unzip anime-gan.zip
```

=>  The dataset folder can be found in your current folder with named `dataset`

#### 1.2 Create custom data from anime video

You need to have a video file located on your machine.

**Step 1.** Create anime images from the video

```bash
python3 script/video_to_images.py --video-path /path/to/your_video.mp4\
                                --save-path dataset/MyCustomData/style\
                                --image-size 256\
```

**Step 2.** Create edge-smooth version of dataset from **Step 1.**

```bash
python3 script/edge_smooth.py --dataset MyCustomData --image-size 256
```

### 2. Train animeGAN

To train the animeGAN from command line, you can run `train.py` as the following:

```bash
python3 train.py --anime_image_dir dataset/Hayao \
                --real_image_dir dataset/photo_train \
                --model v2 \                 # animeGAN version, can be v1 or v2
                --batch 8 \
                --amp \                      # Turn on Automatic Mixed Precision training
                --init_epochs 10 \
                --exp_dir runs \
                --save-interval 1 \
                --gan-loss lsgan \           # one of [lsgan, hinge, bce]
                --init-lr 1e-4 \
                --lr-g 2e-5 \
                --lr-d 4e-5 \
                --wadvd 300.0\               # Aversarial loss weight for D
                --wadvg 300.0\               # Aversarial loss weight for G
                --wcon 1.5\                  # Content loss weight
                --wgra 3.0\                  # Gram loss weight
                --wcol 30.0\                 # Color loss weight
                --use_sn\                    # If set, use spectral normalization, default is False
```

### 3. Transform images

To convert images in a folder or single image, run `inference.py`, for example:

>
> --src and --out can be a directory or a file

```bash
python3 inference.py --weight path/to/Generator.pt \
                     --src dataset/test/HR_photo \
                     --out inference_images
```

### 4. Transform video

To convert a video to anime version:

> Be careful when choosing --batch-size, it might lead to CUDA memory error if the resolution of the video is too large

```bash
python3 inference.py --weight /your/path/to/weight.pth \
                        --src test_vid_3.mp4 \
                        --out test_vid_3_anime.mp4 \
                        --batch-size 4
```
