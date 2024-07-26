<h1 align="center">Audio-Synchronized Visual Animation</h1>
<h2 align="center">ECCV 2024</h2>
<h3 align="center">Lin Zhang, Shentong Mo, Yijing Zhang, Pedro Morgado</h3>

<div align="center">
<a href=https://arxiv.org/abs/2403.05659><img src="https://img.shields.io/badge/arXiv-2008.00951-b31b1b.svg"></a> &nbsp;
<a href='https://lzhangbj.github.io/projects/asva/asva.html'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</div>

<p align="center">
  <a href="https://www.youtube.com/watch?v=Z8IW09yggRk">
    <img src="https://img.youtube.com/vi/Z8IW09yggRk/0.jpg" alt="Watch this video" />
  </a>
</p>

### Checklist
- [x] Release pretrained checkpoints
- [x] Release inference code on audio-conditioned image animation and sync metrics
- [x] Release ASVA training and evaluation code
- [x] Release AVSync classifier training and evaluation code
- [ ] Release Huggingface Demo

## 1. Create environment
We use `video_reader` backend of torchvision to load audio and videos, which requires building torchvision locally
```angular2html
conda create -n asva python==3.10 -y
conda activate asva

pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Build torchvision from source
mkdir -p submodules
cd submodules
git clone https://github.com/pytorch/vision.git
cd vision
git checkout tags/v0.16.0
conda install -c conda-forge 'ffmpeg<4.3' -y
python setup.py install
cd ../..

pip install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/submodules/ImageBind
```

## 2. Download pretrained models

### Download required features/models

* [ImageBind](https://github.com/facebookresearch/ImageBind?fbclid=IwAR2fU8mKKsOLCsqZsP8vn6nbzC5XwksXLuIpAWOaEZ6jQTQWGncQp6FfPc8): Pretrained frozen audio encoder
Please download the following pretrained model weights under folder `pretrained/`
* [I3D](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt): Evaluating FVD
* [Stable Diffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5): Load pretrained image generation model
* [AVID-CMA](https://dl.fbaipublicfiles.com/avid-cma/checkpoints/AVID-CMA_Audioset_InstX-N1024-PosW-N64-Top32_checkpoint.pth.tar): Initialize AVSync Classifier's encoders
* [Precomputed null text encodings](https://drive.google.com/file/d/1fuKlVKdR9tw2wFE3RShH6jRjnvxPNR5u/view?usp=sharing): Ease of computatoin

They should be structured as following:
```angular2html
- submodules/
    - ImageBind/
- pretrained/
    - i3d_torchscript.pt
    - stable-diffusion-v1-5/
    - openai-clip-l_null_text_encoding.pt
    - AVID-CMA_Audioset_InstX-N1024-PosW-N64-Top32_checkpoint.pth.tar
```

### Download pretrained AVSyncD and AVSync Classifier checkpoints

<table>
  <tr>
    <th>Model</th>
	<th>Dataset</th>
	<th>Checkpoint</th>
	<th>Config</th>
	<th>Audio CFG</th>
	<th>FVD</th>
	<th>AlignSync</th>
  </tr>
  <tr>
    <td rowspan="9">AVSyncD</td>
	<td rowspan="3">AVSync15</td>
	<td rowspan="3"><a href="https://drive.google.com/file/d/17ZYopMVM1ZuJ1CBZPzhwAyOa4rR9Eo-_/view?usp=sharing">GoogleDrive</a></td>
	<td rowspan="3"><a href="configs/audio-cond_animation/avsync15_audio-cond_cfg.yaml">Link</a></td>
	<td>1.0</td>
	<td>323.06</td>
	<td>22.21</td>
  </tr>
  <tr>
	<td>4.0</td>
	<td>300.82</td>
	<td>22.64</td>
  </tr>
  <tr>
	<td>8.0</td>
	<td>375.02</td>
	<td>22.70</td>
  </tr>
  <tr>
	<td rowspan="3">Landscapes</td>
	<td rowspan="3"><a href="https://drive.google.com/file/d/1Wa0wK9D_qlkT8U2O8zCz6UoQql-A3zjD/view?usp=sharing">GoogleDrive</a></td>
	<td rowspan="3"><a href="configs/audio-cond_animation/landscapes_audio-cond_cfg.yaml">Link</a></td>
	<td>1.0</td>
	<td>491.37</td>
	<td>24.94</td>
  </tr>
  <tr>
	<td>4.0</td>
	<td>449.59</td>
	<td>25.02</td>
  </tr>
  <tr>
	<td>8.0</td>
	<td>547.97</td>
	<td>25.16</td>
  </tr>
  <tr>
	<td rowspan="3">TheGreatestHits</td>
	<td rowspan="3"><a href="https://drive.google.com/file/d/1u8Ksc9TrDhcr6tV_7xH9RsbdkklH-2y9/view?usp=sharing">GoogleDrive</a></td>
	<td rowspan="3"><a href="configs/audio-cond_animation/thegreatesthits_audio-cond_cfg.yaml">Link</a></td>
	<td>1.0</td>
	<td>305.41</td>
	<td>22.56</td>
  </tr>
  <tr>
	<td>4.0</td>
	<td>255.49</td>
	<td>22.89</td>
  </tr>
  <tr>
	<td>8.0</td>
	<td>279.12</td>
	<td>23.14</td>
  </tr>
</table>

| Model             | Dataset | Checkpoint      | Config                | A2V Sync Acc | V2A Sync Acc |
|-------------------|---------|-----------------|-----------------------|--------------|--------------|
| AVSync Classifier | VGGSS   | [GoogleDrive](https://drive.google.com/file/d/1Paqjad4a8mjujMJBEYqmnNEgGPi595lb/view?usp=sharing) | [Link](configs/avsync/vggss_sync_contrast.yaml) | 40.76        | 40.86        |

They should be structured as following:
```angular2html
- checkpoints/
    - audio-cond_animation/
        - avsync15_audio-cond_cfg/
        - landscapes_audio-cond_cfg/
        - thegreatesthits_audio-cond_cfg/
    - avsync/
        - vggss_sync_contrast/
```

## 3. Demo
### Generate animation on audio / image / video
The program first tries to load audio from `audio` and image from `image`. 
If they are not specified, the program then loads audio or image from `video`.
```angular2html
python -W ignore scripts/animation_demo.py --dataset AVSync15 --category "lions roaring" --audio_guidance 4.0 \
    --audio ./assets/lions_roaring.wav --image ./assets/lion_and_gun.png --save_path ./assets/generation_lion_roaring.mp4

python -W ignore scripts/animation_demo.py --dataset AVSync15 --category "machine gun shooting" --audio_guidance 4.0 \
    --audio ./assets/machine_gun_shooting.wav --image ./assets/lion_and_gun.png --save_path ./assets/generation_lion_shooting_gun.mp4
```
<div align="center">
    <img src="assets/generation_lion_roaring.gif" alt="Lion roaring">
    <img src="assets/generation_lion_shooting_gun.gif" alt="Lion shooting gun">
</div>

### Compute sync metrics for generated video
To compute `alignsync` and `relsync`, a `groundtruth_video` should be input as reference.
To compute `avsync_score`, only `audio` and `video` are needed.
```angular2html
python -W ignore scripts/avsync_metric.py --metric alignsync --audio {audio path} --video {generated video path} --groundtruth_video {groundtruth video path}
```

## 4. Download datasets
Each dataset has 3 files/folders:
* `videos/`: the directory to store all .mp4 video files
* `train.txt`: training file names
* `test.txt`: testing file names

Optionally, we precomputed two files for ease of computation:
* `class_mapping.json`: mapping category string in file name to text string used for conditioning
* `class_clip_text_encodings_stable-diffusion-v1-5.pt`: mapping text string used for conditioning to clip text encodings

Download these files from [GoogleDrive](https://drive.google.com/drive/folders/1rGPBiswIVBgZj5BUViGRWePxEEdq21p6?usp=sharing), and place them under `datasets/` folder.

To download videos:
* AVSync15: download videos from link above
* Landscapes: download videos from [MMDiffusion](https://drive.google.com/drive/folders/14A1zaQI5EfShlv3QirgCGeNFzZBzQ3lq?usp=sharing). 
* TheGreatestHits: download videos from [Visually Indicated Sounds](https://andrewowens.com/vis/).
* VGGSS: for AVSync classifier training/evaluation, download videos from [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/). Only videos listed in `train.txt` and `test.txt` are needed.

Overall, the `datasets` folder has the following structure
```angular2html
- datasets/
    - AVSync15/
        - videos/
            - baby_babbling_crying/
            - cap_gun_shooting/
            - ...
        - train.txt
        - test.txt
        - class_mapping.json
        - class_clip_text_encodings_stable-diffusion-v1-5.pt
    - Landscapes/
        - videos/
            - train/
                - explosion
                - ...
            - test/
                - explosion
                - ...
            - ...
        - train.txt
        - test.txt
        - class_mapping.json
        - class_clip_text_encodings_stable-diffusion-v1-5.pt
    - TheGreatestHits/
        - videos/
            - xxxx_denoised_thumb.mp4
            - ...
        - train.txt
        - test.txt
        - class_clip_text_encodings_stable-diffusion-v1-5.pt
    - VGGSS/
        - videos/
            - air_conditioning_noise/
            - air_horn/
            - ...
        - train.txt
        - test.txt
```

## 5. Train and evaluate AVSyncD

### Train
Training is done on 8 RTX-A4500 GPUs (20G) on AVSync15/Landscapes or 4 A100 GPUs on TheGreatestHits, with a total batch size of 64, accelerate for distributed training, and wandb for logging.
Checkpoints will be flushed every `checkpointing_steps` iterations. 
Besides, checkpoints at the `checkpointing_milestones`-th iteration and the last iteration will both be saved. 
Please adjust these two parameters in .yaml config file to avoid important weights being flushed when you customize training recipes.
```angular2html
PYTHONWARNINGS="ignore" accelerate launch scripts/animation_train.py --config_file configs/audio-cond_animation/{datasetname}_audio-cond_cfg.yaml
```
Results are saved to `exps/audio-cond_animation/{dataset}_audio-cond_cfg`, with the same structure as pretrained checkpoints.

### Evaluation
Evaluation is two-step:
1. Generate 3 clips per video for test set using `scripts/animation_gen.py`
2. Evaluate between groundtruth clips and generated clips using `scripts/animation_eval.py`

Please refer to `scripts/animation_test_{dataset}.sh` for the steps. 
For example, to evaluate AVSyncD pretrained on AVSync15 with audio guidance scale = 4.0:
```angular2html
bash scripts/animation_test_avsync15.sh checkpoints/audio-cond_animation/avsync15_audio-cond_cfg 37000 4.0
```

## 6. Train and evaluate AVSync Classifier

### Train
AVSync Clasifier is trained on VGGSS training split for 4 days, 8 RTX-A4500 GPUs, and batchsize 32.
```angular2html
PYTHONWARNINGS="ignore" accelerate launch scripts/avsync_train.py --config_file configs/avsync/vggss_sync_contrast.yaml
```

### Evaluation
We followed VGGSoundSync to sample 31 clips with 0.04-s gaps on each video. 
Given the audio/video clip at the center, we predict its synchronized video/audio clip's index.
A tolerate range of 5 is applied, since human is tolerant to 0.2s asynchronous.

For example, to evaluate our pretrained AVSync Classifier on 8 GPUs, run:
```angular2html
PYTHONWARNINGS="ignore" accelerate launch --num_processes=8 scripts/avsync_eval.py --checkpoint checkpoints/avsync/vggss_sync_contrast/ckpts/checkpoint-40000/modules --mixed_precision fp16 
```

## Citation
Please consider citing our paper if you find this repo useful:
```bib
@inproceedings{linz2024asva,
    title={Audio-Synchronized Visual Animation},
    author={Lin Zhang and Shentong Mo and Yijing Zhang and Pedro Morgado},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2024}
}
```