from tqdm import tqdm
from einops import rearrange, repeat
import numpy as np
import torch
from torchvision import transforms


def preprocess_videos(videos, audios=None, texts=None):
	# images: BTCHW, [0, 1]
	assert videos.ndim == 5
	b, f, c, h, w = videos.shape
	
	videos = rearrange(videos, "b f c h w -> (b f) c h w")
	
	transform_func = transforms.Compose(
		[
			transforms.Resize(
				(224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
			),
			transforms.CenterCrop(224),
			transforms.Normalize(
				mean=(0.48145466, 0.4578275, 0.40821073),
				std=(0.26862954, 0.26130258, 0.27577711),
			),
		]
	)
	videos = transform_func(videos)
	
	if audios is not None:
		audios = repeat(audios, "b c n t -> (b f) c n t", f=f).contiguous()
	
	if texts is not None:
		new_texts = []
		for text in texts:
			new_texts+=[text]*f
		texts = new_texts

	return videos, audios, texts


@torch.no_grad()
def compute_clip_consistency(videos, audios=None, texts=None, net=None):
	'''
		videos in shape BTCHW in [0., 1.]
		audios in shape BCNT
	'''
	f = videos.shape[1]
	videos, audios, texts = preprocess_videos(videos, audios, texts)
	
	result_dict = net(videos, audios, texts)
	for key, val in result_dict.items():
		result_dict[key] = rearrange(val, "(b f) -> b f", f=f)
		
	return result_dict


