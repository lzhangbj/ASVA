from tqdm import tqdm
from einops import rearrange
import numpy as np
import torch
from torchvision import transforms

from ..dists import frechet_distance

def preprocess_videos(videos, sequence_length=None):
	# video: BCTHW, [0, 1]
	assert videos.ndim == 5
	b, c, t, h, w = videos.shape
	# temporal crop
	if sequence_length is not None:
		assert sequence_length <= t
		videos = videos[:, :, :sequence_length]
		videos = videos.contiguous()
		t = sequence_length
	
	videos = rearrange(videos, "b c t h w -> (b t) c h w")
	
	transform_func = transforms.Compose([
		transforms.Resize(
			(224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
		),
		transforms.CenterCrop(224)
	])
	videos = transform_func(videos)
	videos = videos * 2 - 1
	
	videos = rearrange(videos, "(b t) c h w -> b c t h w", t=t)
	
	return videos

@torch.no_grad()
def compute_fvd_video_features(videos, net):
	'''
		videos in shape BCTHW in [0., 1.]
	'''
	videos = preprocess_videos(videos).contiguous()
	
	detector_kwargs = dict(rescale=False, resize=False, return_features=True)
	logits = net(videos, **detector_kwargs)
	# logits = net(videos)
	return logits