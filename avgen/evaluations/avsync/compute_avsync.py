from typing import Optional
from einops import rearrange

import torch
import torchaudio
from torchvision import transforms

from avsync.models.avsync_classifier import load_avsync_model
from avgen.evaluations.models.clip import load_clip_model
from avgen.evaluations.clip.compute_clip import preprocess_videos as clip_preprocess_videos
from avgen.data.utils import waveform_to_melspectrogram


def preprocess_videos(videos):
	# videos: BCTHW, [0, 1]
	assert videos.ndim == 5
	b, c, t, h, w = videos.shape
	
	data_transform = transforms.Compose([
		transforms.Resize(
			(224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
		),
		transforms.CenterCrop(224),
		transforms.Normalize(
			mean=(0.48145466, 0.4578275, 0.40821073),
			std=(0.26862954, 0.26130258, 0.27577711),
		),
	])
	videos = data_transform(
		rearrange(videos, "b c t h w -> (b t) c h w")
	)
	videos = rearrange(videos, "(b t) c h w -> b c t h w", t=t)
	
	return videos


@torch.no_grad()
def compute_avsync_scores(audios, videos, net):
	'''
		videos in shape BCTHW in [0., 1.]
	'''
	videos = preprocess_videos(videos).contiguous()
	
	avsync_scores = net(audios, videos)
	
	return avsync_scores


@torch.no_grad()
def compute_relsync(audios, groundtruth_videos, generated_videos, net):
	'''
		videos in shape BCTHW in [0., 1.]
	'''
	groundtruth_videos = preprocess_videos(groundtruth_videos)
	generated_videos = preprocess_videos(generated_videos)
	
	groundtruth_avsync_scores = net(audios, groundtruth_videos)
	generated_avsync_scores = net(audios, generated_videos)
	
	cat_scores = torch.stack([groundtruth_avsync_scores, generated_avsync_scores], dim=1) # (b, 2)
	relsync = torch.softmax(cat_scores, dim=1)[:, 1].contiguous().detach().cpu()

	return relsync


@torch.no_grad()
def compute_alignsync(audios, groundtruth_videos, generated_videos, net, clip_net):
	'''
		videos in shape BCTHW in [0., 1.],
		audios in shape BCNT
	'''

	f = groundtruth_videos.shape[2]
	
	relsync = compute_relsync(audios, groundtruth_videos, generated_videos, net)
	
	# Compute AlignProb
	videos = torch.cat([
		groundtruth_videos[:, :, 0:1], generated_videos[:, :, 1:]
	], dim=2)
	videos = rearrange(videos, "b c f h w -> b f c h w")
	videos, audios, _ = clip_preprocess_videos(videos, audios)
	ia_sims = rearrange(
		clip_net(videos, audios)["ia_sim"],
		"(b f) -> b f", f=f
	).detach().cpu()
	
	groundtruth_first_frame_ia_sims = ia_sims[:, 0:1].contiguous()
	generated_pred_frame_ia_sims = ia_sims[:, 1:].contiguous()
	cat_scores = torch.stack([
		groundtruth_first_frame_ia_sims.expand_as(generated_pred_frame_ia_sims), generated_pred_frame_ia_sims
	], dim=2)
	align_probs = torch.softmax(cat_scores, dim=2)[:, :, 1].mean(dim=1)
	
	alignsync = align_probs * relsync
	
	return alignsync


@torch.no_grad()
def compute_sync_metrics_on_av(
	audio_waveform: torch.Tensor,
	audio_sr: int,
	video: torch.Tensor,
	groundtruth_video: Optional[torch.Tensor] = None,
	metric: str = "alignsync",
	device: torch.device = torch.device("cuda"),
	dtype: torch.dtype = torch.float32
):
	'''
		audio_waveform: loaded audio in shape (c t)
		audio_sr: sampling rate of loaded audio_waveforms
		video: input videos in shape (c t h w) in [0, 1]
		groundtruth_video: reference video in same shape as videos, only needed when computing relsyn/alignsync
		metric: 'alignsync', 'relsync', 'avsync_score'
	'''
	c, t, h, w = video.shape
	assert t == 12, "video should have 12 frames in 6 FPS"
	assert metric in ['alignsync', 'relsync', 'avsync_score']
	if metric in ['relsync', 'alignsync']:
		assert groundtruth_video is not None and groundtruth_video.shape == video.shape, \
			f"To compute {metric}, groundtruth_video is needed as reference, and in same shape as video"
	
	avsync_net = load_avsync_model().to(device=device, dtype=dtype)
	if metric == "alignsync":
		clip_net = load_clip_model().to(device=device, dtype=dtype)
	
	# Convert audio into melspectrogram
	audio = torchaudio.functional.resample(audio_waveform, orig_freq=audio_sr, new_freq=16000)  # (c, t)
	audio = waveform_to_melspectrogram(audio)  # (1, n, t)
	
	audio = audio.unsqueeze(0).contiguous().to(device=device, dtype=dtype)
	video = video.unsqueeze(0).contiguous().to(device=device, dtype=dtype)
	if groundtruth_video is not None:
		groundtruth_video = groundtruth_video.unsqueeze(0).contiguous().to(device=device, dtype=dtype)
	
	if metric == "alignsync":
		return compute_alignsync(audio, groundtruth_video, video, avsync_net, clip_net)[0]
	elif metric == "relsync":
		return compute_relsync(audio, groundtruth_video, video, avsync_net)[0]

	return compute_avsync_scores(audio, video, avsync_net)[0]
	
	
	
	
	
	