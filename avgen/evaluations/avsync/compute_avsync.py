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
def compute_relsync(audios, videos, net, ref_audios=None, ref_videos=None):
	'''
		videos in shape BCTHW in [0., 1.]
	'''
	assert (ref_audios is None) ^ (ref_videos is None), "Please specify either ref_audios or ref_videos"
	
	videos = preprocess_videos(videos)
	avsync_scores = net(audios, videos)
	
	if ref_audios is not None:
		ref_avsync_scores = net(ref_audios, videos)
	else:
		ref_videos = preprocess_videos(ref_videos)
		ref_avsync_scores = net(audios, ref_videos)
	
	cat_scores = torch.stack([ref_avsync_scores, avsync_scores], dim=1) # (b, 2)
	relsync = torch.softmax(cat_scores, dim=1)[:, 1].contiguous().detach().cpu()

	return relsync


@torch.no_grad()
def compute_alignsync(audios, videos, ref_videos, net, clip_net):
	'''
		videos in shape BCTHW in [0., 1.],
		audios in shape BCNT
	'''

	f = videos.shape[2]
	
	relsync = compute_relsync(audios, videos, net, ref_videos=ref_videos)
	
	# Compute AlignProb
	videos = torch.cat([
		ref_videos[:, :, 0:1], videos[:, :, 1:]
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
	ref_audio_waveform: Optional[torch.Tensor] = None,
	ref_audio_sr: Optional[int] = None,
	ref_video: Optional[torch.Tensor] = None,
	metric: str = "alignsync",
	device: torch.device = torch.device("cuda"),
	dtype: torch.dtype = torch.float32
):
	'''
		audio_waveform: loaded audio in shape (c t)
		audio_sr: sampling rate of loaded audio_waveforms
		video: input videos in shape (c t h w) in [0, 1]
		ref_audio_waveform: reference audio in  shape (c t), only used in relsync
		ref_audio_sr: reference audio' sampling rate. By default the same value as audio_sr
		ref_video: reference video in same shape as videos, only needed when computing relsyn/alignsync
		metric: 'alignsync', 'relsync', 'avsync_score'
	'''
	c, t, h, w = video.shape
	assert t == 12, "video should have 12 frames in 6 FPS"
	assert metric in ['alignsync', 'relsync', 'avsync_score']
	if metric == 'alignsync':
		assert ref_video is not None and ref_video.shape == video.shape, \
			f"To compute alignsync, ref_video is needed as reference, and in same shape as video"
	if metric == "relsync":
		assert (ref_audio_waveform is None) ^ (ref_video is None), \
			f"To compute relsync, either ref_audio_waveform or ref_video is needed as reference"
	
	avsync_net = load_avsync_model().to(device=device, dtype=dtype)
	if metric == "alignsync":
		clip_net = load_clip_model().to(device=device, dtype=dtype)
	
	# Convert audio into melspectrogram
	audio = torchaudio.functional.resample(audio_waveform, orig_freq=audio_sr, new_freq=16000)  # (c, t)
	audio = waveform_to_melspectrogram(audio)  # (1, n, t)
	audio = audio.unsqueeze(0).contiguous().to(device=device, dtype=dtype)
	video = video.unsqueeze(0).contiguous().to(device=device, dtype=dtype)
	
	if metric == "alignsync":
		# Computing alignsync by referring an video
		ref_video = ref_video.unsqueeze(0).contiguous().to(device=device, dtype=dtype)

		return compute_alignsync(audio, video, ref_video, avsync_net, clip_net)[0]
	
	elif metric == "relsync":
		if ref_audio_waveform is not None:
			# Compute relsync by referring an audio
			if ref_audio_sr is None:
				ref_audio_sr = audio_sr
			ref_audio = torchaudio.functional.resample(ref_audio_waveform, orig_freq=ref_audio_sr, new_freq=16000)  # (c, t)
			ref_audio = waveform_to_melspectrogram(ref_audio)  # (1, n, t)
			ref_audio = ref_audio.unsqueeze(0).contiguous().to(device=device, dtype=dtype)
			
			return compute_relsync(audio, video, avsync_net, ref_audios=ref_audio)[0]
		else:
			# Compute relsync by referring an video
			ref_video = ref_video.unsqueeze(0).contiguous().to(device=device, dtype=dtype)
			
			return compute_relsync(audio, video, avsync_net, ref_videos=ref_video)[0]
		
	# computing raw avsync score
	return compute_avsync_scores(audio, video, avsync_net)[0]
	
	
	
	
	
	