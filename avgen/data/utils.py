from typing import List, Union, Literal, Tuple
import itertools
import PIL
from einops import rearrange

import numpy as np
import torch
import torchaudio
import torchvision
from torchvision.io import VideoReader
torchvision.set_video_backend("video_reader")
import torchvision.transforms as transforms

from transformers import ImageProcessingMixin

from submodules.ImageBind.imagebind.data import waveform2melspec


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

VIDEO_SAMPLING_TYPES = Literal["random", "center"]
AUDIO_SAMPLING_TYPES = Literal["random", "center"]


def waveform_to_melspectrogram(
		waveform: Union[np.ndarray, torch.Tensor],
		num_mel_bins=128,
		target_length=204,
		sample_rate=16000,
		clip_duration=2.,
		mean=-4.268,
		std=9.138
):
	if isinstance(waveform, np.ndarray):
		waveform = torch.from_numpy(waveform)
	
	audio_length = waveform.shape[1]
	audio_target_length = int(clip_duration * sample_rate)
	
	audio_start_idx = 0
	if audio_length > audio_target_length:
		audio_start_idx = (audio_length - audio_target_length) // 2
	audio_end_idx = audio_start_idx + audio_target_length
	waveform_clip = waveform[:, audio_start_idx:audio_end_idx]
	
	waveform_melspec = waveform2melspec(
		waveform_clip, sample_rate, num_mel_bins, target_length
	)  # (1, n_mel, n_frame)
	
	normalize = transforms.Normalize(mean=mean, std=std)
	
	audio_clip = normalize(waveform_melspec)
	
	return audio_clip  # (1, freq, time)


class AudioMelspectrogramExtractor(ImageProcessingMixin):
	
	def __init__(
		self,
		num_mel_bins=128,
		target_length=204,
		sample_rate=16000,
		clip_duration=2,
		mean=-4.268,
		std=9.138
	):
		super().__init__()
		self.num_mel_bins = num_mel_bins
		self.target_length = target_length
		self.sample_rate = sample_rate
		self.clip_duration = clip_duration
		self.mean = mean
		self.std = std
	
	@property
	def max_length_s(self) -> int:
		return self.clip_duration
	
	@property
	def sampling_rate(self) -> int:
		return self.sample_rate
	
	def __call__(
			self,
			waveforms: Union[
				np.ndarray,
				torch.Tensor,
				List[np.ndarray],
				List[torch.Tensor]
			]
	):
		if isinstance(waveforms, (np.ndarray, torch.Tensor)) and waveforms.ndim == 2:
			waveforms = [waveforms, ]
		features = []
		
		for waveform in waveforms:
			feature = waveform_to_melspectrogram(
				waveform=waveform,
				num_mel_bins=self.num_mel_bins,
				target_length=self.target_length,
				sample_rate=self.sample_rate,
				clip_duration=self.clip_duration,
				mean=self.mean,
				std=self.std
			)
			features.append(feature)
		features = torch.stack(features, dim=0)
		
		return features # (b c n t)


def load_and_transform_images_stable_diffusion(
		images: Union[List[np.ndarray], torch.Tensor, np.ndarray],
		size=512,
		flip=False,
		randcrop=False,
		normalize=True
):
	"""
	@images: (List of) np.uint8 images of shape (h, w, 3)
			or tensor of shape (b, c, h, w) in [0., 1.0]

	"""
	
	assert isinstance(images, (List, torch.Tensor, np.ndarray)), type(images)
	if isinstance(images, List):
		assert isinstance(images[0], np.ndarray)
		assert images[0].dtype == np.uint8
		assert images[0].shape[2] == 3
		
		# convert np images into torch float tensor
		images = torch.from_numpy(
			rearrange(np.stack(images, axis=0), "f h w c -> f c h w")
		).float() / 255.
	elif isinstance(images, np.ndarray):
		assert isinstance(images, np.ndarray)
		assert images.dtype == np.uint8
		assert images.shape[3] == 3
		
		# convert np images into torch float tensor
		images = torch.from_numpy(
			rearrange(images, "f h w c -> f c h w")
		).float() / 255.
		
	assert images.shape[1] == 3
	assert torch.all(images<= 1.0) and torch.all(images >= 0.0)
	
	h, w = images.shape[-2:]
	if isinstance(size, int):
		target_h, target_w = size, size
	else:
		target_h, target_w = size
	
	# first crop the image
	target_aspect_ratio = float(target_h) / target_w
	curr_aspect_ratio = float(h) / w
	if target_aspect_ratio >= curr_aspect_ratio: # trim w
		trimmed_w = int(h / target_aspect_ratio)
		images = images[:, :, :, (w-trimmed_w)//2: (w-trimmed_w)//2+trimmed_w]
	else: # trim h
		trimmed_h = int(w * target_aspect_ratio)
		images = images[:, :, (h - trimmed_h) // 2: (h - trimmed_h) // 2 + trimmed_h]
	
	transform_list = [
		transforms.Resize(
			size,
			interpolation=transforms.InterpolationMode.BILINEAR,
			antialias=True
		),
	]
	
	# assert not randcrop
	if randcrop:
		transform_list.append(transforms.RandomCrop(size))
	else:
		transform_list.append(transforms.CenterCrop(size))
		
	if flip:
		transform_list.append(transforms.RandomHorizontalFlip(p=1.0))

	if normalize:
		transform_list.append(transforms.Normalize([0.5], [0.5]))
	
	data_transform = transforms.Compose(transform_list)
	
	images = data_transform(images)
	return images


def load_video_clip_from_videoreader(
		av_reader,
		clip_start_timestamp,
		clip_duration,
		video_fps,
		video_num_frame,
		image_size,
		flip=False,
		randcrop=False,
		normalize=False
):
	av_reader.set_current_stream("video")
	keyframe_coverage = 1. / video_fps
	
	video_frames = []
	frame_timestamp = clip_start_timestamp
	for i, frame in enumerate(itertools.takewhile(
			lambda x: x['pts'] <= clip_start_timestamp + clip_duration + keyframe_coverage / 2.,
			av_reader.seek(max(clip_start_timestamp, 0.))
	)):
		if frame["pts"] >= frame_timestamp:
			video_frames.append(frame["data"])  # (c, h, w) tensor [0, 255]
			frame_timestamp += keyframe_coverage
		
		if len(video_frames) == video_num_frame:
			break
	
	if len(video_frames) < video_num_frame:
		res_length = video_num_frame - len(video_frames)
		for _ in range(res_length):
			video_frames.append(video_frames[-1])
	
	video_frames = torch.stack(video_frames, dim=0).float() / 255.
	
	video_frames = load_and_transform_images_stable_diffusion(
		video_frames,
		size=image_size,
		flip=flip,
		randcrop=randcrop,
		normalize=normalize
	).float()  # (n_frame, 3, h, w) in range [0., 1.]
	
	return video_frames


def load_audio_clip_from_videoreader(
		av_reader,
		clip_start_timestamp,
		clip_duration,
		audio_sr,
		load_audio_as_melspectrogram,
		target_audio_sr=16000
):
	av_reader.set_current_stream("audio")
	
	audio_frames = []
	for frame in itertools.takewhile(
			lambda x: x['pts'] <= clip_start_timestamp + clip_duration,
			av_reader.seek(clip_start_timestamp)
	):
		if frame['pts'] >= clip_start_timestamp and \
				frame['pts'] <= clip_start_timestamp + clip_duration:
			frame_data = frame["data"]
			t, c = frame_data.shape
			frame_data = frame_data.contiguous().view(c, t).contiguous()
			audio_frames.append(frame_data)  # (c, t)
	
	audio = torchaudio.functional.resample(
		torch.cat(audio_frames, dim=1),
		orig_freq=audio_sr,
		new_freq=target_audio_sr
	)  # (C, T)
	
	if load_audio_as_melspectrogram:
		audio = waveform_to_melspectrogram(audio)  # (1, n, t)
	
	return audio


def load_av_clips_uniformly(
		video_path: str,
		video_fps: int = 6,
		video_num_frame: int = 12,
		image_size: Union[int, Tuple[int, int]] = 512,
		num_clips: int = 1,
		load_audio_as_melspectrogram: bool = True,
):
	'''
	Return:
		video_frames: (b f c h w) in [0, 1]
		audio_frames:
			if load_audio_as_melspectrogram is True: (b 1 n t)
			else: List of tensors (b c ti), ti can be different
	'''
	clip_duration = video_num_frame / video_fps
	av_reader = VideoReader(video_path, stream="video")
	meta_data = av_reader.get_metadata()
	video_duration, orig_video_fps = float(meta_data["video"]["duration"][0]), float(meta_data["video"]["fps"][0])
	audio_duration, audio_sr = float(meta_data["audio"]["duration"][0]), int(meta_data["audio"]["framerate"][0])
	av_duration = min(video_duration, audio_duration)
	# assert av_duration >= clip_duration, [video_path, video_duration, audio_duration]
	
	# 1. Sample clip start times
	if num_clips == 1:
		clip_start_timestamps = np.array([(av_duration - clip_duration) / 2.])
	else:
		clip_start_timestamps = np.linspace(0., av_duration - clip_duration, endpoint=True, num=num_clips)
	
	video_frames = []
	audio_frames = []
	for clip_start_timestamp in clip_start_timestamps:
		video_frames.append(
			load_video_clip_from_videoreader(
				av_reader,
				clip_start_timestamp,
				clip_duration,
				video_fps,
				video_num_frame,
				image_size,
				flip=False,
				randcrop=False,
				normalize=False
			)
		)
		audio_frames.append(
			load_audio_clip_from_videoreader(
				av_reader,
				clip_start_timestamp,
				clip_duration,
				audio_sr,
				load_audio_as_melspectrogram
			)
		)
	
	video_frames = torch.stack(video_frames)  # (b, t, c, h, w)
	if load_audio_as_melspectrogram:
		audio_frames = torch.stack(audio_frames)  # (b, 1, c, t)
	
	return video_frames, audio_frames


def load_video_clips_uniformly(
		video_path: str,
		video_fps: int = 6,
		video_num_frame: int = 12,
		image_size: Union[int, Tuple[int, int]] = 512,
		num_clips: int = 1
):
	'''
	Return:
		video_frames: (b f c h w) in [0, 1]
	'''
	clip_duration = video_num_frame / video_fps
	av_reader = VideoReader(video_path, stream="video")
	meta_data = av_reader.get_metadata()
	video_duration, orig_video_fps = float(meta_data["video"]["duration"][0]), float(meta_data["video"]["fps"][0])
	
	# 1. Sample clip start times
	if num_clips == 1:
		clip_start_timestamps = np.array([(video_duration - clip_duration) / 2.])
	else:
		clip_start_timestamps = np.linspace(0., video_duration - clip_duration, endpoint=True, num=num_clips)
	
	video_frames = []
	for clip_start_timestamp in clip_start_timestamps:
		video_frames.append(
			load_video_clip_from_videoreader(
				av_reader,
				clip_start_timestamp,
				clip_duration,
				video_fps,
				video_num_frame,
				image_size
			)
		)
	
	video_frames = torch.stack(video_frames)  # (b, t, c, h, w)
	
	return video_frames


def load_image(image_path, image_size = (256, 256)):
	'''
	Return:
		image: tensor (3, h, w) in [0, 1]
	'''
	image = PIL.Image.open(image_path).convert('RGB')
	image = torch.from_numpy(np.array(image))
	image = rearrange(image, "h w c -> 1 c h w") / 255.
	
	image = load_and_transform_images_stable_diffusion(
		image, size=image_size, flip=False, randcrop=False, normalize=False
	)[0].contiguous()
	
	return image


def load_audio_clips_uniformly(
		audio_path: str,
		clip_duration: float = 2.0,
		num_clips: int = 1,
		load_audio_as_melspectrogram: bool = True
):
	'''
	Return:
		audio_frames:
			if load_audio_as_melspectrogram is True: (b 1 n t)
			else: List of b tensors (c t)
	'''
	audio, sr = torchaudio.load(audio_path)
	audio_duration = audio.shape[1] / float(sr)
	
	audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
	
	# 1. Sample clip start times
	if num_clips == 1:
		clip_start_timestamps = np.array([(audio_duration - clip_duration) / 2.])
	else:
		clip_start_timestamps = np.linspace(0., audio_duration - clip_duration, endpoint=True, num=num_clips)
	
	audio_frames = []
	for clip_start_timestamp in clip_start_timestamps:
		
		audio_clip = audio[:, int(clip_start_timestamp*16000):int((clip_start_timestamp +clip_duration) *16000)].contiguous()
		if load_audio_as_melspectrogram:
			audio_clip = waveform_to_melspectrogram(audio_clip)
		audio_frames.append(audio_clip)

	if load_audio_as_melspectrogram:
		audio_frames = torch.stack(audio_frames)  # (b, 1, c, t)
	
	return audio_frames


def get_avsync15_evaluation_data():
	dataset_root = f"./datasets/AVSync15"
	video_root = f"{dataset_root}/videos"
	
	with open(f"{dataset_root}/test.txt", "r") as f:
		video_paths = [file.strip() for file in f.readlines()]
		categories = [file.split('/')[0] for file in video_paths]
	
	return video_root, video_paths, categories


def get_thegreatesthits_evaluation_data():
	dataset_root = f"./datasets/TheGreatestHits"
	video_root = f"{dataset_root}/videos"
	
	with open(f"{dataset_root}/test.txt", "r") as f:
		video_paths = [file.strip() for file in f.readlines()]
	categories = ["hitting with a stick"] * len(video_paths)
	
	return video_root, video_paths, categories


def get_landscapes_evaluation_data():
	dataset_root = f"./datasets/Landscapes"
	video_root = f"{dataset_root}/videos/test"
	
	with open(f"{dataset_root}/test.txt", "r") as f:
		video_paths = [file.strip() for file in f.readlines()]
	categories = [file.split('/')[0] for file in video_paths]
	
	return video_root, video_paths, categories


def get_evaluation_data(dataset):
	video_path_type = "video"
	
	if dataset == "AVSync15":
		video_root, video_paths, categories = get_avsync15_evaluation_data()
	elif dataset == "TheGreatestHits":
		video_root, video_paths, categories = get_thegreatesthits_evaluation_data()
	elif dataset == "Landscapes":
		video_root, video_paths, categories = get_landscapes_evaluation_data()
	else:
		raise Exception()
	
	return video_root, video_paths, categories, video_path_type


