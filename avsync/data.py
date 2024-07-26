import os.path as osp
import random
import itertools

import numpy as np
from einops import rearrange

import torch
import torch.nn
import torchvision
import torchaudio

from torch.utils.data import Dataset
from torchvision.io import VideoReader
from torchvision import transforms
torchvision.set_video_backend("video_reader")

from avgen.data.utils import waveform_to_melspectrogram


def uniform_sample(start, end, num, endpoint=True):
	'''
		Uniform sample @num numbers in range [@start, @end]
		if endpoint=True, start and end are included
	'''
	if endpoint:
		return np.linspace(start, end, num, endpoint=True)
	
	gap = (end - start) / num
	return np.linspace(start + gap / 2, end - gap / 2, num, endpoint=True)


def random_compact_sample(start, end, num, gap):
	'''
		Random sample @num numbers in range [@start, @end] with gap of @gap
	'''
	assert (num - 1) * gap <= end - start, f"sample range {start} to {end} not enough to sample {num} points with gap>={gap}"
	
	first_number = random.uniform(start, end - (num - 1) * gap)
	sampled_numbers = np.arange(num) * gap + first_number
	
	return sampled_numbers


def center_compact_sample(start, end, num, gap):
	'''
		Sample @num numbers in range [@start, @end] with gap of @gap in the center
	'''
	assert (num - 1) * gap <= end - start, f"sample range {start} to {end} not enough to sample {num} points with gap>={gap}"
	
	first_number = start + (end - start - (num - 1) * gap) / 2.
	sampled_numbers = np.arange(num) * gap + first_number
	
	return sampled_numbers


def random_sample(start, end, num, gap):
	'''
		Random sample @num numbers in range [@start, @end] with any two numbers having gap>=@gap
	'''
	assert (num - 1) * gap <= end - start, f"sample range {start} to {end} not enough to sample {num} points with gap>={gap}"
	
	def _random_sample_one_number(start, end, num_to_sample, gap):
		end -= (num_to_sample - 1) * gap
		return random.uniform(start, end)
	
	sampled_list = []
	while num:
		sampled_num = _random_sample_one_number(start, end, num, gap)
		sampled_list.append(sampled_num)
		
		start = sampled_num + gap
		num -= 1
	
	return np.array(sampled_list)


class AudioVideoAlignedMultiPairDataset(Dataset):
	
	def __init__(
			self,
			data_root: str,
			example_list_path: str,
			mode: str = "test",
			image_size: int = 224,
			video_fps: int = 6,
			video_num_frames: int = 12,
			audio_sample_rate: int = 16000,
			randflip: bool = True,
			shift_time: float = 0.2,
			num_clips: int = 21,
			sampling_type: str = "random-compact",
	):
		super().__init__()
		
		self.data_root = data_root
		self.mode = mode
		self.image_size = image_size
		self.video_fps = video_fps
		self.video_num_frames = video_num_frames
		self.clip_duration = video_num_frames / video_fps
		self.audio_sample_rate = audio_sample_rate
		self.randflip = randflip
		self.shift_time = shift_time
		self.num_clips = num_clips
		assert sampling_type in ["random-compact", "center-compact", "random", "uniform"]
		self.sampling_type = sampling_type
		
		with open(example_list_path, "r") as f:
			examples = f.readlines()
			examples = [example.strip() for example in examples]
			self.examples = examples
	
	def __len__(self):
		return len(self.examples)
	
	def load_multi_video_clips(
			self,
			av_reader,
			clips_frame_seconds,
	):
		# To save memory, we first obtain transformed video frames,
		# Then we assign them to each clip
		av_reader.set_current_stream("video")
		
		video_frames = []
		video_frames_seconds = []
		for i, frame in enumerate(itertools.takewhile(
				lambda x: x['pts'] <= clips_frame_seconds[-1, -1],
				av_reader.seek(clips_frame_seconds[0, 0])
		)):
			video_frames.append(frame["data"])
			video_frames_seconds.append(frame["pts"])
		video_frames_seconds = np.array(video_frames_seconds)  # (F)
		video_frames = torch.stack(video_frames, dim=0).float() / 255.  # (F c h w)
		
		# transform
		transform_list = [
			transforms.Resize(
				self.image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
			),
			transforms.CenterCrop(self.image_size),
			transforms.Normalize(
				mean=(0.48145466, 0.4578275, 0.40821073),
				std=(0.26862954, 0.26130258, 0.27577711),
			)
		]
		transform_func = transforms.Compose(transform_list)
		video_frames = transform_func(video_frames) # (kf c h w)
		
		# Assign frames to each clip
		clips_frame_indices = np.abs(
			clips_frame_seconds[:, :, None] - video_frames_seconds[None, None, :]
		).argmin(axis=2)
		clips_frame_indices = clips_frame_indices.flatten()
		video_frames = video_frames[clips_frame_indices]
		video_frames = rearrange(
			video_frames, "(k f) c h w -> k c f h w",
			k=self.num_clips, f=self.video_num_frames
		)
		
		# Each clip is independently randomly flipped
		if self.randflip:
			for i in range(self.num_clips):
				if random.randint(0, 1):
					video_frames[i] = torch.flip(video_frames[i], dims=(3,))
			video_frames = video_frames.contiguous()
		
		return video_frames
	
	def load_multi_audio_clips(
			self,
			av_reader,
			clips_frame_seconds
	):
		audio_sr = int(av_reader.get_metadata()["audio"]["framerate"][0])
		
		av_reader.set_current_stream("audio")
		
		audio_clips = [[] for _ in range(len(clips_frame_seconds))]
		for frame in itertools.takewhile(
				lambda x: x['pts'] <= clips_frame_seconds[-1, -1],
				av_reader.seek(clips_frame_seconds[0, 0])
		):
			for i in range(len(clips_frame_seconds)):
				if frame['pts'] >= clips_frame_seconds[i, 0] and \
						frame['pts'] <= clips_frame_seconds[i, -1]:
					frame_data = frame["data"]
					t, c = frame_data.shape
					frame_data = frame_data.contiguous().view(c, t).contiguous()
					audio_clips[i].append(frame_data)  # (c, t)
		audio_clips = [torch.cat(audio_clip, dim=1) for audio_clip in audio_clips]
		audio_clips = [
			torchaudio.functional.resample(
				audio_clip,
				orig_freq=audio_sr,
				new_freq=self.audio_sample_rate
			) for audio_clip in audio_clips
		]
		audio_melspectrograms = torch.stack([
			waveform_to_melspectrogram(audio_clip) for audio_clip in audio_clips
		]) # (k, 1, n, t)
		
		return audio_melspectrograms

	def __getitem__(self, index):
		
		try:
			# 1. Load meta data
			example_path = osp.join(self.data_root, self.examples[index])
			
			av_reader = VideoReader(example_path, stream="video")
			meta_data = av_reader.get_metadata()
			video_duration, video_fps = float(meta_data["video"]["duration"][0]), float(meta_data["video"]["fps"][0])
			audio_duration, audio_sr = float(meta_data["audio"]["duration"][0]), int(meta_data["audio"]["framerate"][0])
			assert video_fps >= 12, example_path
			av_duration = min(video_duration, audio_duration)
			scene_start_timestamp = 0.
			scene_end_timestamp = av_duration
			
			shift_duration = (self.num_clips-1) * self.shift_time
			assert av_duration >= self.clip_duration + shift_duration, example_path
	
			# 2. Sampling clips' start timestamps
			clip_range_start = scene_start_timestamp
			clip_range_end = scene_end_timestamp - self.clip_duration
			if self.sampling_type == "random-compact":
				sampled_clips_start_timestamps = random_compact_sample(clip_range_start, clip_range_end, self.num_clips, self.shift_time)
			elif self.sampling_type == "center-compact":
				sampled_clips_start_timestamps = center_compact_sample(clip_range_start, clip_range_end, self.num_clips, self.shift_time)
			elif self.sampling_type == "random":
				sampled_clips_start_timestamps = random_sample(clip_range_start, clip_range_end, self.num_clips, self.shift_time)
			elif self.sampling_type == "uniform":
				sampled_clips_start_timestamps = uniform_sample(clip_range_start, clip_range_end, self.num_clips, endpoint=True)
			else:
				raise Exception()
				
			# Here, we append one to include the last frame for audio sampling
			frame_seconds_per_clip = np.arange(self.video_num_frames + 1) / self.video_fps
			clips_frame_seconds = sampled_clips_start_timestamps[:, None] + frame_seconds_per_clip[None, :]  # (k, f)
			# avoid endpoint rounding error
			clips_frame_seconds[:, 0] += 0.0001
			clips_frame_seconds[:, -1] -= 0.0001
			
			# 3. Load av
			videos = self.load_multi_video_clips(av_reader, clips_frame_seconds[:, :-1])  # p, c, f, h, w
			# append last frame
			audios = self.load_multi_audio_clips(av_reader, clips_frame_seconds)  # p, c, n, t
	
			return {
				"index": index,
				"video": videos,
				"audio": audios
			}

		except:
			return self.__getitem__((index + 1) % len(self.examples))
