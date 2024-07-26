import random
import itertools
import json
import os.path as osp
from typing import Tuple, List, Union, Literal, Optional
from einops import rearrange

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

import torchvision
torchvision.set_video_backend("video_reader")
from torchvision.io import VideoReader

from avgen.data.utils import load_video_clip_from_videoreader, load_audio_clip_from_videoreader


class BaseAudioVideoDataset(Dataset):
	def __init__(
			self,
			example_list_path: str,
			data_root: str,
			mode: str = "test",
			video_fps: int = 6,
			video_num_frame: int = 12,
			img_size: Union[int, Tuple[int]] = 256,
			randflip: bool = False,
			randcrop: bool = False,
			example_list_type: str = "video",
			class_mapping_json: Optional[str] = None,
			class_text_encoding_mapping_pt: Optional[str] = None,
			category: Optional[Union[str, List[str]]] = None,
			sampling_rate: Optional[int] = 16000
	):
		super().__init__()
		
		assert example_list_type in ["video", "clip"]
		self.example_list_type = example_list_type
		with open(example_list_path, 'r') as f:
			example_list = f.readlines()
			example_list = [file.strip() for file in example_list]
			if category is not None:
				if isinstance(category, str): category = [category]
				example_list = [file for file in example_list if file.split('/')[0] in category]
				print(f"Only using category={','.join(category)} with {len(example_list)} samples")
			self.example_list = example_list
		
		self.data_root = data_root
		self.mode = mode
		self.video_fps = video_fps
		self.video_num_frame = video_num_frame
		self.clip_duration = video_num_frame / video_fps
		self.img_size = img_size
		self.randflip = randflip
		self.randcrop = randcrop
		self.sampling_rate = sampling_rate

		if class_mapping_json is not None and class_mapping_json:
			with open(class_mapping_json, 'r') as f:
				self.class_mapping = json.load(f)
		if class_text_encoding_mapping_pt is not None:
			self.class_text_encoding_mapping = torch.load(class_text_encoding_mapping_pt, map_location="cpu")
		
	def __len__(self):
		return len(self.example_list)
	
	def load_class_str(self, index):
		if hasattr(self, "class_mapping"):
			converted_class = self.example_list[index].split("/")[0]
			return self.class_mapping[converted_class]
		
		return None
	
	def load_class_text_encoding(self, prompt=None):
		if prompt is not None:
			assert prompt in self.class_text_encoding_mapping, f"{prompt} not in class text encoding mapping"
			text_encoding = self.class_text_encoding_mapping[prompt]
		else:
			text_encoding = self.class_text_encoding_mapping
		if not isinstance(text_encoding, torch.Tensor):
			text_encoding = torch.from_numpy(text_encoding)
		return text_encoding
	
	def __getitem__(self, index):
		if self.example_list_type == "clip":
			example_path, scene_start_timestamp, scene_end_timestamp = self.example_list[index].split(",")
			scene_start_timestamp, scene_end_timestamp = float(scene_start_timestamp), float(scene_end_timestamp)
			av_duration = scene_end_timestamp - scene_start_timestamp
		else:
			example_path = self.example_list[index]
			scene_start_timestamp = 0.0
			av_duration = None
		example_path = osp.join(self.data_root, example_path)
		
		av_reader = VideoReader(example_path, stream="video")
		meta_data = av_reader.get_metadata()
		video_duration, video_fps = float(meta_data["video"]["duration"][0]), float(meta_data["video"]["fps"][0])
		audio_duration, audio_sr = float(meta_data["audio"]["duration"][0]), int(meta_data["audio"]["framerate"][0])
		# assert video_fps >= 12, example_path
		
		if self.example_list_type == "video":
			av_duration = min(video_duration, audio_duration)
			
		if self.mode == "train":
			clip_start_timestamp = max(0., random.uniform(0., av_duration - self.clip_duration)) + scene_start_timestamp
		else:
			clip_start_timestamp = max(0., (av_duration - self.clip_duration) / 2.) + scene_start_timestamp

		video_frames = load_video_clip_from_videoreader(
			av_reader,
			clip_start_timestamp,
			clip_duration=self.clip_duration,
			video_fps=self.video_fps,
			video_num_frame=self.video_num_frame,
			image_size=self.img_size,
			flip=(self.mode == "train" and self.randflip and random.randint(0, 1)),
			randcrop=(self.mode == "train" and self.randcrop),
			normalize=False
		)
		video_frames = rearrange(video_frames, "f c h w -> c f h w")
		
		melspectrograms = load_audio_clip_from_videoreader(
			av_reader,
			clip_start_timestamp,
			clip_duration=self.clip_duration,
			audio_sr=audio_sr,
			load_audio_as_melspectrogram=True,
			target_audio_sr=self.sampling_rate
		)
		
		ret_dict = {
			"video": video_frames, # (c f h w)
			"audio": melspectrograms # (c n t)
		}
		
		if hasattr(self, "class_text_encoding_mapping"):
			class_str = self.load_class_str(index)
			text_encoding = self.load_class_text_encoding(class_str)
			ret_dict["class_text_encoding"] = text_encoding
	
		return ret_dict
