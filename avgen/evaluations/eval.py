from einops import rearrange
from typing import List, Union, Tuple
from collections import defaultdict
from tqdm import tqdm
import json
from glob import glob
import os
import torch

import torchvision
torchvision.set_video_backend("video_reader")

from avgen.evaluations.models.inception_v3 import load_inceptionv3_pretrained
from avgen.evaluations.models.download import load_i3d_pretrained
from avsync.models.avsync_classifier import load_avsync_model
from avgen.evaluations.models.clip import load_clip_model

from avgen.evaluations.fid import compute_fid_image_features
from avgen.evaluations.fvd import compute_fvd_video_features
from avgen.evaluations.dists import frechet_distance
from avgen.evaluations.avsync import compute_avsync_scores
from avgen.evaluations.clip import compute_clip_consistency

from avgen.data.utils import load_av_clips_uniformly


@torch.no_grad()
def evaluate_generation_results(
		groundtruth_video_root: str,
		groundtruth_video_names: List[str],
		groundtruth_categories: List[str],
		num_clips_per_video: int,
		generated_video_root: str,
		result_save_path: str,
		image_size: Union[int, Tuple[int, int]],
		video_fps: int = 6,
		video_num_frame: int = 12,
		eval_fid: bool = True,
		eval_fvd: bool = True,
		eval_clipsim: bool = True,
		eval_relsync: bool = True,
		eval_alignsync: bool = True,
		record_instance_metrics: bool = False,
		dtype: torch.dtype = torch.float32
):
	device = torch.device("cuda")
	
	# For each xxxx.mp4 file in @groundtruth_video_root, there should be @num_clips_per_video generated clips in @generated_video_root,
	#   with filenames xxxx_clip-{i}.mp4 whose audio is the same as the i-th uniformly sampled clip from groundtruth video.
	for groundtruth_video_name in groundtruth_video_names:
		num_generated_video_paths = len(
			glob(f"{generated_video_root}/{groundtruth_video_name.replace('.mp4', '')}*.mp4"))
		assert num_generated_video_paths == num_clips_per_video, \
			f'number of generated videos({num_generated_video_paths}) does not equal to num_clips_per_video({num_clips_per_video}) for {groundtruth_video_name}'
	
	result_dict = {
		"groundtruth_video_root": groundtruth_video_root,
		"generated_video_root": generated_video_root,
		"num_clips_per_video": num_clips_per_video
	}
	
	if eval_alignsync: assert eval_clipsim and eval_relsync
	
	#####################################################################################
	# 1. Prepare eval models and feature lists
	if eval_fid:
		iv3_fid = load_inceptionv3_pretrained(block_ids=[3], use_fid_inception=True).to(device=device, dtype=dtype)
		groundtruth_fid_features = []
		generated_fid_features = []
	if eval_fvd:
		i3d = load_i3d_pretrained(device).to(device=device, dtype=dtype)
		groundtruth_fvd_features = []
		generated_fvd_features = []
	if eval_clipsim:
		clip_model = load_clip_model().to(device=device, dtype=dtype)
		generated_ias = []
		generated_its = []
	if eval_relsync:
		avsync_net = load_avsync_model().to(device=device, dtype=dtype)
		groundtruth_avsync_scores = []
		generated_avsync_scores = []
	if eval_alignsync:
		groundtruth_first_frame_ia_sims = []
		generated_pred_frame_ia_sims = []
	
	#####################################################################################
	# 2. Compute features for groundtruth clips in sorted order
	groundtruth_video_names.sort()
	for groundtruth_video_name, groundtruth_category in tqdm(
			zip(groundtruth_video_names, groundtruth_categories),
			total=len(groundtruth_video_names),
			desc="Computing features for sampled groundtruth clips ..."
	):
		
		# Uniformly sample audio and video clips from groundtruth video
		# audio of shape (b 1 n t) in melspectrogram
		# video of shape (b f c h w) in [0, 1]
		groundtruth_video_path = os.path.join(groundtruth_video_root, groundtruth_video_name)
		video_tensors, audio_tensors = load_av_clips_uniformly(
			video_path=groundtruth_video_path,
			video_fps=video_fps,
			video_num_frame=video_num_frame,
			image_size=image_size,
			num_clips=num_clips_per_video,
			load_audio_as_melspectrogram=True
		)
		video_tensors = video_tensors.to(device=device, dtype=dtype)
		audio_tensors = audio_tensors.to(device=device, dtype=dtype)
		
		if eval_fid:
			fid_features = compute_fid_image_features(
				rearrange(video_tensors, "b f c h w -> (b f) c h w"),
				iv3_fid
			).detach().cpu()
			fid_features = rearrange(fid_features, "(b f) c -> b f c", f=video_num_frame) # (b * f, c)
			groundtruth_fid_features.append(fid_features)
		
		if eval_fvd:
			fvd_features = compute_fvd_video_features(
				rearrange(video_tensors, "b f c h w -> b c f h w"),
				i3d
			).detach().cpu() # (b, c)
			groundtruth_fvd_features.append(fvd_features)
		
		if eval_alignsync:
			groundtruth_first_frame_ia_sims.append(
				compute_clip_consistency(
					video_tensors[:, 0:1], audio_tensors, net=clip_model
				)["ia_sim"].detach().cpu()
			) # (b, 1)
		
		if eval_relsync:
			groundtruth_avsync_scores.append(
				compute_avsync_scores(
					audio_tensors,
					rearrange(video_tensors, "b f c h w -> b c f h w"),
					avsync_net
				).detach().cpu()  # (b)
			)
	
	#####################################################################################
	# 3. Compute features for generated clips in the same order as groundtruth sampled clips
	for groundtruth_video_name, groundtruth_category in tqdm(
			zip(groundtruth_video_names, groundtruth_categories),
			total=len(groundtruth_video_names),
			desc="Computing features for generated clips ..."
	):
		generated_video_paths = glob(f"{generated_video_root}/{groundtruth_video_name.replace('.mp4', '')}*.mp4")
		generated_video_paths.sort()
		for generated_video_path in generated_video_paths:
			# audio of shape (1 1 n t) in melspectrogram
			# video of shape (1 f c h w) in [0, 1]
			video_tensors, audio_tensors = load_av_clips_uniformly(
				video_path=generated_video_path,
				video_fps=video_fps,
				video_num_frame=video_num_frame,
				image_size=image_size,
				num_clips=1,
				load_audio_as_melspectrogram=True
			)
			video_tensors = video_tensors.to(device=device, dtype=dtype)
			audio_tensors = audio_tensors.to(device=device, dtype=dtype)
		
			if eval_fid:
				fid_features = compute_fid_image_features(
					rearrange(video_tensors, "b f c h w -> (b f) c h w"),
					iv3_fid
				).detach().cpu()
				fid_features = rearrange(fid_features, "(b f) c -> b f c", f=video_num_frame)
				generated_fid_features.append(fid_features)
			
			if eval_fvd:
				fvd_features = compute_fvd_video_features(
					rearrange(video_tensors, "b f c h w -> b c f h w"),
					i3d
				).detach().cpu() # (b, c)
				generated_fvd_features.append(fvd_features)
			
			if eval_clipsim:
				# Obtain categories as text
				texts = [groundtruth_category]
				clipsim_results = compute_clip_consistency(
					video_tensors, audio_tensors, texts, net=clip_model
				)
				ia_sims = clipsim_results["ia_sim"].detach().cpu()[:, 1:] # (b, f-1)
				it_sims = clipsim_results["it_sim"].detach().cpu()[:, 1:] # (b, f-1)
				
				generated_ias.append(ia_sims.mean(dim=1))
				generated_its.append(it_sims.mean(dim=1))
				
				if eval_alignsync:
					generated_pred_frame_ia_sims.append(ia_sims)
	
			if eval_relsync:
				generated_avsync_scores.append(
					compute_avsync_scores(
						audio_tensors,
						rearrange(video_tensors, "b f c h w -> b c f h w"),
						avsync_net
					).detach().cpu()  # (b)
				)
	
	#####################################################################################
	# 4. Compute metrics from computed features
	if eval_fid:
		# Exclude first frame
		groundtruth_fid_features = torch.cat(groundtruth_fid_features)[:, 1:].flatten(end_dim=1) # (B * (f-1), c)
		generated_fid_features = torch.cat(generated_fid_features)[:, 1:].flatten(end_dim=1) # (B * (f-1), c)
		fid_score = frechet_distance(groundtruth_fid_features, generated_fid_features)
		result_dict["FID"] = fid_score.item()
	
	if eval_fvd:
		groundtruth_fvd_features = torch.cat(groundtruth_fvd_features)
		generated_fvd_features = torch.cat(generated_fvd_features)
		fvd_score = frechet_distance(groundtruth_fvd_features, generated_fvd_features)
		result_dict["FVD"] = fvd_score.item()
	
	if eval_clipsim:
		generated_ias = torch.cat(generated_ias)
		generated_its = torch.cat(generated_its)
		result_dict.update({
			"IA_mean": generated_ias.mean().item(),
			"IA_std": generated_ias.std().item(),
			"IT_mean": generated_its.mean().item(),
			"IT_std": generated_its.std().item(),
		})
	
	if eval_relsync:
		groundtruth_avsync_scores = torch.cat(groundtruth_avsync_scores)
		generated_avsync_scores = torch.cat(generated_avsync_scores)
		generated_relsync_scores = torch.exp(generated_avsync_scores) / (torch.exp(groundtruth_avsync_scores) + torch.exp(generated_avsync_scores))
		result_dict.update({
			"RelSync_mean": generated_relsync_scores.mean().item(),
			"RelSync_std": generated_relsync_scores.std().item(),
		})
	
	if eval_alignsync:
		groundtruth_first_frame_ia_sims = torch.cat(groundtruth_first_frame_ia_sims) # (B, 1)
		generated_pred_frame_ia_sims = torch.cat(generated_pred_frame_ia_sims) # (B, f-1)
		generated_align_probs = (
				torch.exp(generated_pred_frame_ia_sims) / (torch.exp(groundtruth_first_frame_ia_sims) + torch.exp(generated_pred_frame_ia_sims))
		).mean(dim=1)
		generated_alignsync_scores = generated_align_probs * generated_relsync_scores
		result_dict.update({
			"AlignSync_mean": generated_alignsync_scores.mean().item(),
			"AlignSync_std": generated_alignsync_scores.std().item(),
		})
	
	#####################################################################################
	# 5. Record metrics for per generated clip
	if record_instance_metrics:
		
		instance_result_dict = defaultdict(dict)
		
		for video_index, groundtruth_video_name in tqdm(
				enumerate(groundtruth_video_names),
				total=len(groundtruth_video_names),
				desc="Recording per-clip metrics ..."
		):
			generated_video_paths = glob(f"{generated_video_root}/{groundtruth_video_name.replace('.mp4', '')}*.mp4")
			generated_video_paths.sort()
			generated_video_names = [generated_video_path.replace(f"{generated_video_root}/", "") for generated_video_path in generated_video_paths]
			
			for clip_index, generated_video_name in enumerate(generated_video_names):
				id = video_index * num_clips_per_video + clip_index
				instance_result_dict[generated_video_name] = {}
				if eval_clipsim:
					instance_result_dict[generated_video_name]["IA"] = generated_ias[id].item()
					instance_result_dict[generated_video_name]["IT"] = generated_ias[id].item()
				if eval_relsync:
					instance_result_dict[generated_video_name]["RelSync"] = generated_relsync_scores[id].item()
				if eval_alignsync:
					instance_result_dict[generated_video_name]["AlignSync"] = generated_alignsync_scores[id].item()
		
		result_dict["instance_metrics"] = instance_result_dict
	
	os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
	with open(result_save_path, "w") as f:
		json.dump(result_dict, f, indent=4)
		
	return result_dict
	
			
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		