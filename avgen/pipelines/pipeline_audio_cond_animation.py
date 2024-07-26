import torchvision.io
from einops import rearrange, repeat
import numpy as np
import inspect
from typing import List, Optional, Union, Tuple
from tqdm import tqdm
import json

import os
import PIL
import torch

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers, PNDMScheduler
from diffusers.utils import logging
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor

from avgen.models.unets import AudioUNet3DConditionModel
from avgen.models.audio_encoders import ImageBindSegmaskAudioEncoder
from avgen.data.utils import AudioMelspectrogramExtractor, get_evaluation_data, load_av_clips_uniformly, load_audio_clips_uniformly, load_image
from avgen.utils import freeze_and_make_eval

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class AudioCondAnimationPipeline(DiffusionPipeline, TextualInversionLoaderMixin):
	"""
	Pipeline for text-guided image to image generation using stable unCLIP.

	This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
	library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

	Args:
		feature_extractor ([`CLIPImageProcessor`]):
			Feature extractor for image pre-processing before being encoded.
		unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
		scheduler ([`KarrasDiffusionSchedulers`]):
			A scheduler to be used in combination with `unet` to denoise the encoded image latents.
		vae ([`AutoencoderKL`]):
			Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
	"""
	text_encoder: CLIPTextModel
	tokenizer: CLIPTokenizer
	unet: AudioUNet3DConditionModel
	scheduler: KarrasDiffusionSchedulers
	vae: AutoencoderKL
	audio_encoder: ImageBindSegmaskAudioEncoder
	
	def __init__(
			self,
			text_encoder: CLIPTextModel,
			tokenizer: CLIPTokenizer,
			unet: AudioUNet3DConditionModel,
			scheduler: KarrasDiffusionSchedulers,
			vae: AutoencoderKL,
			audio_encoder: ImageBindSegmaskAudioEncoder,
			null_text_encodings_path: str = ""
	):
		super().__init__()
		
		self.register_modules(
			text_encoder=text_encoder,
			tokenizer=tokenizer,
			unet=unet,
			scheduler=scheduler,
			vae=vae,
			audio_encoder=audio_encoder
		)
		
		if null_text_encodings_path:
			self.null_text_encoding = torch.load(null_text_encodings_path).view(1, 77, 768)
		
		self.melspectrogram_shape = (128, 204)
		
		self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
		self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
		self.audio_processor = AudioMelspectrogramExtractor()
	
	@torch.no_grad()
	def encode_text(
			self,
			texts,
			device,
			dtype,
			do_text_classifier_free_guidance,
			do_audio_classifier_free_guidance,
			text_encodings: Optional[torch.Tensor] = None
	):
		if text_encodings is None:
			
			text_inputs = self.tokenizer(
				texts,
				padding="max_length",
				max_length=self.tokenizer.model_max_length,
				truncation=True,
				return_tensors="pt",
			)
			text_input_ids = text_inputs.input_ids
			
			if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
				attention_mask = text_inputs.attention_mask.to(device)
			else:
				attention_mask = None

			text_encodings = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
			text_encodings = text_encodings[0] # (b, n, d)
			
		else:
			if isinstance(text_encodings, (List, Tuple)):
				text_encodings = torch.cat(text_encodings)
		
		text_encodings = text_encodings.to(dtype=dtype, device=device)
		batch_size = len(text_encodings)
		
		# get unconditional embeddings for classifier free guidance
		if do_text_classifier_free_guidance:
			if not hasattr(self, "null_text_encoding"):
				uncond_token = ""
	
				max_length = text_encodings.shape[1]
				uncond_input = self.tokenizer(
					uncond_token,
					padding="max_length",
					max_length=max_length,
					truncation=True,
					return_tensors="pt",
				)
				
				if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
					attention_mask = uncond_input.attention_mask.to(device)
				else:
					attention_mask = None
				
				uncond_text_encodings = self.text_encoder(
					uncond_input.input_ids.to(device),
					attention_mask=attention_mask,
				)
				uncond_text_encodings = uncond_text_encodings[0]
				
			else:
				uncond_text_encodings = self.null_text_encoding
			
			uncond_text_encodings = repeat(uncond_text_encodings, "1 n d -> b n d", b=batch_size).contiguous()
			uncond_text_encodings = uncond_text_encodings.to(dtype=dtype, device=device)
		
		if do_text_classifier_free_guidance and do_audio_classifier_free_guidance: # dual cfg
			text_encodings = torch.cat([uncond_text_encodings, text_encodings, text_encodings])
		elif do_text_classifier_free_guidance: # only text cfg
			text_encodings = torch.cat([uncond_text_encodings, text_encodings])
		elif do_audio_classifier_free_guidance: # only audio cfg
			text_encodings = torch.cat([text_encodings, text_encodings])
		
		return text_encodings
	
	@torch.no_grad()
	def encode_audio(
			self,
			audios: Union[List[np.ndarray], List[torch.Tensor]],
			video_length: int = 12,
			do_text_classifier_free_guidance: bool = False,
			do_audio_classifier_free_guidance: bool = False,
			device: torch.device = torch.device("cuda:0"),
			dtype: torch.dtype = torch.float32
	):
		batch_size = len(audios)
		melspectrograms = self.audio_processor(audios).to(device=device, dtype=dtype) # (b c n t)
		
		# audio_encodings: (b, n, c)
		# audio_masks: (b, s, n)
		_, audio_encodings, audio_masks = self.audio_encoder(
			melspectrograms, normalize=False, return_dict=False
		)
		audio_encodings = repeat(audio_encodings, "b n c -> b f n c", f=video_length)
		
		if do_audio_classifier_free_guidance:
			null_melspectrograms = torch.zeros(1, 1, *self.melspectrogram_shape).to(device=device, dtype=dtype)
			_, null_audio_encodings, null_audio_masks = self.audio_encoder(
				null_melspectrograms, normalize=False, return_dict=False
			)
			null_audio_encodings = repeat(null_audio_encodings, "1 n c -> b f n c", b=batch_size, f=video_length)
		
		if do_text_classifier_free_guidance and do_audio_classifier_free_guidance: # dual cfg
			audio_encodings = torch.cat([null_audio_encodings, null_audio_encodings, audio_encodings])
			audio_masks = torch.cat([null_audio_masks, null_audio_masks, audio_masks])
		elif do_text_classifier_free_guidance: # only text cfg
			audio_encodings = torch.cat([audio_encodings, audio_encodings])
			audio_masks = torch.cat([audio_masks, audio_masks])
		elif do_audio_classifier_free_guidance: # only audio cfg
			audio_encodings = torch.cat([null_audio_encodings, audio_encodings])
			audio_masks = torch.cat([null_audio_masks, audio_masks])
		
		return audio_encodings, audio_masks
	
	@torch.no_grad()
	def encode_latents(self, image: torch.Tensor):
		dtype = self.vae.dtype
		image = image.to(device=self.device, dtype=dtype)
		image_latents = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
		return image_latents
	
	# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
	@torch.no_grad()
	def decode_latents(self, latents):
		dtype = next(self.vae.parameters()).dtype
		latents = latents.to(dtype=dtype)
		latents = 1 / self.vae.config.scaling_factor * latents
		image = self.vae.decode(latents).sample
		image = (image / 2 + 0.5).clamp(0, 1).cpu().float()  # ((b t) c h w)
		return image
	
	# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
	def prepare_extra_step_kwargs(self, generator, eta):
		# prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
		# eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
		# eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
		# and should be between [0, 1]
		
		accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
		extra_step_kwargs = {}
		if accepts_eta:
			extra_step_kwargs["eta"] = eta
		
		# check if the scheduler accepts generator
		accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
		if accepts_generator:
			extra_step_kwargs["generator"] = generator
		return extra_step_kwargs
	
	# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
	def prepare_video_latents(
			self,
			image_latents: torch.Tensor,
			num_channels_latents: int,
			video_length: int = 12,
			height: int = 256,
			width: int = 256,
			device: torch.device = torch.device("cuda"),
			dtype: torch.dtype = torch.float32,
			generator: Optional[torch.Generator] = None,
	):
		batch_size = len(image_latents)
		shape = (
			batch_size,
			num_channels_latents,
			video_length-1,
			height // self.vae_scale_factor,
			width // self.vae_scale_factor
		)
		
		image_latents = image_latents.unsqueeze(2) # (b c 1 h w)
		rand_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
		noise_latents = torch.cat([image_latents, rand_noise], dim=2)
		
		# scale the initial noise by the standard deviation required by the scheduler
		noise_latents = noise_latents * self.scheduler.init_noise_sigma
		
		return noise_latents
	
	@torch.no_grad()
	def __call__(
			self,
			images: List[PIL.Image.Image],
			audios: Union[List[np.ndarray], List[torch.Tensor]],
			texts: List[str],
			text_encodings: Optional[List[torch.Tensor]] = None,
			video_length: int = 12,
			height: int = 256,
			width: int = 256,
			num_inference_steps: int = 20,
			audio_guidance_scale: float = 4.0,
			text_guidance_scale: float = 1.0,
			generator: Optional[torch.Generator] = None,
			return_dict: bool = True
	):
		# 0. Default height and width to unet
		device = self.device
		dtype = self.dtype
		
		batch_size = len(images)
		height = height or self.unet.config.sample_size * self.vae_scale_factor
		width = width or self.unet.config.sample_size * self.vae_scale_factor
		
		do_text_classifier_free_guidance = (text_guidance_scale > 1.0)
		do_audio_classifier_free_guidance = (audio_guidance_scale > 1.0)
		
		# 1. Encoder text into ((k b) f n d)
		text_encodings = self.encode_text(
			texts=texts,
			text_encodings=text_encodings,
			device=device,
			dtype=dtype,
			do_text_classifier_free_guidance=do_text_classifier_free_guidance,
			do_audio_classifier_free_guidance=do_audio_classifier_free_guidance
		) # ((k b), n, d)
		text_encodings = repeat(text_encodings, "b n d -> b t n d", t=video_length).to(device=device, dtype=dtype)
		
		# 2. Encode audio
		# audio_encodings: ((k b), n, d)
		# audio_masks: ((k b), s, n)
		audio_encodings, audio_masks = self.encode_audio(
			audios, video_length, do_text_classifier_free_guidance, do_audio_classifier_free_guidance, device, dtype
		)
		
		# 3. Prepare image latent
		image = self.image_processor.preprocess(images)
		image_latents = self.encode_latents(image).to(device=device, dtype=dtype)  # (b c h w)
		
		# 4. Prepare unet noising video latents
		video_latents = self.prepare_video_latents(
			image_latents=image_latents,
			num_channels_latents=self.unet.config.in_channels,
			video_length=video_length,
			height=height,
			width=width,
			dtype=dtype,
			device=device,
			generator=generator,
		)  # (b c f h w)
		
		# 5. Prepare timesteps and extra step kwargs
		self.scheduler.set_timesteps(num_inference_steps, device=device)
		timesteps = self.scheduler.timesteps
		extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta=0.0)
		
		# 7. Denoising loop
		for i, t in enumerate(self.progress_bar(timesteps)):
			latent_model_input = [video_latents]
			if do_text_classifier_free_guidance:
				latent_model_input.append(video_latents)
			if do_audio_classifier_free_guidance:
				latent_model_input.append(video_latents)
			latent_model_input = torch.cat(latent_model_input)
			latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
			
			# predict the noise residual
			noise_pred = self.unet(
				latent_model_input,
				t,
				encoder_hidden_states=text_encodings,
				audio_encoder_hidden_states=audio_encodings,
				audio_attention_mask=audio_masks
			).sample
			
			# perform guidance
			if do_text_classifier_free_guidance and do_audio_classifier_free_guidance: # dual cfg
				noise_pred_uncond, noise_pred_text, noise_pred_text_audio = noise_pred.chunk(3)
				noise_pred = noise_pred_uncond + \
				             text_guidance_scale * (noise_pred_text - noise_pred_uncond) + \
				             audio_guidance_scale * (noise_pred_text_audio - noise_pred_text)
			elif do_text_classifier_free_guidance: # only text cfg
				noise_pred_audio, noise_pred_text_audio = noise_pred.chunk(2)
				noise_pred = noise_pred_audio + \
			                text_guidance_scale * (noise_pred_text_audio - noise_pred_audio)
			elif do_audio_classifier_free_guidance: # only audio cfg
				noise_pred_text, noise_pred_text_audio = noise_pred.chunk(2)
				noise_pred = noise_pred_text + \
				             audio_guidance_scale * (noise_pred_text_audio - noise_pred_text)
			
			# First frame latent will always server as unchanged condition
			video_latents[:, :, 1:, :, :] = self.scheduler.step(noise_pred[:, :, 1:, :, :], t, video_latents[:, :, 1:, :, :], **extra_step_kwargs).prev_sample
			video_latents = video_latents.contiguous()
		
		# 8. Post-processing
		video_latents = rearrange(video_latents, "b c f h w -> (b f) c h w")
		videos = self.decode_latents(video_latents).detach().cpu()
		videos = rearrange(videos, "(b f) c h w -> b f c h w", f=video_length) # value range [0, 1]
		
		if not return_dict:
			return videos
		
		return {"videos": videos}


@torch.no_grad()
def generate_videos(
		pipeline,
		image_path: str = '',
		audio_path: str = '',
		video_path: str = '',
		category: str = '',
		category_text_encoding: Optional[torch.Tensor] = None,
		image_size: Tuple[int, int] = (256, 256),
		video_fps: int = 6,
		video_num_frame: int = 12,
		num_clips_per_video: int = 3,
		audio_guidance_scale: float = 4.0,
		text_guidance_scale: float = 1.0,
		seed: int = 0,
		save_template: str = "",
		device: torch.device = torch.device("cuda"),
):
	# Prioritize loading from image_path and audio_path for image and audio, respectively
	# Otherwise, load from video_path
	# Can not specify all three
	assert not (image_path and audio_path and video_path), "Can not specify image_path, audio_path, video_path all three"
	clip_duration = video_num_frame/video_fps
	
	images = None
	audios = None
	
	if image_path:
		image = load_image(image_path, image_size)
		images = [image,] * num_clips_per_video
	
	if audio_path:
		audios = load_audio_clips_uniformly(audio_path, clip_duration, num_clips_per_video, load_audio_as_melspectrogram=False)
	
	if video_path:
		load_videos, load_audios = load_av_clips_uniformly(
			video_path, video_fps, video_num_frame, image_size, num_clips_per_video,
			load_audio_as_melspectrogram=False
		)
		
		if images is None:
			images = [video[0] for video in load_videos]
		if audios is None:
			audios = load_audios
	
	# convert images to PIL Images
	images = [
		PIL.Image.fromarray((255 * image).byte().permute(1, 2, 0).contiguous().numpy()) for image in images
	]
	
	generated_video_list = []
	generated_audio_list = []
	
	generator = torch.Generator(device=device)
	for k, (image, audio) in enumerate(zip(images, audios)):
		generator.manual_seed(seed)
		generated_video = pipeline(
			images=[image],
			audios=[audio],
			texts=[category],
			text_encodings=[category_text_encoding] if category_text_encoding is not None else None,
			video_length=video_num_frame,
			height=image_size[0],
			width=image_size[1],
			num_inference_steps=50,
			audio_guidance_scale=audio_guidance_scale,
			text_guidance_scale=text_guidance_scale,
			generator=generator,
			return_dict=False
		)[0]  # (f c h w) in range [0, 1]
		generated_video = (generated_video.permute(0, 2, 3, 1).contiguous() * 255).byte()
		
		if save_template:
			save_path = f"{save_template}_clip-{k:02d}.mp4"
			os.makedirs(os.path.dirname(save_path), exist_ok=True)
			torchvision.io.write_video(
				filename=save_path,
				video_array=generated_video,
				fps=video_fps,
				audio_array=audio,
				audio_fps=16000,
				audio_codec="aac"
			)
		else:
			generated_video_list.append(generated_video)
			generated_audio_list.append(audio)
	
	if save_template:
		return
		
	return generated_video_list, generated_audio_list


@torch.no_grad()
def generate_videos_for_dataset(
		exp_root: str,
		checkpoint: int,
		dataset: str = "AVSync15",
		image_size: Tuple[int, int] = (256, 256),
		video_fps: int = 6,
		video_num_frame: int = 12,
		num_clips_per_video: int = 3,
		audio_guidance_scale: float = 4.0,
		text_guidance_scale: float = 1.0,
		random_seed: int = 0,
		device: torch.device = torch.device("cuda"),
		dtype: torch.dtype = torch.float16
):
	
	checkpoint_path = f"{exp_root}/ckpts/checkpoint-{checkpoint}/modules"
	save_root = f"{exp_root}/evaluations/checkpoint-{checkpoint}/AG-{audio_guidance_scale}_TG-{text_guidance_scale}/seed-{random_seed}/videos"
	
	# 1. Prepare datasets and precomputed features
	video_root, filenames, categories, video_type = get_evaluation_data(dataset)
	
	null_text_encoding_path = "./pretrained/openai-clip-l_null_text_encoding.pt"
	if dataset == "TheGreatestHits":
		category_text_encoding = torch.load("./datasets/TheGreatestHits/class_clip_text_encodings_stable-diffusion-v1-5.pt", map_location="cpu")
		category_mapping = {"hitting with a stick": "hitting with a stick"}
		category_text_encoding_mapping = {"hitting with a stick": category_text_encoding}
	elif dataset == "Landscapes":
		category_mapping = json.load(open('./datasets/Landscapes/class_mapping.json', 'r'))
		category_text_encoding_mapping = torch.load('./datasets/Landscapes/class_clip_text_encodings_stable-diffusion-v1-5.pt', map_location="cpu")
	elif dataset == "AVSync15":
		category_mapping = json.load(open('./datasets/AVSync15/class_mapping.json', 'r'))
		category_text_encoding_mapping = torch.load('./datasets/AVSync15/class_clip_text_encodings_stable-diffusion-v1-5.pt', map_location="cpu")
	else:
		raise Exception()
	
	# 2. Prepare models
	pretrained_stable_diffusion_path = "./pretrained/stable-diffusion-v1-5"
	
	tokenizer = CLIPTokenizer.from_pretrained(pretrained_stable_diffusion_path, subfolder="tokenizer")
	scheduler = PNDMScheduler.from_pretrained(pretrained_stable_diffusion_path, subfolder="scheduler")
	text_encoder = CLIPTextModel.from_pretrained(pretrained_stable_diffusion_path, subfolder="text_encoder").to(device=device, dtype=dtype)
	vae = AutoencoderKL.from_pretrained(pretrained_stable_diffusion_path, subfolder="vae").to(device=device, dtype=dtype)
	audio_encoder = ImageBindSegmaskAudioEncoder(n_segment=video_num_frame).to(device=device, dtype=dtype)
	freeze_and_make_eval(audio_encoder)
	unet = AudioUNet3DConditionModel.from_pretrained(checkpoint_path, subfolder="unet").to(device=device, dtype=dtype)

	pipeline = AudioCondAnimationPipeline(
		text_encoder=text_encoder,
		tokenizer=tokenizer,
		unet=unet,
		scheduler=scheduler,
		vae=vae,
		audio_encoder=audio_encoder,
		null_text_encodings_path=null_text_encoding_path
	)
	pipeline.to(torch_device=device, dtype=dtype)
	pipeline.set_progress_bar_config(disable=True)
	
	# 3. Generating one by one

	for filename, category in tqdm(zip(filenames, categories), total=len(filenames)):
		video_path = os.path.join(video_root, filename)
		save_template = os.path.join(save_root, filename.replace(".mp4", ""))
		
		category_text_encoding = category_text_encoding_mapping[category_mapping[category]].view(1, 77, 768)
		
		generate_videos(
			pipeline,
			video_path=video_path,
			category_text_encoding=category_text_encoding,
			image_size=image_size,
			video_fps=video_fps,
			video_num_frame=video_num_frame,
			num_clips_per_video=num_clips_per_video,
			text_guidance_scale=text_guidance_scale,
			audio_guidance_scale=audio_guidance_scale,
			seed=random_seed,
			save_template=save_template,
			device=device
		)
	