import argparse
import torch

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler
from diffusers.utils import logging

from avgen.models.unets import AudioUNet3DConditionModel
from avgen.models.audio_encoders import ImageBindSegmaskAudioEncoder
from avgen.utils import freeze_and_make_eval

from avgen.pipelines.pipeline_audio_cond_animation import AudioCondAnimationPipeline, generate_videos

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, default="AVSync15", help="dataset and model to evaluate on")
	parser.add_argument("--audio", type=str, default="", help="path of input audio")
	parser.add_argument("--image", type=str, default="", help="path of input image")
	parser.add_argument("--video", type=str, default="", help="path of input video")
	parser.add_argument("--category", type=str, default="", help="audio category")
	parser.add_argument("--audio_guidance", type=float, default=4.0, help="audio guidance scale >= 1.0")
	parser.add_argument("--save_path", type=str, default="", help="path to save generated video, will be saved as save_path_clip-00.mp4")
	args = parser.parse_args()
	
	assert not (args.audio and args.image and args.video), \
		"Video is provided as alternative to provide image or audio as input. So please specify one of (audio, image), (video, image), (audio, video), or only video."
	
	device = torch.device("cuda")
	dtype = torch.float32
	
	# 1. Load checkpoint and config
	if args.dataset == "AVSync15":
		checkpoint_path = f"checkpoints/audio-cond_animation/avsync15_audio-cond_cfg/ckpts/checkpoint-37000/modules"
		categories = [
			"baby babbling crying", "dog barking", "hammering", "striking bowling", "cap gun shooting",
			"chicken crowing", "frog croaking", "lions roaring", "machine gun shooting", "playing cello",
			"playing trombone", "playing trumpet", "playing violin fiddle", "sharpen knife", "toilet flushing"
		]
		category_text_encoding_mapping = torch.load('datasets/AVSync15/class_clip_text_encodings_stable-diffusion-v1-5.pt', map_location="cpu")
		image_size = (256, 256)
		
	elif args.dataset == "Landscapes":
		checkpoint_path = f"checkpoints/audio-cond_animation/landscapes_audio-cond_cfg/ckpts/checkpoint-24000/modules"
		categories = [
			"explosion", "fire crackling", "raining", "splashing water", "squishing water",
			"thunder", "underwater bubbling", "waterfall burbling", "wind noise"
		]
		category_text_encoding_mapping = torch.load('datasets/Landscapes/class_clip_text_encodings_stable-diffusion-v1-5.pt', map_location="cpu")
		image_size = (256, 256)
		
	elif args.dataset == "TheGreatestHits":
		checkpoint_path = f"checkpoints/audio-cond_animation/thegreatesthits_audio-cond_cfg/ckpts/checkpoint-16000/modules"
		category_text_encoding = torch.load("datasets/TheGreatestHits/class_clip_text_encodings_stable-diffusion-v1-5.pt", map_location="cpu")
		categories = ["hitting with a stick"]
		category_text_encoding_mapping = {"hitting with a stick": category_text_encoding}
		image_size = (128, 256)
		
	else:
		raise Exception()

	assert args.category in categories, f"{args.dataset} does not have category {args.category}, please choose from {','.join(categories)}"
	category_text_encoding = category_text_encoding_mapping[args.category].view(1, 77, 768)
	
	assert args.save_path.endswith(".mp4"), f"save path must be .mp4 file, but is {args.save_path}"
	
	# 2. Prepare model
	pretrained_stable_diffusion_path = "./pretrained/stable-diffusion-v1-5"
	
	tokenizer = CLIPTokenizer.from_pretrained(pretrained_stable_diffusion_path, subfolder="tokenizer")
	scheduler = PNDMScheduler.from_pretrained(pretrained_stable_diffusion_path, subfolder="scheduler")
	text_encoder = CLIPTextModel.from_pretrained(pretrained_stable_diffusion_path, subfolder="text_encoder").to(device=device, dtype=dtype)
	vae = AutoencoderKL.from_pretrained(pretrained_stable_diffusion_path, subfolder="vae").to(device=device,dtype=dtype)
	audio_encoder = ImageBindSegmaskAudioEncoder(n_segment=12).to(device=device, dtype=dtype)
	freeze_and_make_eval(audio_encoder)
	unet = AudioUNet3DConditionModel.from_pretrained(checkpoint_path, subfolder="unet").to(device=device, dtype=dtype)
	
	pipeline = AudioCondAnimationPipeline(
		text_encoder=text_encoder,
		tokenizer=tokenizer,
		unet=unet,
		scheduler=scheduler,
		vae=vae,
		audio_encoder=audio_encoder,
		null_text_encodings_path="./pretrained/openai-clip-l_null_text_encoding.pt"
	)
	pipeline.to(torch_device=device, dtype=dtype)
	pipeline.set_progress_bar_config(disable=True)
	
	# 3. Generating one by one
	generate_videos(
		pipeline,
		audio_path=args.audio,
		image_path=args.image,
		video_path=args.video,
		category_text_encoding=category_text_encoding,
		image_size=image_size,
		video_fps=6,
		video_num_frame=12,
		num_clips_per_video=1,
		text_guidance_scale=1.0,
		audio_guidance_scale=args.audio_guidance,
		seed=123,
		save_template=args.save_path[:-4],
		device=device
	)
