import argparse
import torch

from avgen.pipelines.pipeline_audio_cond_animation import generate_videos_for_dataset


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_root", type=str, default="")
	parser.add_argument("--checkpoint", type=int, default=37000)
	parser.add_argument("--dataset", type=str, default="AVSync15")
	parser.add_argument("--image_h", type=int, default=256)
	parser.add_argument("--image_w", type=int, default=256)
	parser.add_argument("--video_fps", type=int, default=6)
	parser.add_argument("--video_num_frame", type=int, default=12)
	parser.add_argument("--num_clips_per_video", type=int, default=3)
	parser.add_argument("--audio_guidance_scale", type=float, default=4.0)
	parser.add_argument("--text_guidance_scale", type=float, default=1.0)
	parser.add_argument("--random_seed", type=int, default=0)
	args = parser.parse_args()
	
	
	print(
		f"########################################\n"
		f"Evaluating Audio-Cond Animation model on\n"
		f"dataset: {args.dataset}\n"
		f"exp: {args.exp_root}\n"
		f"checkpoint: {args.checkpoint}\n"
		f"########################################"
	)
	generate_videos_for_dataset(
		exp_root=args.exp_root,
		checkpoint=args.checkpoint,
		dataset=args.dataset,
		image_size=(args.image_h, args.image_w),
		video_fps=args.video_fps,
		video_num_frame=args.video_num_frame,
		num_clips_per_video=args.num_clips_per_video,
		audio_guidance_scale=args.audio_guidance_scale,
		text_guidance_scale=args.text_guidance_scale,
		random_seed=args.random_seed,
		device=torch.device("cuda"),
		dtype=torch.float32
	)

