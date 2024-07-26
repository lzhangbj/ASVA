import argparse
import logging
import os
from einops import rearrange, repeat
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from avsync.models.audio import AudioConv2DNet
from avsync.models.video import VideoR2Plus1DNet
from avsync.models.head import FCHead
from avsync.data import AudioVideoAlignedMultiPairDataset

from avgen.utils import freeze_and_make_eval

torch.backends.cuda.enable_flash_sdp(False)
logger = get_logger(__name__, log_level="INFO")


def setup(mixed_precision="no", seed=123):
	accelerator = Accelerator(mixed_precision=mixed_precision)
	
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)
	logger.info(accelerator.state, main_process_only=False)
	
	set_seed(seed)
	
	return accelerator, logger


def get_dataloaders(
		data_root="./datasets/VGGSS/videos",
		example_list_path="./datasets/VGGSS/test.txt",
		shift_time: float = 0.04,
		num_clips: int = 31
):
	# Get the testing dataset
	test_dataset = AudioVideoAlignedMultiPairDataset(
		data_root=data_root,
		example_list_path=example_list_path,
		mode="test",
		image_size=224,
		video_fps=6,
		video_num_frames=12,
		audio_sample_rate=16000,
		randflip=False,
		shift_time=shift_time,
		num_clips=num_clips,
		sampling_type="center-compact"
	)
	# DataLoaders creation:
	test_dataloader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=8,
		shuffle=False,
		pin_memory=False,
		drop_last=False
	)
	
	return test_dataloader


@torch.no_grad()
def main(
	checkpoint: str,
	data_root: str = "./datasets/VGGSS/videos",
	example_list_path: str = "./datasets/VGGSS/test.txt",
	shift_time: float = 0.04,
	num_clips: int = 31,
	tolerate_range: int = 5,
	mixed_precision: str = "no",
	seed: int = 123
):
	accelerator, logger = setup(mixed_precision=mixed_precision, seed=seed)
	
	logger.info("... creating models ...")
	audio_encoder = AudioConv2DNet.from_pretrained(os.path.join(checkpoint, "audio_encoder"), use_safetensors=False)
	video_encoder = VideoR2Plus1DNet.from_pretrained(os.path.join(checkpoint, "video_encoder"), use_safetensors=False)
	head = FCHead.from_pretrained(os.path.join(checkpoint, "head"), use_safetensors=False)
	freeze_and_make_eval(audio_encoder)
	freeze_and_make_eval(video_encoder)
	freeze_and_make_eval(head)
	
	logger.info("... creating dataloaders ...")
	test_dataloader = get_dataloaders(
		data_root=data_root,
		example_list_path=example_list_path,
		shift_time=shift_time,
		num_clips=num_clips
	)
	
	# Prepare everything with our `accelerator`.
	audio_encoder, video_encoder, head, test_dataloader = accelerator.prepare(
		audio_encoder, video_encoder, head, test_dataloader
	)

	logger.info("***** Running testing *****")
	logger.info(f"  Num examples = {len(test_dataloader.dataset)}")
	logger.info(f"  Instantaneous batch size per device = 8")
	
	logger.info("############ Start Evaluation ############")
	all_indices = []
	all_av_accs = []
	all_va_accs = []
	for batch in tqdm(test_dataloader, desc="... Evaluating ..."):

		example_indices = batch["index"]
		audios = batch["audio"]  # (b k c n t)
		videos = batch["video"]  # (b k c f h w)
		device = audios.device
		batchsize = audios.shape[0]
		num_pairs = audios.shape[1]
		center_index = num_pairs//2
		labels = (center_index * torch.ones(batchsize)).long().to(device=device)
		
		# Forward
		audios = rearrange(audios, "b k c n t -> (b k) c n t")
		videos = rearrange(videos, "b k c f h w -> (b k) c f h w")
		
		audio_embeddings = audio_encoder(audios)  # (bk c)
		center_audio_embeddings = rearrange(
			audio_embeddings, "(b k) c -> b k c", b=batchsize, k=num_pairs
		)[:, center_index]
		video_embeddings = video_encoder(videos)  # (bk c)
		center_video_embeddings = rearrange(
			video_embeddings, "(b k) c -> b k c", b=batchsize, k=num_pairs
		)[:, center_index]
		
		av_predictions = head(
			repeat(center_audio_embeddings, "b c -> (b k) c", k=num_pairs),
			video_embeddings
		)[:, 0].contiguous().view(batchsize, num_pairs).argmax(dim=1)  # (b,)
		per_sample_av_acc = (torch.abs(av_predictions - labels) <= tolerate_range).float()
		
		va_predictions = head(
			audio_embeddings,
			repeat(center_video_embeddings, "b c-> (b k) c", k=num_pairs)
		)[:, 0].contiguous().view(batchsize, num_pairs).argmax(dim=1) # (b, )
		per_sample_va_acc = (torch.abs(va_predictions - labels) <= tolerate_range).float()
		
		example_indices = accelerator.gather(example_indices).detach().cpu().numpy()
		av_accs = accelerator.gather(per_sample_av_acc).detach().cpu().numpy()
		va_accs = accelerator.gather(per_sample_va_acc).detach().cpu().numpy()
		
		all_indices.append(example_indices)
		all_av_accs.append(av_accs)
		all_va_accs.append(va_accs)

	all_indices = np.concatenate(all_indices)
	all_av_accs = np.concatenate(all_av_accs)
	all_va_accs = np.concatenate(all_va_accs)
	
	# During evaluation, there can be some videos being sampled multiple times,
	# due to either loading errors or multi-gpu processing.
	# We therefore only evaluate on uniquely computed results
	unique_indices = np.unique(all_indices, return_index=True)[1]
	av_acc = all_av_accs[unique_indices].mean().item()
	va_acc = all_va_accs[unique_indices].mean().item()
	
	logger.info(
		f"#########################################################################\n"
		f"Evaluation result: \n"
		f"checkpoint = {checkpoint}\n"
		f"#########################################################################\n"
		f"valid examples = {len(unique_indices)}\n"
		f"av_acc = {av_acc:.4f}\n"
		f"va_acc = {va_acc:.4f}"
	)
	
	accelerator.wait_for_everyone()
	accelerator.end_training()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", type=str, required=True)
	parser.add_argument("--data_root", type=str, default="./datasets/VGGSS/videos")
	parser.add_argument("--example_list_path", type=str, default="./datasets/VGGSS/test.txt")
	parser.add_argument("--shift_time", type=float, default=0.04)
	parser.add_argument("--num_clips", type=int, default=31)
	parser.add_argument("--tolerate_range", type=int, default=5)
	parser.add_argument("--mixed_precision", type=str, default="no")
	parser.add_argument("--seed", type=int, default=123)
	args = parser.parse_args()
	
	main(
		args.checkpoint,
		args.data_root,
		args.example_list_path,
		args.shift_time,
		args.num_clips,
		args.tolerate_range,
		args.mixed_precision,
		args.seed
	)

