import argparse
import logging
import math
import os
import shutil
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import torch
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers.optimization import get_scheduler
from avgen.utils import AverageMeter
from avsync.models.audio import AudioConv2DNet
from avsync.models.video import VideoR2Plus1DNet
from avsync.models.head import FCHead
from avsync.models.sync_contrastive_trainer import AVSyncContrastiveTrainer
from avsync.data import AudioVideoAlignedMultiPairDataset

torch.backends.cuda.enable_flash_sdp(False)
logger = get_logger(__name__, log_level="INFO")


def setup_logger(cfg):
	# Make one log on every process with the configuration for debugging.
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)
	
	file_handler = logging.FileHandler(cfg.exp.log_file)
	console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	                                   datefmt="%m/%d/%Y %H:%M:%S")
	file_handler.setFormatter(console_format)
	file_handler.setLevel(logging.INFO)
	
	logger.logger.addHandler(file_handler)
	
	return logger


def setup(cfg):
	accelerator = Accelerator(
		gradient_accumulation_steps=cfg.optim.gradient_accumulation_steps,
		mixed_precision=cfg.optim.mixed_precision,
		log_with=cfg.exp.log_with,
	)
	
	# Handle the output folder creation
	if accelerator.is_main_process:
		os.makedirs(cfg.exp.output_dir, exist_ok=True)
		OmegaConf.save(cfg, os.path.join(cfg.exp.output_dir, 'config.yaml'))
	accelerator.wait_for_everyone()
	
	logger = setup_logger(cfg)
	logger.info(accelerator.state, main_process_only=False)
	
	# If passed along, set the training seed now.
	if cfg.exp.seed is not None:
		set_seed(cfg.exp.seed)
	
	return accelerator, logger


def get_optimizer(cfg, accelerator, model):
	learning_rate = cfg.optim.learning_rate
	if cfg.optim.scale_lr:
		learning_rate = (
				learning_rate *
				cfg.optim.gradient_accumulation_steps *
				cfg.train.batch_size *
				accelerator.num_processes
		)
	
	# Initialize the optimizer
	if cfg.optim.use_8bit_adam:
		try:
			import bitsandbytes as bnb
		except ImportError:
			raise ImportError(
				"Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
			)
		
		optimizer_cls = bnb.optim.AdamW8bit
	else:
		optimizer_cls = torch.optim.AdamW
	
	optimizer = optimizer_cls(
		[param for param in model.parameters() if param.requires_grad],
		lr=learning_rate,
		betas=(cfg.optim.adam_beta1, cfg.optim.adam_beta2),
		weight_decay=cfg.optim.adam_weight_decay,
		eps=cfg.optim.adam_epsilon,
	)
	
	# Scheduler
	lr_scheduler = get_scheduler(
		cfg.optim.lr_scheduler,
		optimizer=optimizer,
		num_warmup_steps=cfg.optim.lr_warmup_steps,
		num_training_steps=cfg.optim.max_train_steps,
	)
	
	return optimizer, lr_scheduler


def get_dataloaders(cfg):
	# Get the training dataset
	train_dataset = AudioVideoAlignedMultiPairDataset(**cfg.train.dataset)
	# DataLoaders creation:
	train_dataloader = torch.utils.data.DataLoader(train_dataset,
	                                               batch_size=cfg.train.batch_size,
	                                               shuffle=True,
	                                               pin_memory=True,
	                                               drop_last=True)
	
	# Get the testing dataset
	test_dataset = AudioVideoAlignedMultiPairDataset(**cfg.test.dataset)
	# DataLoaders creation:
	test_dataloader = torch.utils.data.DataLoader(test_dataset,
	                                               batch_size=cfg.test.batch_size,
	                                               shuffle=False,
	                                               pin_memory=True,
	                                               drop_last=True)
	
	return train_dataloader, test_dataloader


def get_net(cfg):
	net_name = cfg.name
	cfg.pop("name")
	pretrained_model_path = cfg.get("pretrained_model_path", None)
	if pretrained_model_path is not None:
		cfg.pop("pretrained_model_path")
		pretrained_model = eval(net_name).from_pretrained(pretrained_model_path)
		pretrained_state_dict = pretrained_model.state_dict()
		
		net = eval(net_name)(**cfg)
		load_state_dict = {}
		for key, val in net.state_dict().items():
			if key in pretrained_state_dict and \
				val.shape == pretrained_state_dict[key].shape:
				load_state_dict[key] = pretrained_state_dict[key]
			else:
				load_state_dict[key] = val
		net.load_state_dict(load_state_dict)
	else:
		net = eval(net_name)(**cfg)
	
	return net
	

def main(args):
	cfg = OmegaConf.load(args.config_file)
	
	accelerator, logger = setup(cfg)
	
	logger.info("... creating models ...")
	audio_encoder = get_net(cfg.model.audio_encoder)
	video_encoder = get_net(cfg.model.video_encoder)
	head = get_net(cfg.model.head)

	trainer = AVSyncContrastiveTrainer(
		audio_encoder=audio_encoder,
		video_encoder=video_encoder,
		head=head,
		tau=cfg.model.tau
	)
	for param in trainer.parameters():
		param.requires_grad_(True)
	
	logger.info("... creating optimizers ...")
	optimizer, lr_scheduler = get_optimizer(cfg, accelerator, trainer)
	
	logger.info("... creating dataloaders ...")
	train_dataloader, test_dataloader = get_dataloaders(cfg)
	num_train_epochs = math.ceil(cfg.optim.max_train_steps / len(train_dataloader))
	
	# Prepare everything with our `accelerator`.
	trainer, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
		trainer, optimizer, train_dataloader, test_dataloader, lr_scheduler
	)
	# We need to initialize the trackers we use, and also store our configuration.
	# The trackers initializes automatically on the main process.
	if accelerator.is_main_process:
		config_file_name = args.config_file.split("/")[-1].replace(".yaml", "")
		accelerator.init_trackers(
			project_name="avsync-classifier",
			init_kwargs={
				"wandb": {
				  "name": config_file_name,
				}
			}
		)
	
	# We need to recalculate our total training steps as the size of the training dataloader may have changed.
	num_update_steps_per_epoch = math.floor(len(train_dataloader) / cfg.optim.gradient_accumulation_steps)
	dropped_steps = len(train_dataloader) % cfg.optim.gradient_accumulation_steps
	# Afterwards we recalculate our number of training epochs
	cfg.optim.max_train_steps = num_train_epochs * num_update_steps_per_epoch
	total_batch_size = cfg.train.batch_size * accelerator.num_processes * cfg.optim.gradient_accumulation_steps
	
	logger.info("***** Running training *****")
	logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
	logger.info(f"  Num Epochs = {num_train_epochs}")
	logger.info(f"  Instantaneous batch size per device = {cfg.train.batch_size}")
	logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
	logger.info(f"  Gradient Accumulation steps = {cfg.optim.gradient_accumulation_steps}")
	logger.info(f"  Total optimization steps = {cfg.optim.max_train_steps}")
	global_step = 0
	first_epoch = 0
	
	checkpoint_dir = os.path.join(cfg.exp.output_dir, "ckpts")
	if accelerator.is_main_process:
		os.makedirs(checkpoint_dir, exist_ok=True)
	
	# Potentially load in the weights and states from a previous save
	if cfg.optim.resume_from_checkpoint:
		path = None
		if cfg.optim.resume_from_checkpoint != "latest":
			path = os.path.basename(cfg.optim.resume_from_checkpoint)
		elif os.path.exists(checkpoint_dir):
			# Get the most recent checkpoint
			dirs = os.listdir(checkpoint_dir)
			dirs = [d for d in dirs if d.startswith("checkpoint")]
			dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
			if len(dirs) == 0:
				path = None
			else:
				path = dirs[-1]
		
		if path is not None:
			accelerator.print(f"Resuming from checkpoint {path}")
			accelerator.load_state(os.path.join(checkpoint_dir, path))
			global_step = int(path.split("-")[1])
			
			first_epoch = global_step // num_update_steps_per_epoch
			resume_step = global_step % num_update_steps_per_epoch * cfg.optim.gradient_accumulation_steps
		else:
			resume_step = 0
	
	logger.info("############ Start Training ############")
	for epoch in range(first_epoch, num_train_epochs):
		train_loss_meter = AverageMeter()
		train_av_loss_meter = AverageMeter()
		train_va_loss_meter = AverageMeter()
		train_av_acc_meter = AverageMeter()
		train_va_acc_meter = AverageMeter()
		
		train_loss = 0.0
		train_av_loss = 0.0
		train_va_loss = 0.0
		train_av_acc = 0.0
		train_va_acc = 0.0
		
		for step, batch in enumerate(train_dataloader):
			trainer.train()
			# Skip steps until we reach the resumed step
			if cfg.optim.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
				continue
			
			# drop some steps to prevent error in gradient accumulation
			if (len(train_dataloader) - step) <= dropped_steps: continue
			
			with accelerator.accumulate(trainer):
				audios = batch["audio"] # (b k c n t)
				videos = batch["video"] # (b k c f h w)
				
				av_loss, va_loss, av_acc, va_acc = trainer(audios, videos)
				loss = (av_loss + va_loss) / 2.
			
				# Gather the losses across
				avg_av_loss = accelerator.gather(av_loss).mean()
				train_av_loss += avg_av_loss.item() / cfg.optim.gradient_accumulation_steps
				avg_va_loss = accelerator.gather(va_loss).mean()
				train_va_loss += avg_va_loss.item() / cfg.optim.gradient_accumulation_steps
				train_loss += ((avg_av_loss + avg_va_loss) / 2.).item() / cfg.optim.gradient_accumulation_steps
				
				avg_av_acc = accelerator.gather(av_acc).mean()
				train_av_acc += avg_av_acc.item() / cfg.optim.gradient_accumulation_steps
				avg_va_acc = accelerator.gather(va_acc).mean()
				train_va_acc += avg_va_acc.item() / cfg.optim.gradient_accumulation_steps

				# Backpropagate
				accelerator.backward(loss)
				if accelerator.sync_gradients:
					accelerator.clip_grad_norm_(trainer.parameters(), cfg.optim.max_grad_norm)
				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()
			
			# Checks if the accelerator has performed an optimization step behind the scenes
			if accelerator.sync_gradients:
				# progress_bar.update(1)
				global_step += 1
				accelerator.log({
					"train_loss": train_loss,
					"train_av_loss": train_av_loss,
					"train_va_loss": train_va_loss,
					"train_av_acc": train_av_acc,
					"train_va_acc": train_va_acc,
					"lr": lr_scheduler.get_last_lr()[0]
				}, step=global_step)
				train_loss_meter.update(train_loss, cfg.train.batch_size)
				train_av_loss_meter.update(train_av_loss, cfg.train.batch_size)
				train_va_loss_meter.update(train_va_loss, cfg.train.batch_size)
				train_av_acc_meter.update(train_av_acc, cfg.train.batch_size)
				train_va_acc_meter.update(train_va_acc, cfg.train.batch_size)
				
				train_loss = 0.0
				train_av_loss = 0.0
				train_va_loss = 0.0
				train_av_acc = 0.0
				train_va_acc = 0.0
				
				if global_step % cfg.train.log_steps == 0:
					logger.info(
						f"Epoch={epoch} Step={global_step}/{cfg.optim.max_train_steps} "
						f"lr={lr_scheduler.get_last_lr()[0]:.7f} "
						f"train_loss/av_loss/va_loss={train_loss_meter.avg:.4f}/{train_av_loss_meter.avg:.4f}/{train_va_loss_meter.avg:.4f} "
						f"av_acc/va_acc={train_av_acc_meter.avg:.4f}/{train_va_acc_meter.avg:.4f} "
					)
					train_loss_meter.reset()
					train_av_loss_meter.reset()
					train_va_loss_meter.reset()
					train_av_acc_meter.reset()
					train_va_acc_meter.reset()
					
				if global_step % cfg.test.test_steps == 0:
					trainer.eval()
					
					if accelerator.is_main_process:
						test_loss_meter = AverageMeter()
						test_av_loss_meter = AverageMeter()
						test_va_loss_meter = AverageMeter()
						test_av_acc_meter = AverageMeter()
						test_va_acc_meter = AverageMeter()
					
					with torch.no_grad():
						for batch in tqdm(test_dataloader):
							audios = batch["audio"] # (b k c n t)
							videos = batch["video"] # (b k c f h w)
							batchsize = len(audios)
							
							av_loss, va_loss, av_acc, va_acc = trainer(audios, videos)
							
							# Gather the losses across all processes for logging (if we use distributed training).
							test_av_loss = accelerator.gather(av_loss).mean().item()
							test_av_acc = accelerator.gather(av_acc).mean().item()
							
							test_va_loss = accelerator.gather(va_loss).mean().item()
							test_va_acc = accelerator.gather(va_acc).mean().item()
							
							if accelerator.is_main_process:
								test_loss_meter.update((test_av_loss + test_va_loss)/2., batchsize)
								test_av_loss_meter.update(test_av_loss, batchsize)
								test_va_loss_meter.update(test_va_loss, batchsize)
								test_av_acc_meter.update(test_av_acc, batchsize)
								test_va_acc_meter.update(test_va_acc, batchsize)
								
						if accelerator.is_main_process:
							logger.info(
								f"Test Epoch={epoch} Step={global_step}/{cfg.optim.max_train_steps} "
								f"test_loss/av_loss/va_loss={test_loss_meter.avg:.4f}/{test_av_loss_meter.avg:.4f}/{test_va_loss_meter.avg:.4f} "
								f"av_acc/va_acc={test_av_acc_meter.avg:.4f}/{test_va_acc_meter.avg:.4f} "
							)
							accelerator.log({
								"test_loss": test_loss_meter.avg,
								"test_av_loss": test_av_loss_meter.avg,
								"test_va_loss": test_va_loss_meter.avg,
								"test_av_acc": test_av_acc_meter.avg,
								"test_va_acc": test_va_acc_meter.avg,
							}, step=global_step)
				
				if accelerator.is_main_process:
					if global_step % cfg.optim.checkpointing_milestones == 0:
						# save accelerator state
						save_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
						accelerator.save_state(save_path)
						logger.info(f"Saved state to {save_path}")
						# save individual models
						module_save_path = os.path.join(save_path, "modules")
						os.makedirs(module_save_path, exist_ok=True)
						accelerator.unwrap_model(trainer).save_pretrained(module_save_path)
					
					if global_step % cfg.optim.checkpointing_steps == 0:
						save_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
						accelerator.save_state(save_path)
						logger.info(f"Saved state to {save_path}")
						# save individual models
						module_save_path = os.path.join(save_path, "modules")
						os.makedirs(module_save_path, exist_ok=True)
						accelerator.unwrap_model(trainer).save_pretrained(module_save_path)
						
						# remove previous one
						if (global_step - cfg.optim.checkpointing_steps) % cfg.optim.checkpointing_milestones != 0:
							prev_save_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step - cfg.optim.checkpointing_steps}")
							if os.path.exists(prev_save_path):
								shutil.rmtree(prev_save_path)
				
				accelerator.wait_for_everyone()
			
			if global_step >= cfg.optim.max_train_steps:
				break
	
	if accelerator.is_main_process:
		# save accelerator state
		save_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
		accelerator.save_state(save_path)
		logger.info(f"Saved state to {save_path}")
		# save individual models
		module_save_path = os.path.join(save_path, "modules")
		os.makedirs(module_save_path, exist_ok=True)
		accelerator.unwrap_model(trainer).save_pretrained(module_save_path)
	
	# Create the pipeline using the trained modules and save it.
	accelerator.wait_for_everyone()
	accelerator.end_training()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_file", type=str, default="./configs/mini_train.yaml")
	args = parser.parse_args()
	
	main(args)

