import argparse
import logging
import math
import os
from omegaconf import OmegaConf
import shutil

import torch
import torch.utils.checkpoint

import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import diffusers
from diffusers.schedulers import DDPMScheduler
from diffusers.models import AutoencoderKL
from avgen.models.unets import AudioUNet3DConditionModel
from avgen.models.audio_encoders import ImageBindSegmaskAudioEncoder
from diffusers.optimization import get_scheduler

from avgen.data import BaseAudioVideoDataset
from avgen.models.trainers import AudioCondAnimationTrainer
from avgen.utils import freeze_model, AverageMeter

from diffusers.utils import check_min_version
check_min_version("0.10.0.dev0")

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
	# Set up accelerator
	accelerator = Accelerator(
		gradient_accumulation_steps=cfg.optim.gradient_accumulation_steps,
		mixed_precision=cfg.optim.mixed_precision,
		log_with=cfg.exp.log_with
	)
	
	# Save config file
	if accelerator.is_main_process:
		os.makedirs(cfg.exp.output_dir, exist_ok=True)
		OmegaConf.save(cfg, os.path.join(cfg.exp.output_dir, 'config.yaml'))
	
	# accelerator.wait_for_everyone()
	
	logger = setup_logger(cfg)
	logger.info(accelerator.state, main_process_only=False)
	if accelerator.is_local_main_process:
		transformers.utils.logging.set_verbosity_warning()
		diffusers.utils.logging.set_verbosity_info()
	else:
		transformers.utils.logging.set_verbosity_error()
		diffusers.utils.logging.set_verbosity_error()
	
	if cfg.exp.seed is not None:
		set_seed(cfg.exp.seed)
	
	return accelerator, logger


def build_model(config):
	model_name = config.name
	model_class = eval(model_name)
	
	config = config.copy()
	config.pop("name")
	# Create model in diffusers style
	if "pretrained_model_name_or_path" in config:
		pretrained_model_name_or_path = config.pretrained_model_name_or_path
		config.pop("pretrained_model_name_or_path")
		subfolder = config.get("subfolder", None)
		if "subfolder" in config: config.pop("subfolder")
		pretrained_model = model_class.from_pretrained(
			pretrained_model_name_or_path=pretrained_model_name_or_path,
			subfolder=subfolder
		)
		return pretrained_model
	
	return model_class(**config)


def build_unet(config):
	unet_config = config.model.unet
	unet = AudioUNet3DConditionModel.from_pretrained_2d(
		unet_config,
		unet_config.pretrained_model_name_or_path,
		subfolder="unet"
	)
	# Freeze pretrained image weights
	# Only train weights covered by trainable_modules
	if not unet_config.train_image_modules:
		unet.requires_grad_(False)
		for name, module in unet.named_modules():
			if any(ele in name for ele in tuple(unet_config.trainable_modules)):
				for params in module.parameters():
					params.requires_grad = True
	else:
		unet.requires_grad_(True)
	
	if config.optim.get("enable_gradient_checkpoint"):
		unet.enable_gradient_checkpointing()
	
	return unet

	
def create_models(cfg, weight_dtype):
	unet = build_unet(cfg)
	vae = AutoencoderKL.from_pretrained(cfg.model.vae.pretrained_model_name_or_path, subfolder="vae").to(dtype=weight_dtype)
	freeze_model(vae)
	audio_encoder = build_model(cfg.model.audio_encoder).to(dtype=weight_dtype)
	freeze_model(audio_encoder)
	scheduler = build_model(cfg.model.scheduler)
	
	trainer = AudioCondAnimationTrainer(
		vae=vae,
		scheduler=scheduler,
		unet=unet,
		audio_encoder=audio_encoder,
		audio_cond_drop_prob=cfg.model.audio_cond_drop_prob,
		text_cond_drop_prob=cfg.model.text_cond_drop_prob,
		loss_on_first_frame=cfg.model.loss_on_first_frame
	)
	
	return trainer


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
		# num_warmup_steps=cfg.optim.lr_warmup_steps * cfg.optim.gradient_accumulation_steps,
		num_training_steps=cfg.optim.max_train_steps * cfg.optim.gradient_accumulation_steps,
	)
	
	return optimizer, lr_scheduler


def get_dataloaders(cfg):
	# Get the training dataset
	train_dataset = BaseAudioVideoDataset(**cfg.train.dataset)
	# DataLoaders creation:
	train_dataloader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=cfg.train.batch_size,
		shuffle=True,
		pin_memory=True,
		drop_last=True
	)
	return train_dataloader


def main(args):
	cfg = OmegaConf.load(args.config_file)
	
	accelerator, logger = setup(cfg)

	weight_dtype = torch.float32
	if accelerator.mixed_precision == "fp16":
		weight_dtype = torch.float16
	elif accelerator.mixed_precision == "bf16":
		weight_dtype = torch.bfloat16
	
	logger.info("... creating models ...")
	trainer = create_models(cfg, weight_dtype)
	
	logger.info("... creating optimizers ...")
	optimizer, lr_scheduler = get_optimizer(cfg, accelerator, trainer)
	
	logger.info("... creating dataloaders ...")
	train_dataloader = get_dataloaders(cfg)
	
	num_train_epochs = math.ceil(cfg.optim.max_train_steps / len(train_dataloader))
	
	trainer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
		trainer, optimizer, train_dataloader, lr_scheduler
	)
	
	# We need to recalculate our total training steps as the size of the training dataloader may have changed.
	num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.optim.gradient_accumulation_steps)
	# Afterwards we recalculate number of training epochs
	cfg.optim.max_train_steps = num_train_epochs * num_update_steps_per_epoch
	
	# We need to initialize the trackers we use, and also store our configuration.
	# The trackers initializes automatically on the main process.
	if accelerator.is_main_process:
		config_file_name = args.config_file.split("/")[-1].replace(".yaml", "")
		accelerator.init_trackers(
			project_name="audio-cond_animation",
			init_kwargs={
				"wandb": {
					"name": config_file_name,
				}
			}
		)
	
	# Train!
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
			resume_step = global_step % num_update_steps_per_epoch
		else:
			resume_step = 0
	
	logger.info("############ Start Training ############")
	
	# When resume training, the unprocesed step in the last epoch may be less than gradient_accumulation_steps
	# If this happens, we skip those steps,
	#   to ensure the total number of steps in one epoch is divisible by gradient_accumulation_steps
	dropped_steps = len(train_dataloader) % cfg.optim.gradient_accumulation_steps

	for epoch in range(first_epoch, num_train_epochs):
		train_loss_meter = AverageMeter()
		train_loss = 0.0
		for step, batch in enumerate(train_dataloader):
			trainer.train()
			# Skip steps until we reach the resumed step
			if cfg.optim.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
				continue
				
			# drop some steps to prevent error in gradient accumulation
			if (len(train_dataloader) - step) <= dropped_steps: continue
			
			with accelerator.accumulate(trainer):
				videos = batch["video"].to(dtype=weight_dtype) # (b c f h w)
				audios = batch["audio"].to(dtype=weight_dtype) # (b c n t)
				class_text_encodings = batch.get('class_text_encoding', None)
				loss = trainer(videos=videos, audios=audios, text_encodings=class_text_encodings)
				
				# Gather the losses across all processes for logging (if we use distributed training).
				avg_loss = accelerator.gather(loss.repeat(cfg.train.batch_size)).mean()
				train_loss += avg_loss.item() / cfg.optim.gradient_accumulation_steps
				
				# Backpropagate
				accelerator.backward(loss)
				if accelerator.sync_gradients:
					accelerator.clip_grad_norm_(trainer.parameters(), cfg.optim.max_grad_norm)
				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()
			
			# Checks if the accelerator has performed an optimization step behind the scenes
			if accelerator.sync_gradients:
				global_step += 1
				accelerator.log({
					"train_loss": train_loss,
					"lr": lr_scheduler.get_last_lr()[0]
				}, step=global_step)
				train_loss_meter.update(train_loss, cfg.train.batch_size)
				train_loss = 0.0
				
				if global_step % cfg.train.log_steps == 0:
					logger.info(
						f"Epoch={epoch} Step={global_step}/{cfg.optim.max_train_steps} train_loss={train_loss_meter.avg:.4f} lr={lr_scheduler.get_last_lr()[0]:.7f}")
					train_loss_meter.reset()
				
				accelerator.wait_for_everyone()
				
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
