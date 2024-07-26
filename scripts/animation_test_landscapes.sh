#!/bin/sh

exp_root=$1
iter=$2
audio_cfg=$3

python -W ignore scripts/animation_gen.py \
--exp_root ${exp_root} \
--checkpoint ${iter} \
--dataset Landscapes \
--image_h 256 \
--image_w 256 \
--video_fps 6 \
--video_num_frame 12 \
--num_clips_per_video 3 \
--audio_guidance_scale ${audio_cfg} \
--text_guidance_scale 1.0 \
--random_seed 0

wait

python -W ignore scripts/animation_eval.py \
--dataset Landscapes \
--generated_video_root ${exp_root}/evaluations/checkpoint-${iter}/AG-${audio_cfg}_TG-1.0/seed-0/videos \
--result_save_path ${exp_root}/evaluations/checkpoint-${iter}/AG-${audio_cfg}_TG-1.0/seed-0/metrics/eval_result.json \
--num_clips_per_video 3 \
--image_h 256 \
--image_w 256 \
--eval_fid \
--eval_fvd \
--eval_clipsim \
--eval_relsync \
--eval_alignsync