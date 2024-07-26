import argparse
import itertools
import torch
import torchaudio
import torchvision
from torchvision.io import VideoReader
torchvision.set_video_backend("video_reader")

from avgen.evaluations.avsync import compute_sync_metrics_on_av


def load_audio(audio_path):
	if audio_path.endswith(".mp4"): # load with torchvision
		av_reader = VideoReader(audio_path, stream="audio")
		audio_sr = int(av_reader.get_metadata()["audio"]["framerate"][0])
		
		audio_waveform = []
		for frame in itertools.takewhile(lambda x: x['pts'] <= 2.0, av_reader.seek(0.0)):
			if frame['pts'] >= 0.0 and frame['pts'] <= 2.0:
				frame_data = frame["data"]
				t, c = frame_data.shape
				frame_data = frame_data.contiguous().view(c, t).contiguous()
				audio_waveform.append(frame_data)  # (c, t)
		audio_waveform = torch.cat(audio_waveform, dim=1)
		
	else:
		audio_waveform, audio_sr = torchaudio.load(audio_path)
		audio_waveform = audio_waveform[:, :int(audio_sr * 2.0)].contiguous()
		
	return audio_waveform, audio_sr


def load_video(video_path):
	av_reader = VideoReader(video_path, stream="video")
	
	video_fps = 6.
	video_num_frame = 12
	
	av_reader.set_current_stream("video")
	keyframe_coverage = 1. / video_fps
	
	video_frames = []
	frame_timestamp = 0.
	for i, frame in enumerate(itertools.takewhile(
			lambda x: x['pts'] <= clip_duration + keyframe_coverage / 2.,
			av_reader.seek(0.)
	)):
		if frame["pts"] + 1e-6 >= frame_timestamp:
			video_frames.append(frame["data"])  # (c, h, w) tensor [0, 255]
			frame_timestamp += keyframe_coverage
		
		if len(video_frames) == video_num_frame:
			break
	
	assert len(video_frames) == video_num_frame, "video not long enough to sample 12 frames in 2 seconds"
	
	video_frames = torch.stack(video_frames, dim=1).float() / 255. # (c f h w) in [0, 1]
	
	return video_frames
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--audio", type=str, required=True, help="path of audio")
	parser.add_argument("--video", type=str, required=True, help="path of video to evaluate")
	parser.add_argument("--groundtruth_video", type=str, default="", help="path of video to be reference")
	parser.add_argument("--metric", type=str, default="alignsync", help="metric name: alignsync, relsync, or avsync_score")
	args = parser.parse_args()
	
	clip_duration = 2.0
	
	# Load 2-second audio
	audio_waveform, audio_sr = load_audio(args.audio)
	
	# Load 2-second video with 12 frames
	video = load_video(args.video)
	if args.metric in ["alignsync", "relsync"]:
		groundtruth_video = load_video(args.groundtruth_video)
	else:
		groundtruth_video = None
	
	score = compute_sync_metrics_on_av(
		audio_waveform=audio_waveform,
		audio_sr=audio_sr,
		video=video,
		groundtruth_video=groundtruth_video,
		metric=args.metric,
	)
	
	print("########################################")
	print(f"audio: {args.audio}")
	print(f"video: {args.video}")
	print(f"groundtruth_video: {args.groundtruth_video}")
	print(f"{args.metric}: {score.item()}")
	
	
