import torch
from torchvision import transforms


def preprocess_images(images):
	# images: BCHW, [0, 1]
	assert images.ndim == 4
	
	transform_func = transforms.Compose([
		transforms.Resize(
			(229, 229), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
		),
		transforms.CenterCrop(229)
	])
	images = transform_func(images)
	images = images * 2 - 1
	
	return images


@torch.no_grad()
def compute_fid_image_features(images, net):
	'''
		images in shape BCHW in [0., 1.]
	'''
	images = images.contiguous()
	images = preprocess_images(images).contiguous()
	
	features = net(images)[0]
	
	return features

