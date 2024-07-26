import torch
import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
	u, s, v = torch.svd(mat)
	si = torch.where(s < eps, s, torch.sqrt(s))
	return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
	sqrt_sigma = _symmetric_matrix_square_root(sigma)
	sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
	return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
	'''Estimate a covariance matrix given data.

	Covariance indicates the level to which two variables vary together.
	If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
	then the covariance matrix element `C_{ij}` is the covariance of
	`x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

	Args:
		m: A 1-D or 2-D array containing multiple variables and observations.
			Each row of `m` represents a variable, and each column a single
			observation of all those variables.
		rowvar: If `rowvar` is True, then each row represents a
			variable, with observations in the columns. Otherwise, the
			relationship is transposed: each column represents a variable,
			while the rows contain observations.

	Returns:
		The covariance matrix of the variables.
	'''
	if m.dim() > 2:
		raise ValueError('m has more than 2 dimensions')
	if m.dim() < 2:
		m = m.view(1, -1)
	if not rowvar and m.dim() != 1:
		m = m.t()

	fact = 1.0 / (m.size(1) - 1) # unbiased estimate
	m -= torch.mean(m, dim=1, keepdim=True)
	mt = m.t()  # if complex: mt = m.t().conj()
	
	return fact * m.matmul(mt).squeeze()


def frechet_distance(x1, x2, eps=1e-6):
	"""Numpy implementation of the Frechet Distance.
	The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
	and X_2 ~ N(mu_2, C_2) is
			d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

	Stable version by Dougal J. Sutherland.

	Params:
	-- mu1   : Numpy array containing the activations of a layer of the
			   inception net (like returned by the function 'get_predictions')
			   for generated samples.
	-- mu2   : The sample mean over activations, precalculated on an
			   representative data set.
	-- sigma1: The covariance matrix over activations for generated samples.
	-- sigma2: The covariance matrix over activations, precalculated on an
			   representative data set.

	Returns:
	--   : The Frechet Distance.
	"""
	
	x1 = x1.numpy()
	x2 = x2.numpy()
	
	mu1 = np.mean(x1, axis=0)
	sigma1 = np.cov(x1, rowvar=False)
	
	mu2 = np.mean(x2, axis=0)
	sigma2 = np.cov(x2, rowvar=False)

	mu1 = np.atleast_1d(mu1)
	mu2 = np.atleast_1d(mu2)

	sigma1 = np.atleast_2d(sigma1)
	sigma2 = np.atleast_2d(sigma2)

	assert mu1.shape == mu2.shape, \
		'Training and test mean vectors have different lengths'
	assert sigma1.shape == sigma2.shape, \
		'Training and test covariances have different dimensions'

	diff = mu1 - mu2

	# Product might be almost singular
	covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
	if not np.isfinite(covmean).all():
		msg = ('fid calculation produces singular product; '
			   'adding %s to diagonal of cov estimates') % eps
		print(msg)
		offset = np.eye(sigma1.shape[0]) * eps
		covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

	# Numerical error might give slight imaginary component
	if np.iscomplexobj(covmean):
		if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
			m = np.max(np.abs(covmean.imag))
			raise ValueError('Imaginary component {}'.format(m))
		covmean = covmean.real

	tr_covmean = np.trace(covmean)

	return (diff.dot(diff) + np.trace(sigma1)
			+ np.trace(sigma2) - 2 * tr_covmean)