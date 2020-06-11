import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

__all__ = ['device',
           'to_multiple',
           'BatchLinear',
           'init_layer_weights',
           'init_sequential_weights',
           'compute_dists',
           'pad_concat',
           'gaussian_logpdf']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
"""Device perform computations on."""


def to_multiple(x, multiple):
    """Convert `x` to the nearest above multiple.

    Args:
        x (number): Number to round up.
        multiple (int): Multiple to round up to.

    Returns:
        number: `x` rounded to the nearest above multiple of `multiple`.
    """
    if x % multiple == 0:
        return x
    else:
        return x + multiple - x % multiple

def to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.squeeze().detach().cpu().numpy()

class BatchLinear(nn.Linear):
    """Helper class for linear layers on order-3 tensors.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): Use a bias. Defaults to `True`.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(BatchLinear, self).__init__(in_features=in_features,
                                          out_features=out_features,
                                          bias=bias)
        nn.init.xavier_normal_(self.weight, gain=1)
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        """Forward pass through layer. First unroll batch dimension, then pass
        through dense layer, and finally reshape back to a order-3 tensor.

        Args:
              x (tensor): Inputs of shape `(batch, n, in_features)`.

        Returns:
              tensor: Outputs of shape `(batch, n, out_features)`.
        """
        num_functions, num_inputs = x.shape[0], x.shape[1]
        x = x.view(num_functions * num_inputs, self.in_features)
        out = super(BatchLinear, self).forward(x)
        return out.view(num_functions, num_inputs, self.out_features)


def init_layer_weights(layer):
    """Initialize the weights of a :class:`nn.Layer` using Glorot
    initialization.

    Args:
        layer (:class:`nn.Module`): Single dense or convolutional layer from
            :mod:`torch.nn`.

    Returns:
        :class:`nn.Module`: Single dense or convolutional layer with
            initialized weights.
    """
    nn.init.xavier_normal_(layer.weight, gain=1)
    nn.init.constant_(layer.bias, 1e-3)


def init_sequential_weights(model, bias=0.0):
    """Initialize the weights of a nn.Sequential model with Glorot
    initialization.

    Args:
        model (:class:`nn.Sequential`): Container for model.
        bias (float, optional): Value for initializing bias terms. Defaults
            to `0.0`.

    Returns:
        (nn.Sequential): model with initialized weights
    """
    for layer in model:
        if hasattr(layer, 'weight'):
            nn.init.xavier_normal_(layer.weight, gain=1)
        if hasattr(layer, 'bias'):
            nn.init.constant_(layer.bias, bias)
    return model


def compute_dists(x, y):
    """Fast computation of pair-wise distances for the 1d case.

    Args:
        x (tensor): Inputs of shape `(batch, n, 1)`.
        y (tensor): Inputs of shape `(batch, m, 1)`.

    Returns:
        tensor: Pair-wise distances of shape `(batch, n, m)`.
    """
    assert x.shape[2] == 1 and y.shape[2] == 1, \
        'The inputs x and y must be 1-dimensional observations.'
    return (x - y.permute(0, 2, 1)) ** 2


def pad_concat(t1, t2):
    """Concat the activations of two layer channel-wise by padding the layer
    with fewer points with zeros.

    Args:
        t1 (tensor): Activations from first layers of shape `(batch, n1, c1)`.
        t2 (tensor): Activations from second layers of shape `(batch, n2, c2)`.

    Returns:
        tensor: Concatenated activations of both layers of shape
            `(batch, max(n1, n2), c1 + c2)`.
    """
    if t1.shape[2] > t2.shape[2]:
        padding = t1.shape[2] - t2.shape[2]
        if padding % 2 == 0:  # Even difference
            t2 = F.pad(t2, (int(padding / 2), int(padding / 2)), 'reflect')
        else:  # Odd difference
            t2 = F.pad(t2, (int((padding - 1) / 2), int((padding + 1) / 2)),
                       'reflect')
    elif t2.shape[2] > t1.shape[2]:
        padding = t2.shape[2] - t1.shape[2]
        if padding % 2 == 0:  # Even difference
            t1 = F.pad(t1, (int(padding / 2), int(padding / 2)), 'reflect')
        else:  # Odd difference
            t1 = F.pad(t1, (int((padding - 1) / 2), int((padding + 1) / 2)),
                       'reflect')

    return torch.cat([t1, t2], dim=1)


def gaussian_logpdf(inputs, mean, sigma, reduction=None):
    """Gaussian log-density.

    Args:
        inputs (tensor): Inputs.
        mean (tensor): Mean.
        sigma (tensor): Standard deviation.
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        tensor: Log-density.
    """
    dist = Normal(loc=mean, scale=sigma)
    logp = dist.log_prob(inputs)

    if not reduction:
        return logp
    elif reduction == 'sum':
        return torch.sum(logp)
    elif reduction == 'mean':
        return torch.mean(logp)
    elif reduction == 'batched_mean':
        return torch.mean(torch.sum(logp, 1))
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')


def img_mask_to_np_input(img, mask, normalize=True):
    """
    Given an image and a mask, return x and y tensors expected by Neural
    Process. Specifically, x will contain indices of unmasked points, e.g.
    [[1, 0], [23, 14], [24, 19]] and y will contain the corresponding pixel
    intensities, e.g. [[0.2], [0.73], [0.12]] for grayscale or
    [[0.82, 0.71, 0.5], [0.42, 0.33, 0.81], [0.21, 0.23, 0.32]] for RGB.

    Parameters
    ----------
    img : torch.Tensor
        Shape (N, C, H, W). Pixel intensities should be in [0, 1]

    mask : torch.ByteTensor
        Binary matrix where 0 corresponds to masked pixel and 1 to a visible
        pixel. Shape (N, H, W). Note the number of unmasked pixels must be the
        SAME for every mask in batch.

    normalize : bool
        If true normalizes pixel locations x to [-1, 1] and pixel intensities to
        [-0.5, 0.5]
    """
    batch_size, num_channels, height, width = img.size()
    # Create a mask which matches exactly with image size which will be used to
    # extract pixel intensities
    mask_img_size = mask.unsqueeze(1).repeat(1, num_channels, 1, 1)
    # Number of points corresponds to number of visible pixels in mask, i.e. sum
    # of non zero indices in a mask (here we assume every mask has same number
    # of visible pixels)
    num_points = mask[0].nonzero().size(0)
    # Compute non zero indices
    # Shape (num_nonzeros, 3), where each row contains index of batch, height and width of nonzero
    nonzero_idx = mask.nonzero()
    # The x tensor for Neural Processes contains (height, width) indices, i.e.
    # 1st and 2nd indices of nonzero_idx (in zero based indexing)
    x = nonzero_idx[:, 1:].view(batch_size, num_points, 2).float()
    # The y tensor for Neural Processes contains the values of non zero pixels
    y = img[mask_img_size].view(batch_size, num_channels, num_points)
    # Ensure correct shape, i.e. (batch_size, num_points, num_channels)
    y = y.permute(0, 2, 1)

    if normalize:
        # TODO: make this separate for height and width for non square image
        # Normalize x to [-1, 1]
        x = (x - float(height) / 2) / (float(height) / 2)
        # Normalize y's to [-0.5, 0.5]
        y -= 0.5

    return x, y

def xy_to_img(x, y, img_size):
    """Given an x and y returned by a Neural Process, reconstruct image.
    Missing pixels will have a value of 0.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, 2) containing normalized indices.

    y : torch.Tensor
        Shape (batch_size, num_points, num_channels) where num_channels = 1 for
        grayscale and 3 for RGB, containing normalized pixel intensities.

    img_size : tuple of ints
        E.g. (1, 32, 32) for grayscale or (3, 64, 64) for RGB.
    """
    batch_size, channel, height, width = img_size
    # Unnormalize x and y
    x = x * float(height / 2) + float(height / 2)
    x = x.long()
    y += 0.5
    # Permute y so it matches order expected by image
    # (batch_size, num_points, num_channels) -> (batch_size, num_channels, num_points)
    y = y.permute(0, 2, 1)
    # Initialize empty image
    img = torch.zeros(img_size)
    for i in range(batch_size):
        img[i, :, x[i, :, 0], x[i, :, 1]] = y[i, :, :]
    return img

def channel_last(x):
    return x.transpose(1, 2).transpose(2, 3)

class RunningAverage:
    """Maintain a running average."""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        """Reset the running average."""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """Update the running average.

        Args:
            val (float): Value to update with.
            n (int): Number elements used to compute `val`.
        """
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt