import numpy as np
import cv2
from torchvision import transforms
from torch import nn
import torch
from scipy.ndimage import gaussian_filter1d
import tsaug
from typing import List


class RootSet(object):
    def __init__(
        self,
        tree=[
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28],
            [29, 30, 31, 32, 33, 34, 35, 36, 37],
        ],
    ):
        self.tree = tree

    def __call__(self, X):
        X = im2skeleton(X)
        for t in self.tree:
            # print(X[t].size(), X[[t[0]]].size())
            # print(X.size(), X.size(), X[:, t[0]].size())
            X[:, t] -= X[:, [t[0]]]
        return skeleton2im(X)


class Subset(object):
    def __init__(self, subset=None):
        ''' selects subset of the channel dimensions. output will have smaller number of channels'''
        self.subset = subset
        if self.subset is None:
            self.subset = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 38, 39, 40, 41,
                                    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                                    59, 60, 61, 62, 63, 64, 65, 66, 67])
    def __call__(self, X):
        X_copy = torch.zeros_like(X)
        X_copy[:, self.subset] = X[:, self.subset]
        return X_copy


class GaussSmooth(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X):
        sigma = (torch.rand(size=(1,)).item() * self.sigma) + 0.01
        X = gaussian_filter1d(X, sigma, axis=0)
        return torch.from_numpy(X)


class Squeeze(object):
    def __call__(self, X):
        return torch.squeeze(X)


class toTorch(object):
    def __call__(self, X):
        return torch.from_numpy(X).float()


class MinMaxNormalization(object):
    """ converts into [0, 1] range"""

    def __init__(self, mi, ma):
        self.mi, self.ma = mi, ma
        assert self.ma > self.mi

    def __call__(self, X):
        X = (X - self.mi) / (self.ma - self.mi)
        return torch.clamp(X, min=0, max=1)


class NormalizeTimeSeries(object):
    def __init__(self, m, s):
        self.m, self.s = m, s

    def __call__(self, X):
        X = (X - self.m.flatten()) / self.s.flatten()
        return X


class Jitter(object):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, X):
        return X + (torch.rand(X.size()) * self.sigma)


class Translate(object):
    # x, y rastgele olmali
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, X):
        # to give independent trasnlation on x and y axis
        return skeleton2im(im2skeleton(X) + (torch.rand(2) * self.sigma))


class TranslateGroup(object):
    # x, y rastgele olmali
    def __init__(self, sigma: float, group: List[int]):
        self.sigma = sigma
        self.group = group

    def __call__(self, X):
        X = im2skeleton(X)
        for g in self.group:
            X[:, g] += torch.rand(2) * self.sigma
        return skeleton2im(X)


class Scaling(object):
    # Scaling each channel seperately does not really make sense in human pose case
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, X):
        assert X.ndim == 2
        scalingFactor = (torch.rand(2) * self.sigma) + 1
        # to give independent trasnlation on x and y axis
        return skeleton2im(im2skeleton(X) * scalingFactor)


def im2skeleton(X):
    assert X.ndim == 2
    return X.reshape(X.shape[0], 38, 2)


def skeleton2im(X):
    assert X.ndim == 3
    return X.reshape(X.shape[0], 38 * 2)


class Mirror(object):
    # bunun icin bir de mirrorini almak lazim
    def __call__(self, X):
        # select origin randomly;
        assert X.ndim == 2
        X = im2skeleton(X)
        X[:, :, 0] *= -1
        c = torch.clone(X[:, 19:])

        # T x CJ
        X[:, 19:] = X[:, :19]
        X[:, :19] = c

        return skeleton2im(X)


class TimeWarp(object):
    def __init__(self):
        pass

    def __call__(self, X):
        # select origin randomly;
        # X = im2skeleton(X)
        X = X.unsqueeze(0)  # (N, T) # (1, T, C)
        warper = tsaug.TimeWarp(  # (N, T, C)
            n_speed_change=5,
            max_speed_ratio=3.0,
            seed=torch.randint(low=0, high=1024, size=(1,)).item(),
        )
        X = warper.augment(X.data.numpy())
        # X = DA_TimeWarp(X.data.numpy(), sigma=0.4)
        return torch.squeeze(torch.from_numpy(X))


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze(p @ R)


class Rotate(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X):
        angle = torch.rand(size=(1,)).item() * self.sigma
        X = im2skeleton(X).data.numpy()
        return torch.from_numpy(skeleton2im(rotate(X, degrees=angle)))


class Invert(object):
    def __call__(self, X):
        return torch.flip(X, dims=(0,))


class Shear(object):
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    def random_shear_matrix(self):
        s1 = torch.rand((1,)).item() * self.s1
        s2 = torch.rand((1,)).item() * self.s2
        shear1 = np.array(([1, 0], [s1, 1]))
        shear2 = np.array(([1, s2], [0, 1]))
        shear = shear1 @ shear2

        return torch.from_numpy(shear).float()

    def __call__(self, X):
        return skeleton2im(im2skeleton(X) @ self.random_shear_matrix())

