from .tstrans import *
from torchvision.transforms import Compose
from torchaudio import transforms as audio_transforms
class TimeSeriesTransformEval(object):
    def __init__(self,):
        data_transforms = [
            toTorch(),
            RootSet(),
            Subset(),
            #MinMaxNormalization(mi=-960, ma=960)
        ]
        self.transform = Compose(data_transforms)

    def __call__(self, X):
        return self.transform(X)
    
    
class TimeSeriesTransformTrain(object):
    def __init__(
        self,
        sigma_jitter=30,
        sigma_scaling=0.5,
        sigma_gauss=0.5,
        sigma_rotate=30,
        sigma_sheer=0.5,
        mask_freq=5,
        mask_time=5,
    ):
        data_transforms = [
            toTorch(),
            RootSet(),
            GaussSmooth(sigma_gauss),
            Jitter(sigma_jitter),
            Scaling(sigma_scaling),
            Shear(sigma_sheer, sigma_sheer),
            Rotate(sigma_rotate),
            #TimeWarp(),
            transforms.RandomApply([Mirror()], p=0.5),
            RootSet(),
            audio_transforms.FrequencyMasking(
                freq_mask_param=mask_freq, iid_masks=False
            ),
            audio_transforms.TimeMasking(time_mask_param=mask_time, iid_masks=False),
            Subset(),
        ]

        self.transform = Compose(data_transforms)

    def __call__(self, X):
        return self.transform(X)