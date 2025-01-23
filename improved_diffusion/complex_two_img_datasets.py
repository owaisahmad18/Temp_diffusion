import os
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import h5py
import random
from fastmri import rss
from fastmri.data import transforms as T
from fastmri.fftc import ifft2c_new
from torchvision.transforms.functional import resize
from tqdm import tqdm
import json
from torch import fft as tfft
from torch.nn.functional import interpolate
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip


def load_data(
    *,
    train_path,
    val_path,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    num_dl_workers=4,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    train_path = '../2023-dataset.json'
    
    if not train_path:
        raise ValueError("unspecified train.txt")
    # if not val_path:
    #     # val is ignored for iDDPM training
    #     raise ValueError("unspecified val path")

    all_files = _list_image_files_recursively(train_path)

    print("Found", len(all_files), "files")
    print("\n".join(all_files[:16]))

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    # dataset = ImageDataset(
    #     image_size,
    #     all_files,
    #     classes=classes,
    #     shard=MPI.COMM_WORLD.Get_rank(),
    #     num_shards=MPI.COMM_WORLD.Get_size(),
    # )
    dataset = Dataset2023(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )

    if deterministic:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_dl_workers,
            drop_last=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_dl_workers,
            drop_last=True,
        )
    while True:
        yield from loader


# def _list_image_files_recursively(data_dir: str):
#     assert data_dir.endswith("txt"), "Expected text file"
#     with open(data_dir, "r") as f:
#         files = f.readlines()
#         files = [f.strip() for f in files]
#     results = files
#     return results

import json

def _list_image_files_recursively(data_dir: str):
    assert data_dir.endswith("json"), "Expected JSON file"
    with open(data_dir, "r") as f:
        files = json.load(f)
    results = files
    return results


def normalize_complex(data, eps=1e-8):
    mag = np.abs(data)
    mag_std = mag.std()
    return data / (mag_std + eps), mag_std


def ifft(x: torch.Tensor) -> torch.Tensor:
    x = tfft.ifftshift(x, dim=[-2,-1])
    x = tfft.ifft2(x, dim=[-2, -1], norm='ortho')
    x = tfft.fftshift(x, dim=[-2, -1])
    return x

def make_gaussian_kernel(ksize: int, sigma: float = 0.5) -> torch.Tensor:
    x = torch.linspace(-ksize // 2 + 1, ksize // 2, ksize)
    x = x.expand(ksize, -1)
    y = x.t()
    gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return gaussian / gaussian.sum()

def estimate_sensitivity_maps_smooth(image_complex, eps=1e-6):
    """
    Estimate sensitivity maps using adaptive combine method.
    """
    # Compute RSS
    rss_image = rss(image_complex, dim=1)
    
    # Estimate initial sensitivities
    sens_maps = image_complex / (rss_image.unsqueeze(1) + eps)
    f, coil, h, w = sens_maps.shape
    # print(shape)
    
    # Apply Gaussian smoothing (this is a simplified version, consider using proper 2D Gaussian filter)
    kernel_size = 5
    kernel = make_gaussian_kernel(kernel_size, sigma=0.5)[None, None, ...]
    # print(kernel.shape)
    sens_maps = sens_maps.view(f*coil, 1, h, w)

    real_smooth = torch.nn.functional.conv2d(sens_maps.real, kernel, padding=kernel_size//2)
    imag_smooth = torch.nn.functional.conv2d(sens_maps.imag, kernel, padding=kernel_size//2)
    
    sens_maps_smooth = torch.complex(real_smooth, imag_smooth)
    sens_maps_smooth = sens_maps_smooth.view(f, coil, h, w)
    
    # Normalize smoothed sensitivity maps
    sens_maps_norm = sens_maps_smooth / (torch.sum(torch.abs(sens_maps_smooth)**2, dim=1, keepdim=True).sqrt() + eps)
    
    return sens_maps_norm

class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

        self.build_self()

    def build_self(self):
        mats = []
        # if os.path.exists('./dataset-mat.json'):
        #     with open('./dataset-mat.json', 'r') as f:
        #         mats = json.load(f)
        #     self.mats = mats
        #     return
        if os.path.exists('./dataset_os.json'):
            print("LOADING OVERSAMPLED DATASET")
            with open("./dataset_os.json", "r") as f:
                mats = json.load(f)
            random.shuffle(mats)
            self.mats = mats
            return
        prog = tqdm(self.local_images, desc="Building Dataset Meta-Data")
        for file in prog:
            prog.set_description(f'{file.split("/")[-1]}')
            arr = np.load(file)
            for f in range(arr.shape[0]):
                mats.append({"file": file, "frame": f, "total": arr.shape[0]})

        random.shuffle(mats)
        with open('./dataset-mat.json', 'w') as f:
            json.dump(mats, f, indent=2)

        self.mats = mats

    def __len__(self):
        return len(self.mats)

    def __getitem__(self, idx):
        """
        fastMRI is preprocessed and stored as pickle files, where kspace raw data is stored under 'img';
        """

        item = self.mats[idx]
        fpath = item["file"]
        f0 = item["frame"]  # curr-frame
        t = item["total"]
        f1 = (f0 + 1) % t  # cframe + 1
        f2 = (f0 + 2) % t  # cframe + 2

        kspace = torch.from_numpy(np.load(fpath))
        image_space = ifft(kspace)
        sense_map_e = estimate_sensitivity_maps_smooth(image_space)
        fused = torch.sum(image_space * sense_map_e.conj(), dim=1).numpy()

        # frame0 = rss(ifft2c_new(T.to_tensor(kspace[f0])), dim=0)
        # frame1 = rss(ifft2c_new(T.to_tensor(kspace[f1])), dim=0)
        # frame2 = rss(ifft2c_new(T.to_tensor(kspace[f2])), dim=0)


        # frame0 = torch.view_as_complex(frame0).numpy()
        # frame1 = torch.view_as_complex(frame1).numpy()
        # frame2 = torch.view_as_complex(frame2).numpy()
        frame0 = fused[f0]
        frame1 = fused[f1]
        frame2 = fused[f2]

        if np.isnan(frame0).any() or np.isnan(frame1).any() or np.isnan(frame2).any():
            raise Exception(f"NANS IN RSS {fpath}")

        frame0, _ = normalize_complex(frame0)
        frame1, _ = normalize_complex(frame1)
        frame2, _ = normalize_complex(frame2)

        real0 = np.real(frame0)
        imag0 = np.imag(frame0)

        real1 = np.real(frame1)
        imag1 = np.imag(frame1)

        real2 = np.real(frame2)
        imag2 = np.imag(frame2)

        out = np.stack([real0, imag0, real1, imag1, real2, imag2]).astype(np.float32)
        assert ~np.isnan(out).any(), f"FOUND NAN AFTER NORM {fpath}"
        out = torch.from_numpy(out)
        max_val = abs(out).max()
        out /= max_val
        # out = resize(out, (320, 320), antialias=True, interpolation=0)
        out = interpolate(out.unsqueeze(0), (256, 512), mode='nearest-exact').squeeze()
        np.save("input-batch.npy", out.numpy())
        assert ~np.isnan(out.numpy()).any(), f"FOUND NAN AFTER RESIZE {fpath}"
        return out, {}


class Dataset2023(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

        self.build_self()

    def build_self(self):
        mats = []
        # if os.path.exists('./dataset-mat.json'):
        #     with open('./dataset-mat.json', 'r') as f:
        #         mats = json.load(f)
        #     self.mats = mats
        #     return
        if os.path.exists('../2023-dataset.json'):
            print("LOADING DATASET 2023-dataset.json")
            with open('../2023-dataset.json', "r") as f:
                mats = json.load(f)
            # random.shuffle(mats)
            self.mats = mats
            return
        else:
            raise FileNotFoundError("Can't find trainf files")

    def __len__(self):
        return len(self.mats)

    def __getitem__(self, idx):
        """
        fastMRI is preprocessed and stored as pickle files, where kspace raw data is stored under 'img';
        """

        current_file = self.mats[idx]
        path, name = os.path.split(current_file)
        path = path + '/'
        slice_0 = int(name[-7])
        # case 1, 0, +1, +2 all exist

        if os.path.exists(path + name[:-7] + str(slice_0+1) + name[-6:]): # s+1
            if os.path.exists(path + name[:-7] + str(slice_0+2) + name[-6:]): # s+2
                slice_1, slice_2 = slice_0 + 1, slice_0 + 2
            else: # only s+1
                slice_0, slice_1, slice_2 = slice_0 - 1, slice_0, slice_0 + 1
        else: # s is last,
            slice_0, slice_1, slice_2 = slice_0 - 2, slice_0 -1, slice_0
        
        frame1 = torch.from_numpy(np.load(path + name[:-7] + str(slice_0) + name[-6:], allow_pickle=True)[()]['complex-image-space'])
        frame2 = torch.from_numpy(np.load(path + name[:-7] + str(slice_1) + name[-6:], allow_pickle=True)[()]['complex-image-space'])
        frame3 = torch.from_numpy(np.load(path + name[:-7] + str(slice_2) + name[-6:], allow_pickle=True)[()]['complex-image-space'])

        if 'multi_coil' in path:
            csm = estimate_sensitivity_maps_smooth(frame1.unsqueeze(0)).squeeze()
            frame1 = torch.sum(frame1 * csm.conj(), dim=0).numpy()
            csm = estimate_sensitivity_maps_smooth(frame2.unsqueeze(0)).squeeze()
            frame2 = torch.sum(frame2 * csm.conj(), dim=0).numpy()
            csm = estimate_sensitivity_maps_smooth(frame3.unsqueeze(0)).squeeze()
            frame3 = torch.sum(frame3 * csm.conj(), dim=0).numpy()
            
        if np.isnan(frame1).any() or np.isnan(frame2).any() or np.isnan(frame3).any():
            raise Exception(f"NANS IN RSS {path+name}")

        frame1, _ = normalize_complex(frame1)
        frame2, _ = normalize_complex(frame2)
        frame3, _ = normalize_complex(frame3)

        real0 = np.real(frame1)
        imag0 = np.imag(frame1)

        real1 = np.real(frame2)
        imag1 = np.imag(frame2)

        real2 = np.real(frame3)
        imag2 = np.imag(frame3)

        out = np.stack([real0, imag0, real1, imag1, real2, imag2]).astype(np.float32)
        assert ~np.isnan(out).any(), f"FOUND NAN AFTER NORM {path+name}"
        out = torch.from_numpy(out)
        max_val = abs(out).max()
        out /= max_val
        # out = resize(out, (320, 320), antialias=True, interpolation=0)
        out = interpolate(out.unsqueeze(0), (160, 512), mode='nearest-exact').squeeze()

        out = RandomHorizontalFlip()(out)
        out = RandomVerticalFlip()(out)

        np.save("input-batch.npy", out.numpy())
        assert ~np.isnan(out.numpy()).any(), f"FOUND NAN AFTER RESIZE {path+name}"
        return out, {}
    

if __name__ == '__main__':
    from tqdm.auto import tqdm
    dataset = Dataset2023(None, [])
    pbar = tqdm(range(len(dataset)))
    for i in pbar:
        y = dataset[i][0]
        m = y.min().item()
        M = y.max().item()
        shape = list(y.shape)
        pbar.set_description(f'[{m:2.4f}/{M:2.4f}] @ [{shape}]')