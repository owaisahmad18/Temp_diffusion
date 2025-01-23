import argparse
import os
import time
import numpy as np
import torch as th
import torch.distributed as dist
import pickle
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util_duo import (
    model_and_diffusion_defaults,
    create_model_and_two_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import imageio
import torchvision.transforms.functional as F
import glob
from tqdm import tqdm
import json
import h5py
from fastmri import rss

from functools import cmp_to_key

MASK_TYPE = "ktUniform4"
CONTRAST = 'Mapping'
MASK_PATHS = '/bigdata/CMRxRecon2024/ChallengeData/MultiCoil/Mapping/TrainingSet/Mask_Task2/'
DEST = "/bigdata/CMRxRecon2024/ValResults/DiffuseRecon/Mapping/ktUniform4/"

def main():
    args = create_argparser().parse_args()
    print("USING DDIM:", args.use_ddim)
    print("Saving to", args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    dist_util.setup_dist()
    logger.configure(dir=args.save_path)

    logger.log("creating model and diffusion...")
    model, diffusion, diffusion_two = create_model_and_two_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    with open(args.data_path, 'r') as f:
        files = json.load(f)
    IMAGES = files[CONTRAST]
    prog_bar = tqdm(IMAGES)
    for image in prog_bar:
        batches  = load_data(image)
        # print(len(batches))
        print(batches[0][0].shape, batches[0][1].shape, batches[0][2].shape, batches[0][3], batches[0][4])
        for batch in batches:
            samples = []
            coarse = []
            uskspace = batch[1].cuda()
            np.save('ukspace.npy', uskspace.cpu().numpy())
            mask = batch[2].cuda()
            hw = batch[3]
            csm = batch[0].cuda()
            for _ in range(2):
                model_kwargs = {}
                sample = diffusion.p_sample_loop_condition(
                    model,
                    (args.batch_size, 6, 256, 512),
                    uskspace.unsqueeze(0),
                    mask,
                    hw,
                    csm,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    progress=True,
                )[-1]
                samples.append(sample)
            samples = th.cat(samples)
            coarse.append(samples.contiguous())

            coarse = th.stack(coarse)
            print(coarse.shape)
            aggregate = coarse.mean(dim=1)

            sample2 = diffusion_two.p_sample_loop_condition(
                model,
                (1, 6, 256, 512),
                uskspace.unsqueeze(0),
                mask,
                hw,
                csm,
                noise=aggregate.float(),
                clip_denoised=args.clip_denoised,
                model_kwargs={},
                refine=True,
                progress=True
            )
            sample2 = sample2[-1].cpu().data.numpy()
            np.save(f'{DEST}/{batch[4][2]}_{batch[4][3]}__{batch[4][0]}_{batch[4][1]}.npy', sample2)
            # print(sample2.shape)
        # result_dict = {
        #     'undersampled_input': undersampled,
        #     'ground_truth': ground_truth,
        #     'coarse': coarse.cpu().data.numpy(),
        #     'fine': sample2,
        #     'reconstruction': sample2,
        #     'undersampled_kspace': undersampled_kspace
        # }
        # np.save(os.path.join(args.save_path, image + "_" + str(slice) + ".npy"), result_dict, allow_pickle=True)

def normalize_complex(data, eps=0.):
    mag = np.abs(data)
    mag_std = mag.std()
    return data / (mag_std + eps), mag_std


def ifft2c(kdata_tensor, dim=(-2,-1), norm='ortho'):
    """
    ifft2c -  ifft2 from centered kspace data tensor
    """
    kdata_tensor_uncentered = th.fft.fftshift(kdata_tensor,dim=dim)
    image_uncentered = th.fft.ifft2(kdata_tensor_uncentered,dim=dim, norm=norm)
    image = th.fft.fftshift(image_uncentered,dim=dim)
    return image

def fft2c(kdata_tensor, dim=(-2,-1), norm='ortho'):
    """
    ifft2c -  ifft2 from centered kspace data tensor
    """
    kdata_tensor_uncentered = th.fft.ifftshift(kdata_tensor,dim=dim)
    image_uncentered = th.fft.fft2(kdata_tensor_uncentered,dim=dim, norm=norm)
    image = th.fft.ifftshift(image_uncentered,dim=dim)
    return image

def zf_recon(filename):
    '''
    load kdata and direct IFFT + RSS recon
    return shape [t,z,y,x]
    '''
    # st = time.time()
    kdata = load_kdata(filename)
    # print(f'time: {time.time() - st}')
    # st = time.time()
    kdata_tensor = th.tensor(kdata)
    image_coil = ifft2c(kdata_tensor)
    # print(f'time: {time.time() - st}')
    return kdata, image_coil.cpu().numpy()

def loadmat(filename):
    """
    Load Matlab v7.3 format .mat file using h5py.
    """
    with h5py.File(filename, 'r') as f:
        data = {}
        for k, v in f.items():
            if isinstance(v, h5py.Dataset):
                data[k] = v[()]
            elif isinstance(v, h5py.Group):
                data[k] = loadmat_group(v)
    return data

def loadmat_group(group):
    """
    Load a group in Matlab v7.3 format .mat file using h5py.
    """
    data = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            data[k] = v[()]
        elif isinstance(v, h5py.Group):
            data[k] = loadmat_group(v)
    return data

def load_kdata(filename):
    '''
    load kdata from .mat file
    return shape: [t,nz,nc,ny,nx]
    '''
    data = loadmat(filename)
    keys = list(data.keys())[0]
    kdata = data[keys]
    kdata = kdata['real'] + 1j*kdata['imag']
    return kdata

def make_gaussian_kernel(ksize: int, sigma: float = 0.5) -> th.Tensor:
    x = th.linspace(-ksize // 2 + 1, ksize // 2, ksize)
    x = x.expand(ksize, -1)
    y = x.t()
    gaussian = th.exp(-(x**2 + y**2) / (2 * sigma**2))
    return gaussian / gaussian.sum()

def estimate_sensitivity_maps_smooth(image_complex, eps=1e-6):
    """
    Estimate sensitivity maps using adaptive combine method.
    """
    # Compute RSS
    rss_image = rss(image_complex, dim=2)
    
    # Estimate initial sensitivities
    sens_maps = image_complex / (rss_image.unsqueeze(2) + eps)
    f, s, coil, h, w = sens_maps.shape
    # print(shape)
    
    # Apply Gaussian smoothing (this is a simplified version, consider using proper 2D Gaussian filter)
    kernel_size = 5
    kernel = make_gaussian_kernel(kernel_size, sigma=0.5)[None, None, ...]
    # print(kernel.shape)
    sens_maps = sens_maps.view(f*coil*s, 1, h, w)

    real_smooth = th.nn.functional.conv2d(sens_maps.real.float(), kernel, padding=kernel_size//2)
    imag_smooth = th.nn.functional.conv2d(sens_maps.imag.float(), kernel, padding=kernel_size//2)
    
    sens_maps_smooth = th.complex(real_smooth, imag_smooth)
    sens_maps_smooth = sens_maps_smooth.view(f, s, coil, h, w)
    
    # Normalize smoothed sensitivity maps
    sens_maps_norm = sens_maps_smooth / (th.sum(th.abs(sens_maps_smooth)**2, dim=1, keepdim=True).sqrt() + eps)
    
    return sens_maps_norm


def load_data(data_path):

    pid = data_path.split('/')[-2]
    fname = data_path.split('/')[-1]

    mask = os.path.join(MASK_PATHS, pid, fname.replace('.mat', f'_mask_{MASK_TYPE}.mat'))
    mask = loadmat(mask)['mask'] # [FRAME X H X W]

    kspace, _ = zf_recon(data_path) # [FRAME x SLICE x Coil x H x W]

    mask = th.from_numpy(mask)
    kspace = th.from_numpy(kspace)

    uskspace = kspace * mask[:, None, None, ...]
    # np.save('uks.npy', uskspace)
    usimagespace = ifft2c(uskspace)
    f, s, c, h, w = usimagespace.shape
    s1 = s//2 - 1
    s2 = s//2 + 1
    if s == 1:
        s1, s2 = 0, 1
    gt_images = ifft2c(kspace[:, s1:s2])
    usimagespace = usimagespace[:, s1:s2]
    slices = [s1, s2-1]
    csm = estimate_sensitivity_maps_smooth(usimagespace)

    fused = th.sum(usimagespace * csm.conj(), dim=2) #[F, S, H, W]
    # np.save('fused.npy', fused.numpy())

    # we do batching here now
    batches = []
    # print(fused.shape, mask.shape)
    for i in range(2):
        for j in range(f):
            frame1 = fused[j, i].numpy()
            frame2 = fused[(j+1)%f, i].numpy()
            frame3 = fused[(j+2)%f, i].numpy()
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
            out = th.from_numpy(out)
            out = out / th.abs(out).max()
            f1 = out[:2]
            f1 = f1[0] + f1[1] * 1j
            f1 = fft2c(f1)
            f2 = out[2:4]
            f2 = f2[0] + f2[1] * 1j
            f2 = fft2c(f2)
            f3 = out[4:]
            f3 = f3[0] + f3[1] * 1j
            f3 = fft2c(f3)
            # print(out.shape)
            # print(f1.shape, f2.shape, f3.shape)
            target_kspace = th.stack([f1.real, f1.imag, f2.real, f2.imag, f3.real, f3.imag])
            # print('TARGET: ', target_kspace.shape) 

            mask_f = mask[[j, (j+1)%f, (j+2)%f], ...]
            csm_f = csm[[j, (j+1)%f, (j+2)%f], i, ...]
            out = th.nn.functional.interpolate(out.unsqueeze(0), (256, 512), mode='nearest-exact').squeeze()
            assert out.min().item() >= -1 and out.max().item() <= 1, f"Got Values: {out.min().item()} {out.max().item()}"
            batches.append([csm_f, target_kspace, mask_f, [h, w], [slices[i], j, pid, fname]])
    return batches


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=5,
        use_ddim=False,
        model_path="",
        data_path="",
        save_path="",
        contrast="t1",
        acc_factor=4,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
