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

from functools import cmp_to_key

def fname_sorter(a, b):
    a_id = int(a[1:4])
    b_id = int(b[1:4])

    if a_id < b_id:
        return -1
    elif a_id == b_id:
        a_slice = int(a[-1])
        b_slice = int(b[-1])
        if a_slice < b_slice:
            return -1
        elif a_slice == b_slice:
            return 0
        else:
            return 1
    else:
        return 1


def get_mask(dtype, acc):
    if dtype == 't1':
        if acc == 4:
            mask = np.load('acc4-t1-mask.npy')
        elif acc == 8:
            mask = np.load('acc8-t1-mask.npy')
        elif acc == 10:
            mask = np.load('acc10-t1-mask.npy')
    elif dtype == 't2':
        if acc == 4:
            mask = np.load('acc4-t2-mask.npy')
        elif acc == 8:
            mask = np.load('acc8-t2-mask.npy')
        elif acc == 10:
            mask = np.load('acc10-t2-mask.npy')
    else:
        raise ValueError('Invalid dtype')

    mask = th.from_numpy(mask)[None, None, ...].cuda().float()
    mask = th.cat([mask, mask], 1)
    return mask


def main():
    args = create_argparser().parse_args()
    print("USING DDIM:", args.use_ddim)

    contrast_type = 't1' if 't1' in args.contrast else 't2'
    acc_factor = int(args.acc_factor)
    images = os.listdir(args.data_path)
    images = [image[:-6] for image in images]
    # run only on t2 images
    images = [image for image in images if contrast_type in image.lower()]  
    images = list(set(images))
    images = sorted(images, key=cmp_to_key(fname_sorter))
    unique_pids = sorted(list(set([image[1:4] for image in images])))[:3]
    print(unique_pids)
    images = [image for image in images if image[1:4] in unique_pids]
    print(json.dumps(images, indent=1))
    print('Found', len(images), 'files')
    # args.save_path = dest
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
    prog_bar = tqdm(images)
    for image in prog_bar:
        slice_nums = [0,1,2,3,4,5,6,7,8] if 't1' in image.lower() else [0,1,2]
        # image = image[:-5]
        mask = get_mask('t1' if 't1' in image.lower() else 't2', acc_factor)
        for slice in slice_nums:
            # print(image, slice)
            prog_bar.set_description_str(f"{image} / {slice}")
            coarse = []
            for i in range(slice - 1, slice + 2):
                # i = i % (8 if 't1' in image.lower() else 3)
                file_name1 = image + "_" + str(i % (8 if 't1' in image.lower() else 3)) + ".npy"
                # file_name2 = image + "_" + str((i + 1) % (8 if 't1' in image.lower() else 3)) + ".pt"
                file_name2 = file_name1
                kspace, fs_gt, us_in, us_ks = load_data(args.data_path, file_name1, file_name2, args.batch_size, mask)
                # save for refining
                if i == slice:
                    input = kspace[[0]]
                    ground_truth = fs_gt.copy()
                    undersampled = us_in.copy()
                    undersampled_kspace = us_ks.copy()
                # logger.log("sampling...")
                samples = []
                for _ in range(2):
                    model_kwargs = {}
                    sample = diffusion.p_sample_loop_condition(
                        model,
                        (args.batch_size, 4, 116, 384) if contrast_type == 't2' else (args.batch_size, 4, 144, 512),
                        kspace,
                        mask,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                        progress=True
                    )[-1]
                    samples.append(sample)
                samples = th.cat(samples)
                coarse.append(samples.contiguous())

            coarse = th.stack(coarse)
            # print(coarse.shape)
            aggregate = []
            for k in range(2):

                # print(coarse[k, :, [2, 3]].mean(0).shape)
                # print(coarse[k + 1, :, [0, 1]].mean(0).shape)
                if contrast_type == 't2':
                    aggregate.append(
                        (coarse[k, :, [2, 3]].mean(0) + coarse[k + 1, :, [0, 1]].mean(0)).view(
                            1, 2, 116, 384
                        )
                        / 2
                    )
                else:
                    aggregate.append(
                        (coarse[k, :, [2, 3]].mean(0) + coarse[k + 1, :, [0, 1]].mean(0)).view(
                            1, 2, 144, 512
                        )
                        / 2
                    )
            aggregate = th.cat(aggregate, 1)
            # print(aggregate.shape)

            sample2 = diffusion_two.p_sample_loop_condition(
                model,
                (1, 4, 116, 384) if contrast_type == 't2' else (1, 4, 144, 512),
                input,
                mask,
                noise=aggregate.float(),
                clip_denoised=args.clip_denoised,
                model_kwargs={},
                refine=True,
                progress=True
            )
            sample2 = sample2[-1].cpu().data.numpy()
            result_dict = {
                'undersampled_input': undersampled,
                'ground_truth': ground_truth,
                'coarse': coarse.cpu().data.numpy(),
                'fine': sample2,
                'reconstruction': sample2,
                'undersampled_kspace': undersampled_kspace
            }
            np.save(os.path.join(args.save_path, image + "_" + str(slice) + ".npy"), result_dict, allow_pickle=True)
            # exit()
            # pickle.dump(
            #     {"coarse": coarse.cpu().data.numpy(), "fine": sample2},
            #     open(os.path.join(args.save_path, image + "_" + str(slice) + ".pt"), "wb"),
            # )
        # vis = np.abs(sample2[0, 0] + sample2[0, 1] * 1j)
        # imageio.imsave(
        #     os.path.join(args.save_path, image + "_" + str(slice) + ".png"),
        #     vis / vis.max(),
        # )

def normalize_complex(data, eps=0.):
    mag = np.abs(data)
    mag_std = mag.std()
    return data / (mag_std + eps), mag_std

def load_data(data_path, file1, file2, batch_size, mask):
    mask = mask.squeeze()[0].cpu().numpy()
    img_prior1 = np.load(os.path.join(data_path, file1), allow_pickle=True)[()]["coil-combined-image-space"]
    img_prior2 = np.load(os.path.join(data_path, file2), allow_pickle=True)[()]["coil-combined-image-space"]
    fs_gt = img_prior1.copy()

    undersampled_kspace_1 = np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(img_prior1))) * mask
    undersampled_kspace_1[mask == 0] = 0
    undersampled_kspace_2 = np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(img_prior2))) * mask
    undersampled_kspace_2[mask == 0] = 0
    img_prior1 = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(undersampled_kspace_1)))
    img_prior2 = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(undersampled_kspace_2)))

    # normalize
    img_prior1, _ = normalize_complex(img_prior1)
    img_prior2, _ = normalize_complex(img_prior2)
    us_inp = img_prior1.copy()

    data = np.stack(
        [
            np.real(img_prior1),
            np.imag(img_prior1),
            np.real(img_prior2),
            np.imag(img_prior2),
        ]
    ).astype(np.float32)

    max_val = abs(data[:2]).max()
    data[:2] /= max_val
    max_val = abs(data[2:4]).max()
    data[2:4] /= max_val
    # regularizing over max value ensures this model works over different preprocessing schemes;
    # to not use the gt max value, selecting an appropriate averaged max value from training set leads to
    # similar performance, e.g.
    # data /= 7.21 (average max value); in general max_value is at DC and should be accessible.
    data1 = data[0] + data[1] * 1j
    data2 = data[2] + data[3] * 1j
    kspace1 = np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(data1)))
    kspace2 = np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(data2)))


    vh, vw = data.shape[-2:]
    kspace = (
        th.FloatTensor(
            np.stack(
                [np.real(kspace1), np.imag(kspace1), np.real(kspace2), np.imag(kspace2)]
            )
        )
        .cuda()
        .view(1, 4, vh, vw)
        .repeat(batch_size, 1, 1, 1)
        .float()
    )
    return kspace, fs_gt, us_inp, undersampled_kspace_1


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
