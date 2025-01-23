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

# mask = th.from_numpy(np.load("mask.pt.npy")[None, None, ...]).cuda().float()
# # mask = F.pad(mask, (0, 0, 56, 56))
# mask = mask[:, :, 8:-8, :]
# mask = th.cat([mask, mask], 1)

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
    # run without cropping
    # ht = mask.shape[-2]
    # if ht < 128:
    #     padding_needed = (128 - ht) // 2
    #     mask = F.pad(mask, (0, 0, padding_needed, padding_needed))
    # wd = mask.shape[-1]
    # if wd < 512:
    #     padding_needed = (512 - wd) // 2
    #     mask = F.pad(mask, (padding_needed, padding_needed, 0, 0))
    
    # mask = F.center_crop(mask, (128, 512))
    mask = th.cat([mask, mask], 1)
    return mask


def main():
    args = create_argparser().parse_args()
    print("USING DDIM:", args.use_ddim)

    images = os.listdir(args.data_path)
    images = [image[:-5] for image in images]
    # run only on t2 images
    images = [image for image in images if 't1' in image.lower()]
    images = list(set(images))
    images.sort()
    images = images[:9*5]
    print(images[:10])
    print('Found', len(images), 'files')
    images.sort()
    

    acc_factor = 4 if args.data_path.endswith('acc4') else (8 if args.data_path.endswith('acc8') else 10)

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
                file_name1 = image + "_" + str(i % (8 if 't1' in image.lower() else 3)) + ".pt"
                # file_name2 = image + "_" + str((i + 1) % (8 if 't1' in image.lower() else 3)) + ".pt"
                file_name2 = file_name1
                kspace = load_data(args.data_path, file_name1, file_name2, args.batch_size)
                # save for refining
                if i == slice:
                    input = kspace[[0]]
                # logger.log("sampling...")
                samples = []
                for _ in range(2):
                    model_kwargs = {}
                    sample = diffusion.p_sample_loop(
                        model,
                        (args.batch_size, 4, 144, 512),
                        th.randn_like(kspace),
                        # mask,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                        progress=True
                    )[-1]
                    print("sample", sample.shape)
                    samples.append(sample)
                samples = th.cat(samples)
                print(samples.shape)
                np_samples = samples.detach().cpu().numpy()
                np.save('samples.npy', np_samples)
                exit()
                coarse.append(samples.contiguous())

            coarse = th.stack(coarse)
            print(coarse.shape)
            aggregate = []
            for k in range(2):
                print(coarse[k, [2, 3]].mean(0).shape)
                print(coarse[k + 1, [0, 1]].mean(0).shape)
                aggregate.append(
                    (coarse[k, [2, 3]].mean(0) + coarse[k + 1, [0, 1]].mean(0)).view(
                        1, 2, 116, 384
                    )
                    / 2
                )
            aggregate = th.cat(aggregate, 1)
            print(aggregate.shape)

            # sample2 = diffusion_two.p_sample_loop_condition(
            #     model,
            #     (1, 4, 116, 384),
            #     input,
            #     mask,
            #     noise=aggregate.float(),
            #     clip_denoised=args.clip_denoised,
            #     model_kwargs={},
            #     refine=True,
            #     progress=True
            # )
            # sample2 = sample2[-1].cpu().data.numpy()
            # pickle.dump(
            #     {"coarse": coarse.cpu().data.numpy(), "fine": sample2},
            #     open(os.path.join(args.save_path, image + "_" + str(slice) + ".pt"), "wb"),
            # )
        # vis = np.abs(sample2[0, 0] + sample2[0, 1] * 1j)
        # imageio.imsave(
        #     os.path.join(args.save_path, image + "_" + str(slice) + ".png"),
        #     vis / vis.max(),
        # )


def load_data(data_path, file1, file2, batch_size):
    # load two slices
    img_prior1 = pickle.load(open(os.path.join(data_path, file1), "rb"))["img"][0]
    # img_prior1 = np.sqrt(np.sum(img_prior1**2, axis=0)) # multiple coils
    img_prior2 = pickle.load(open(os.path.join(data_path, file2), "rb"))["img"][0]
    # img_prior2 = np.sqrt(np.sum(img_prior2**2, axis=0)) # multiple coils

    # print(img_prior1.shape, img_prior2.shape)

    # print("loading", file1, file2)
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

    # run without cropping
    # ht = kspace.shape[-2]
    # if ht < 128:
    #     padding_needed = (128 - ht) // 2
    #     kspace = F.pad(kspace, (0, 0, padding_needed, padding_needed))
    # wd = kspace.shape[-1]
    # if wd < 512:
    #     padding_needed = (512 - wd) // 2
    #     kspace = F.pad(kspace, (padding_needed, padding_needed, 0, 0))

    # kspace = F.center_crop(kspace, (128, 512))
    return kspace


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=5,
        use_ddim=False,
        model_path="",
        data_path="",
        save_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
