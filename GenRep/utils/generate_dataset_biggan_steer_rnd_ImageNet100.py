''' 
code adapted from lucy's code
you will need to pip install pytorch_pretrained_biggan, see https://github.com/huggingface/pytorch-pretrained-BigGAN
'''

import torch
from pytorch_pretrained_biggan import (
    BigGAN,
    truncated_noise_sample,
    one_hot_from_int
)
import PIL.Image
import numpy as np
import os
import argparse
from tqdm import tqdm
import json
from scipy.stats import truncnorm

def convert_to_images(obj):
    """ Convert an output tensor from BigGAN in a list of images.
    """
    # need to fix import, see: https://github.com/huggingface/pytorch-pretrained-BigGAN/pull/14/commits/68a7446951f0b9400ebc7baf466ccc48cdf1b14c
    if not isinstance(obj, np.ndarray):
        obj = obj.detach().numpy()
    obj = obj.transpose((0, 2, 3, 1))
    obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)
    img = []
    for i, out in enumerate(obj):
        out_array = np.asarray(np.uint8(out), dtype=np.uint8)
        img.append(PIL.Image.fromarray(out_array))
    return img

def truncated_noise_sample_neighbors(batch_size=1, dim_z=128, truncation=1., seed=None):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    norm_1 = np.random.normal(0., 1.1, size=(batch_size, dim_z)).astype(np.float32)
    norm_2 = np.random.normal(0., 1.2, size=(batch_size, dim_z)).astype(np.float32)
    norm_3 = np.random.normal(0., 1.3, size=(batch_size, dim_z)).astype(np.float32)
    norm_4 = np.random.normal(0., 1.4, size=(batch_size, dim_z)).astype(np.float32)

    zs = truncation * values

    return [zs, zs + norm_1, zs + norm_2, zs + norm_3, zs + norm_4]

def sample(opt):
    output_path = (os.path.join(opt.out_dir, 'biggan%dtr%d-%s_%s' %
                   (opt.size, int(opt.truncation), opt.imformat, opt.desc)))
    partition = opt.partition
    # start_seed, nimg = constants.get_seed_nimg(partition)
    start_seed = opt.start_seed
    nimg = opt.num_imgs
    model_name = 'biggan-deep-%s' % opt.size
    truncation = opt.truncation
    imformat = opt.imformat
    batch_size = opt.batch_size
    with open('./imagenet_class_index.json', 'rb') as fid:
        imagenet_class_index_dict = json.load(fid)
    imagenet_class_index_keys = imagenet_class_index_dict.keys()
    print('Loading the model ...')
    model = BigGAN.from_pretrained(model_name).cuda()
    
    list100 = os.listdir('/data/scratch-oc40/jahanian/ganclr_results/ImageNet100/train')

    for key in imagenet_class_index_keys:
        if imagenet_class_index_dict[key][0] not in list100:
            continue
        class_dir_name = os.path.join(output_path, partition, imagenet_class_index_dict[key][0])
        os.makedirs(class_dir_name, exist_ok=True)
        idx = int(key)
        print('Generating images for class {}'.format(idx))
        class_vector = one_hot_from_int(idx, batch_size=nimg)
        seed = start_seed + idx
        noise_vector_neighbors = truncated_noise_sample_neighbors(truncation=truncation,
                                                        batch_size=nimg,
                                                        seed=seed)
        class_vector = torch.from_numpy(class_vector).cuda()
        for ii in range(len(noise_vector_neighbors)):
            noise_vector = noise_vector_neighbors[ii]
            noise_vector = torch.from_numpy(noise_vector).cuda()
            for batch_start in range(0, nimg, batch_size):
                s = slice(batch_start, min(nimg, batch_start + batch_size))

                with torch.no_grad():
                    output = model(noise_vector[s], class_vector[s], truncation)
                output = output.cpu()
                ims = convert_to_images(output)
                for i, im in enumerate(ims):
                    if ii == 0:
                        im_name = 'seed%04d_sample%05d_anchor.%s' % (seed, batch_start+i, imformat)
                    elif ii == 1:
                        im_name = 'seed%04d_sample%05d_1.1.%s' % (seed, batch_start+i, imformat)
                    elif ii == 2:
                        im_name = 'seed%04d_sample%05d_1.2.%s' % (seed, batch_start+i, imformat)
                    elif ii == 3:
                        im_name = 'seed%04d_sample%05d_1.3.%s' % (seed, batch_start+i, imformat)
                    elif ii == 4:
                        im_name = 'seed%04d_sample%05d_1.4.%s' % (seed, batch_start+i, imformat)
                    im.save(os.path.join(class_dir_name, im_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Sample from biggan")
    parser.add_argument('--out_dir', default='/data/scratch/jahanian/ganclr_results_2/', type=str)
    parser.add_argument('--partition', default='val', type=str)
    parser.add_argument('--truncation', default=1.0, type=float)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--imformat', default='png', type=str)
    parser.add_argument('--num_imgs', default=50, type=int, help='num imgs per class')
    parser.add_argument('--start_seed', default=500, type=int)
    parser.add_argument('--desc', default='steer_rnd_100', type=str, help='this will be the tag of this specfic dataset, added to the end of the dataset name')
    opt = parser.parse_args()
    sample(opt)
