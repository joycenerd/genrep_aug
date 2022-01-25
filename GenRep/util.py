from __future__ import print_function

import pdb
import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from pytorch_pretrained_biggan import (
    BigGAN,
    truncated_noise_sample,
    one_hot_from_int
)
from torchvision import datasets
import random
import glob
import os 
import xml.etree.ElementTree as ET
from PIL import Image
import json
from scipy.stats import truncnorm
import random
import pickle

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
        img.append(Image.fromarray(out_array))
    return img

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, grad_update, class_count, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'grad_updates': grad_update,
        'class_count': class_count,
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

class Caltech101(datasets.Caltech101):
    # See https://github.com/pytorch/vision/blob/master/torchvision/datasets/caltech.py
    def __init__(self,
                 root,
                 split='train',
                 target_type='category',
                 transform=None,
                 target_transform=None,
                 download=False):
        super(Caltech101, self).__init__(root, target_type, transform, target_transform, download)
        self.index = []
        self.split = split
        self.y = []
        self.random_seed = random.Random(0)
        for (i, c) in enumerate(self.categories):
            samples = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            index_ims = list(range(1, len(samples)+1))
            self.random_seed.shuffle(index_ims)
            # https://arxiv.org/pdf/2002.05709.pdf
            if self.split == 'train':
                index_ims = index_ims[:30]
            else:
                index_ims = index_ims[30:]
            n = len(index_ims)
            self.index.extend(index_ims)
            self.y.extend(n * [i])

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(os.path.join(self.root,
                                      "101_ObjectCategories",
                                      self.categories[self.y[index]],
                                      "image_{:04d}.jpg".format(self.index[index]))).convert('RGB')

        target: Any = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(os.path.join(self.root,
                                                     "Annotations",
                                                     self.annotation_categories[self.y[index]],
                                                     "annotation_{:04d}.mat".format(self.index[index])))
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target_onehot = np.zeros(101)
        target_onehot[target] = 1

        return img, target_onehot

class VOCDetectionDataset(datasets.VOCDetection):
    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None):

        super(VOCDetectionDataset, self).__init__(root, 
                                                  year, 
                                                  image_set, 
                                                  download, 
                                                  transform, 
                                                  target_transform, 
                                                  transforms)
        self.class_names = [
                'person',
                'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
                'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        self.class_dict = {name: i for i, name in enumerate(self.class_names)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)


        objects = list(set([self.class_dict[obj['name']] for obj in target['annotation']['object']]))
        objects_one_hot = np.zeros((len(self.class_names)))
        objects_one_hot[np.array(objects)] = 1

        return img, objects_one_hot

class OnlineGansetDataset(Dataset):
    """The idea is to load the anchor image and its neighbor"""

    def __init__(self, root_dir, neighbor_std=1.0, transform=None, truncation=1.0, dim_z=128,
                 seed=None, walktype='gaussian', uniformb=None, num_samples=130000, size_biggan=256, device_id=0):
        """
        Args:
            neighbor_std: std in the z-space
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            walktype: whether we are moving in a gaussian ball or a uniform ball
        """
        super(OnlineGansetDataset, self).__init__()
        self.neighbor_std = neighbor_std
        self.uniformb = uniformb
        self.root_dir = root_dir
        self.transform = transform
        self.walktype = walktype
        self.classes, self.class_to_idx = self.find_classes(self.root_dir)
        self.dim_z = dim_z
        self.truncation = truncation
        self.random_state = None if seed is None else np.random.RandomState(seed) 

        model_name = 'biggan-deep-%s' % size_biggan
        self.device_id = device_id
        self.model = BigGAN.from_pretrained(model_name).cuda(self.device_id)
        self.num_samples = num_samples
        with open('utils/imagenet_class_index.json', 'rb') as fid:
            self.imagenet_class_index_dict = json.load(fid)
        list100 = os.listdir('/data/scratch-oc40/jahanian/ganclr_results/ImageNet100/train')
        self.class_indices_interest = []

        imagenet_class_index_keys = self.imagenet_class_index_dict.keys()
        for key in imagenet_class_index_keys:
            if self.imagenet_class_index_dict[key][0] in list100:
                self.class_indices_interest.append(key)


        # get list of anchor images

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def gen_images(self, z, class_ind, use_grad=False):
        if class_ind.ndim == 3:
            class_ind = class_ind.squeeze(1)
            z = z.squeeze(1)

        class_vector = class_ind.cuda(self.device_id)
        noise_vector = z.cuda(self.device_id)
        if not use_grad:
            with torch.no_grad():
                output = self.model(noise_vector, class_vector, self.truncation)
        else:
            output = self.model(noise_vector, class_vector, self.truncation)
        output = output.cpu()
        output = convert_to_images(output)
        # TODO: clipping and other ops to convert to image. Unsure if differentiable
        return output

    def gen_images_transform(self, z, class_ind, use_grad=False):
        images = self.gen_images(z, class_ind, use_grad)
        images_orig = images
        if self.transform:
            images = [self.transform(image) for image in images]
        images = torch.cat([im.unsqueeze(0) for im in images], 0)
        return images, images_orig

    def sample_class(self):
        indices = random.choices(self.class_indices_interest, k=1)
        return indices

    def sample_noise(self):
        values = truncnorm.rvs(-2, 2, size=(1, self.dim_z), random_state=self.random_state).astype(np.float32)
        return values

    def sample_neighbors(self, values):
        if self.walktype == 'gaussian':
            neighbors = np.random.normal(0, self.neighbor_std, size=(1, self.dim_z)).astype(np.float32)
        else:
            raise NotImplementedError
        return values + neighbors
            


        


    def __len__(self):
        
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = int(self.sample_class()[0])

        one_hot_index = one_hot_from_int(label)
        w = self.sample_noise()
        dw = self.sample_neighbors(w)



        return w, dw, one_hot_index, label


class OnlineDataset():
    """The idea is to load the anchor image and its neighbor"""

    def __init__(self, root_dir, neighbor_std=1.0, transform=None, walktype='gaussian', uniformb=None, numcontrast=5):
        """
        Args:
            neighbor_std: std in the z-space
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            walktype: whether we are moving in a gaussian ball or a uniform ball
        """
        self.numcontrast = numcontrast
        self.neighbor_std = neighbor_std
        self.uniformb = uniformb
        self.root_dir = root_dir
        self.transform = transform
        self.walktype = walktype
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)

    
    def _find_classes(self, root_dir):
        classes = glob.glob('{}/*'.format(root_dir))
        classes = [x.split('/')[-1] for x in classes]
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        return classes, class_to_idx


    def getitem(idx, root_dir=None, transform=None, class_to_idx={}, numcontrast=0):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = '{}/{}_anchor'.format(root_dir, idx)

        if numcontrast > 0:
            img_name_neighbor = img_name.replace('anchor','neighbor')
        else:
            img_name_neighbor = img_name
       
        while not os.path.isfile(img_name) or not os.path.isfile(img_name_neighbor):
            print("Image {} missing ".format(img_name))
            time.sleep(2)


        image = Image.open(img_name)
        image_neighbor = Image.open(img_name_neighbor)
        label = img_name.split('/')[-2]
        label = class_to_idx[label]
        if transform:
            image = transform(image)
            image_neighbor = transform(image_neighbor)

        return image, image_neighbor, label


class GansetDataset(Dataset):
    """The idea is to load the anchor image and its neighbor"""

    def __init__(self, root_dir, neighbor_std=1.0, transform=None, walktype='gaussian', uniformb=None, numcontrast=5, method=None, ratio_data=1):
        """
        Args:
            neighbor_std: std in the z-space
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            walktype: whether we are moving in a gaussian ball or a uniform ball
        """
        super(GansetDataset, self).__init__()
        self.numcontrast = numcontrast
        self.neighbor_std = neighbor_std
        self.uniformb = uniformb
        self.root_dir = root_dir
        self.transform = transform
        self.walktype = walktype
        self.z_dict = dict()
        self.method = method
        self.ratiodata = ratio_data
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)
        self.imglist = []
        
        # get list of anchor images, first check if we have the list in a txt file ow list the images
        extra_rootdir = self.root_dir.replace('indep_20_samples', 'indep_1_samples')
        imgList_filename = os.path.join(extra_rootdir.replace('/train',''), 'ratiodata{}_imgList.txt'.format(self.ratiodata))
        if os.path.isfile(imgList_filename):
            print('Listing images by reading from ', imgList_filename)
            with open(imgList_filename, 'r') as fid:
                self.imglist = fid.readlines()
            self.imglist = [x.rstrip() for x in self.imglist]
        else:
            print('Listing images...')
            self.imglist = glob.glob(os.path.join(extra_rootdir, '*/*_anchor.png'))
            print('Listed')
            if self.ratiodata == 1:
                max_per_class = 1300
            else:
                max_per_class = int(1300 * self.ratiodata)
            # maks sure we only work on 1300 samples per class (for consistency with imagenet100)
            indices = [int(x.split('sample')[-1].split('_')[0]) for x in self.imglist]
            self.imglist = [imname for imname, ind in zip(self.imglist, indices) if ind < max_per_class]

        if self.ratiodata < 1.:
            # Repeat the dataset to compensate for the lower number of images
            print('Length: {}, will repeat the dataset to compensate for the lower number of images...'.format(len(self.imglist)))
            self.imglist = self.imglist * int(1/self.ratiodata)
        self.dir_size = len(self.imglist)
        print('Length: {}'.format(self.dir_size))

    def _find_classes(self, root_dir):
        classes = glob.glob('{}/*'.format(root_dir))
        print(root_dir)
        classes = [x.split('/')[-1] for x in classes]
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        if self.method == 'SupInv' or self.method == 'UnsupInv':
            # append the z_dataset to the dict:
            for classname in classes:
                with open(os.path.join(self.root_dir, classname, 'z_dataset.pkl'), 'rb') as fid:
                    z_dict = pickle.load(fid)
                self.z_dict[classname] = z_dict
        
        return classes, class_to_idx

    def __len__(self):
        
        return self.dir_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.imglist[idx]
        image = Image.open(img_name)

        if self.numcontrast > 0:
            neighbor = random.randint(1,self.numcontrast)
            img_name_neighbor = self.imglist[idx].replace('anchor','{:.1f}_{}'.format(self.neighbor_std, str(neighbor)))
            if neighbor > 1:
                img_name_neighbor = img_name_neighbor.replace('indep_1_samples', 'indep_20_samples')
            if not os.path.isfile(img_name_neighbor):
                img_name_neighbor = self.imglist[idx].replace('anchor','neighbor_{}'.format( str(neighbor-1)))
        else:
            img_name_neighbor = img_name


        image_neighbor = Image.open(img_name_neighbor)
        label_class = self.imglist[idx].split('/')[-2]
        label = self.class_to_idx[label_class]
        if self.transform:
            oldim = image
            oldimneighbor = image_neighbor
            image = self.transform(image)
            image_neighbor = self.transform(image_neighbor)
            oldim.close()
            oldimneighbor.close()

        z_vect = []
        if self.method == 'SupInv' or self.method == 'UnsupInv': # later can check for Unsupervised inverter will empty labels
            label_dict = self.imglist[idx].split('/')[-2]
            z_vect.append(self.z_dict[label_dict][os.path.basename(img_name)][0]) 
            z_vect.append(self.z_dict[label_dict][os.path.basename(img_name_neighbor)][0])
           # z = np.random.normal(size=128).astype(np.float32)
            # z_vect.append(z)
            # z_vect.append(z)
            return image, image_neighbor, label, label_class, z_vect
        else:
            return image, image_neighbor, label, label_class



class GansteerDataset(Dataset):
    """The idea is to load the negative-alpha image and its neighbor (positive-alpha)"""

    def __init__(self, root_dir, transform=None, numcontrast=5, method=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print("Creating dataset: ", root_dir)
        super(GansteerDataset, self).__init__()
        print("Done")
        self.numcontrast = numcontrast
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)
        
        # get list of nalpha images
        self.imglist = glob.glob(os.path.join(self.root_dir, '*/*_anchor.png'))
        print("Loading data...")
        # Make sure there are at most 1300 images per class
        #indices = [int(x.split('sample')[1].split('_')[0]) for x in self.imglist]
        #self.imglist = [imname for imname, ind in zip(self.imglist, indices) if ind < 1300]
        self.dir_size = len(self.imglist)
        print('Length: {}'.format(self.dir_size))
        
    def _find_classes(self, root_dir):
        classes = glob.glob('{}/*'.format(root_dir))
        classes = [x.split('/')[-1] for x in classes]
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        return classes, class_to_idx        

    def __len__(self):
        
        return self.dir_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.imglist[idx]
        image = Image.open(img_name)
        if self.numcontrast > 0:
            neighbor = random.randint(1,self.numcontrast)
            img_name_neighbor = self.imglist[idx].replace('anchor','neighbor_{}'.format(str(neighbor-1)))
            #if neighbor > 1:
            #    img_name_neighbor = img_name_neighbor.replace('indep_1_samples', 'indep_20_samples')
            #if not os.path.isfile(img_name_neighbor):
            #    img_name_neighbor = self.imglist[idx].replace('anchor','neighbor_{}'.format( str(neighbor-1)))
        else:
            img_name_neighbor = img_name
       # print('anchor, neighbor', img_name, img_name_neighbor)
        image_neighbor = Image.open(img_name_neighbor)
        label = self.imglist[idx].split('/')[-2]
        # with open('./utils/imagenet_class_index.json', 'rb') as fid:
        #     imagenet_class_index_dict = json.load(fid)
        # for key, value in imagenet_class_index_dict.items():
        #     if value[0] == label:
        #         label = key
        #         break
        label = self.class_to_idx[label]
        if self.transform:
            image = self.transform(image)
            image_neighbor = self.transform(image_neighbor)
        return image, image_neighbor, label    
    
    
    
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = self.imglist[idx]
#         # print('anchor image:', img_name)
#         image = Image.open(img_name)

# #         ## for composite
# #         neighbor_names = ['palpha', 'nalpha']
# #         neighbor_choice = np.random.choice(neighbor_names)
# #         img_name_neighbor = self.imglist[idx].replace('anchor',neighbor_choice)

#         # for composite2
#         neighbor_names = ['palpha_R', 'palpha_G', 'palpha_B', 'palpha_W', 'palpha_D', 'nalpha_R', 'nalpha_G', 'nalpha_B', 'nalpha_W', 'nalpha_D']
#         random.shuffle(neighbor_names)
#         for i in range(len(neighbor_names)):            
#             neighbor_choice = neighbor_names[i]
#             img_name_neighbor = self.imglist[idx].replace('anchor',neighbor_choice)
#             if os.path.isfile(img_name_neighbor):
#                 break
                

# #         ## for zoom-rgb-max-alt
# #         coin = np.random.rand()
# #         if coin < 0.5:
# #             img_name_neighbor = self.imglist[idx].replace('anchor','palpha')

# #         if (int(os.path.basename(self.imglist[idx]).split('_')[4].replace('sample', '')) % 2 ==0):
# #             img_name_neighbor = self.imglist[idx].replace('anchor','palpha')
# #         else:
# #             img_name_neighbor = self.imglist[idx].replace('anchor','nalpha')

# #         # lets randomly switch to a steered color neighbor:
# #         coin = np.random.rand()
# #         # if coin < 0.66 and coin >= 0.33:
# #         #     img_name_neighbor.replace('biggan256tr1-png_steer_rot3d_100', 'biggan256tr1-png_steer_color_100')
# #         # elif coin < 0.33:
# #         #     img_name_neighbor.replace('biggan256tr1-png_steer_rot3d_100', 'biggan256tr1-png_steer_zoom_100')
# #         if coin < 0.5:
# #             color_list = ['W_sample', 'R_sample', 'G_sample', 'B_sample']
# #             color_choice = np.random.choice(color_list)
# #             img_name_neighbor = img_name_neighbor.replace('biggan256tr1-png_steer_zoom_100', 'biggan256tr1-png_steer_color_100')
# #             img_name_neighbor = img_name_neighbor.replace('sample', color_choice)

# #             if color_choice in ['R_sample', 'G_sample', 'B_sample'] and 'nalpha' in img_name_neighbor:
# #                 img_name_neighbor = img_name_neighbor.replace('nalpha', 'palpha')

# #         print('anchor, neighbor: ', img_name, img_name_neighbor)
# #         # print(os.path.exists(img_name_neighbor))
#         image_neighbor = Image.open(img_name_neighbor)
#         label = self.imglist[idx].split('/')[-2]
#         # with open('./utils/imagenet_class_index.json', 'rb') as fid:
#         #     imagenet_class_index_dict = json.load(fid)
#         # for key, value in imagenet_class_index_dict.items():
#         #     if value[0] == label:
#         #         label = key
#         #         break
#         label = self.class_to_idx[label]
#         if self.transform:
#             image = self.transform(image)
#             image_neighbor = self.transform(image_neighbor)

#         return image, image_neighbor, label


class MixDataset(Dataset):
    """The idea is to randomly load data from bigbigan100"""

    def __init__(self, root_dir, mix_ratio, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            mix_ratio: how much from fake images (from bigbigan100)
        """
        super(MixDataset, self).__init__()
        self.root_dir = root_dir
        self.root_dir_fake = '/data/vision/phillipi/ganclr/datasets/bigbigan100/train'
        self.transform = transform
        self.mix_ratio = mix_ratio
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)

        # get list of real images
        print("Listing real images...")
        self.imglist = glob.glob(os.path.join(self.root_dir, '*/*.JPEG')) #n01558993_3482.JPEG
        self.dir_size = len(self.imglist)
        print('Length of real images list: {}'.format(self.dir_size))
        
        # get list of anchor of fake images
        print("Listing fake images...")
        self.imglist_fake = glob.glob(os.path.join(self.root_dir_fake, '*/*_anchor.png'))
        self.dir_fake_size = len(self.imglist_fake)
        print('Length of fake iamges list: {}'.format(self.dir_fake_size))

    def _find_classes(self, root_dir):
        classes = glob.glob('{}/*'.format(root_dir))
        print(root_dir)
        classes = [x.split('/')[-1] for x in classes]
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

        return classes, class_to_idx

    def __len__(self):
        
        return self.dir_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.imglist[idx].split('/')[-2]
        label = self.class_to_idx[label]

        if np.random.uniform() < self.mix_ratio:
            img_name = self.imglist_fake[idx]
        else:
            img_name = self.imglist[idx]
#         print('Loading img_name:', img_name)
        
        image = Image.open(img_name)
        #if len(np.array(image).shape) < 3:
            # print(img_name, np.array(image).shape)
        image = image.convert('RGB')


        if self.transform:
            image = self.transform(image)
        
        return [image, label]
