from __future__ import print_function


import numpy as np
import pdb
from multiprocessing import Pool
import functools


import numpy as np
import os
import sys
import argparse
import time
import math
from PIL import Image

import torchvision.utils as vutils
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter, GansetDataset, GansteerDataset, OnlineDataset
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, SupCEResNet
from losses import SupConLoss
import oyaml as yaml
import pbar as pbar

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def getitem(idx, root_dir, transform, class_to_idx, numcontrast=0):
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

=======
>>>>>>> 0c8c9108f080fd28b5bcef9c7e0c39e06fedb243
def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--encoding_type', type=str, default='contrastive',
                        choices=['contrastive', 'crossentropy', 'autoencoding'])
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')

    parser.add_argument('--numiter', type=int, default=130000,
                        help='save frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--showimg', action='store_true', help='display image in tensorboard')
    parser.add_argument('--resume', default='', type=str, help='whether to resume training')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='biggan',
                        choices=['biggan', 'cifar10', 'cifar100', 'imagenet100', 'imagenet100K', 'imagenet'], help='dataset')

    ## Ali: todo: this should be based on opt.encoding type and remove the default (revisit every default) and name of the model for saving
    # method
    parser.add_argument('--numcontrast', type=int, default=20,
                        help='num of workers to use')
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--walk_method', type=str, help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # specifying folders
    parser.add_argument('-d', '--data_folder', type=str,
                        default='/data/scratch-oc40/jahanian/ganclr_results/ImageNet100',
                        help='the data folder')
    parser.add_argument('-s', '--cache_folder', type=str,
                        default='/data/scratch-oc40/jahanian/ganclr_results/',
                        help='the saving folder')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = opt.data_folder
    if opt.encoding_type == 'crossentropy':
        opt.method = 'SupCE'
        opt.model_path = os.path.join(opt.cache_folder, 'SupCE/{}_models'.format(opt.dataset))
        opt.tb_path = os.path.join(opt.cache_folder, 'SupCE/{}_tensorboard'.format(opt.dataset))
    else:
        opt.model_path = os.path.join(opt.cache_folder, 'SupCon/{}_models'.format(opt.dataset))
        opt.tb_path = os.path.join(opt.cache_folder, 'SupCon/{}_tensorboard'.format(opt.dataset))

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}online_{}_{}_ncontrast.{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
            format(opt.method, opt.dataset, opt.walk_method, opt.model, opt.numcontrast, opt.learning_rate, 
            opt.weight_decay, opt.batch_size, opt.temp, opt.trial)


    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    opt.model_name = '{}_{}'.format(opt.model_name, os.path.basename(opt.data_folder))
    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
 
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'biggan' or opt.dataset == 'imagenet100' or opt.dataset == 'imagenet100K' or opt.dataset == 'imagenet':
        # or 256 as you like
        opt.img_size = 256
        opt.n_cls = 1000
    elif opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
        opt.img_size = 32

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'biggan' or opt.dataset == 'imagenet100' or opt.dataset == 'imagenet100K' or opt.dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)
    opt.mean = mean
    opt.std = std

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=int(opt.img_size*0.875), scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = OnlineDataset(root_dir=os.path.join(opt.data_folder, 'train'), transform=train_transform, numcontrast=opt.numcontrast)
    return train_dataset


def set_model(opt):
    if opt.encoding_type == 'contrastive':
        model = SupConResNet(name=opt.model, img_size=int(opt.img_size*0.875))
        criterion = SupConLoss(temperature=opt.temp)

    elif opt.encoding_type == 'crossentropy':
        model = SupCEResNet(name=opt.model, num_classes=opt.n_cls, img_size=int(opt.img_size*0.875))
        criterion = torch.nn.CrossEntropyLoss()

    elif opt.encoding_type == 'autoencoding':
        print("TODO(ali): Implement here")
        raise NotImplementedError

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt, pool):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    top1 = AverageMeter()
    end = time.time()

    ## Ali: Todo: this data loading depends on if we generate positive on the fly or load them. if loading then we have data[0] and data[1]
    ## so we need to check for opt.encoding_type (currently it's good for SupCon with gan data, i.e., positive pairs are from gan)

    # for idx, (images, labels) in enumerate(train_loader):
    #     data_time.update(time.time() - end)
    print("Start train")
    index_total = (epoch-1) * opt.numiter
    index_batch = 0


    func_pool = functools.partial(getitem,
                root_dir=train_loader.root_dir, 
                transform=train_loader.transform, 
                class_to_idx=train_loader.class_to_idx)


    while index_batch < opt.numiter:
        im_indices = [i+index_total for i in range(opt.batch_size)]
        data = pool.map(func_pool, im_indices)
        index_batch += opt.batch_size
        index_total += opt.batch_size
        pdb.set_trace()
=======
    start_ep = 0
    # if loading from ckpt: Ali todo
    #     start_ep = checkpoint['epoch'] + 1
    
    epoch_batches = 1600 // opt.batch_size
    for epoch, epoch_loader in enumerate(pbar(
        epoch_grouper(train_loader, epoch_batches),
        total=(opt.niter-start_ep)), start_ep):

        # stopping condition
        if epoch > opt.niter:
            break

        # run a train epoch of epoch_batches batches
        for step, (z_batch,) in enumerate(pbar(
            epoch_loader, total=epoch_batches), 1):
            
        # now get z_batch as if it's your data



    for idx, data in enumerate(train_loader):
>>>>>>> 0c8c9108f080fd28b5bcef9c7e0c39e06fedb243

        if len(data) == 2:
            images = data[0]
            labels = data[1]
        elif len(data) == 3:
            images = data[:2]
            labels = data[2]
        else:
            raise NotImplementedError
        
        data_time.update(time.time() - end)
        if opt.encoding_type != 'contrastive':
            # We only pick one of images
            images = images[1]
        else:
            ims = images[0]
            anchors = images[1]
            images = torch.cat([images[0].unsqueeze(1), images[1].unsqueeze(1)],
                               dim=1)
            # print('2) images shape', images.shape)

            images = images.view(-1, 3, int(opt.img_size*0.875), int(opt.img_size*0.875)).cuda(non_blocking=True)
            # print('3) images shape', images.shape)

        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        # compute loss


        if opt.encoding_type == 'contrastive':
            features = model(images)
            features = features.view(bsz, 2, -1)
            if opt.method == 'SupCon':
                loss = criterion(features, labels)
            elif opt.method == 'SimCLR':
                loss = criterion(features)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                 format(opt.method))
        elif opt.encoding_type == 'crossentropy':
            output = model(images)
            loss = criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)
        else:
            raise NotImplementedError


        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            if opt.encoding_type == 'crossentropy':
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1))
            else:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses))
            sys.stdout.flush()
    other_metrics = {}

    if opt.encoding_type == 'crossentropy':
        other_metrics['top1_acc'] = top1.avg
    else:
        if opt.showimg:
            other_metrics['image'] = [ims[:8], anchors[:8]]

    return losses.avg, other_metrics


def training_loader(truncation, batch_size, global_seed=0):
    '''
    Returns an infinite generator that runs through randomized z
    batches, forever.
    '''
    g_epoch = 1
    while True:
        z_data = biggan_networks.truncated_noise_dataset(truncation=truncation,
                                                         batch_size=10000, 
                                                         seed=g_epoch + global_seed)
        dataloader = torch.utils.data.DataLoader(
                z_data,
                shuffle=False,
                batch_size=batch_size,
                num_workers=10,
                pin_memory=True)
        for batch in dataloader:
            yield batch
        g_epoch += 1

def epoch_grouper(loader, epoch_size, num_epochs=None):
    '''
    To use with the infinite training loader: groups the training data
    batches into epochs of the given size.
    '''
    it = iter(loader)
    epoch = 0
    while True:
        chunk_it = itertools.islice(it, epoch_size)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)
        epoch += 1
        if num_epochs is not None and epoch >= num_epochs:
            return
>>>>>>> 0c8c9108f080fd28b5bcef9c7e0c39e06fedb243

def main():
    opt = parse_option()

    with open(os.path.join(opt.save_folder, 'optE.yml'), 'w') as f:
        yaml.dump(vars(opt), f, default_flow_style=False)

    # build data loader
    # opt.encoding_type tells us how to get training data
    train_loader = set_loader(opt)

    # build model and criterion
    # opt.encoding_type tells us what to put as the head; choices are:
    # contrastive -> mlp or linear
    # crossentropy -> one linear for pred_y
    # autoencoding -> one linear for pred_z and one linear for pred_y
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    init_epoch = 1
    if len(opt.resume) > 0:
        model_ckp = torch.load(opt.resume)
        init_epoch = model_ckp['epoch'] + 1
        model.load_state_dict(model_ckp['model'])
        optimizer.load_state_dict(model_ckp['optimizer'])
    pool = Pool(processes=opt.num_workers)

    for epoch in range(init_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, other_metrics = train(train_loader, model, criterion, optimizer, epoch, opt, pool)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        for metric_name, metric_value in other_metrics.items():
            if metric_name == 'image':
                images = metric_value
                anchors = images[0]
                otherims = images[1]
                bs = anchors.shape[0]
                grid_images = vutils.make_grid(
                        torch.cat((anchors, otherims)), nrow=bs)
                grid_images *= np.array(opt.std)[:, None, None]
                grid_images += np.array(opt.mean)[:, None, None]
                grid_images = (255*grid_images.cpu().numpy()).astype(np.uint8)
                grid_images = grid_images[None, :].transpose(0,2,3,1)
                logger.log_images(metric_name, grid_images, epoch)
            else:
                logger.log_value(metric_name, metric_value, epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
