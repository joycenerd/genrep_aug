from __future__ import print_function

import os
import sys
import argparse
from pathlib import Path
import time
import math
import torch
import torch.backends.cudnn as cudnn

from torchvision import transforms, datasets
import tensorboard_logger as tb_logger
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from torchnet.meter import mAPMeter
from util import set_optimizer
from util import VOCDetectionDataset
from networks.resnet_big import SupConResNet, LinearClassifier, SupCEResNet

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=64,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.06,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='biggan',
                        choices=['biggan', 'cifar10', 'cifar100', 'imagenet100', 'imagenet100K', 'imagenet'], help='dataset')
    parser.add_argument('-s', '--cache_folder', type=str,
                        default='.',
                        help='the saving folder')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    # specifying folders
    parser.add_argument('-d', '--data_folder', type=str, default='./data',
                        help='the folder of the dataset you want to evaluate')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_path = os.path.join(opt.cache_folder, 'tensorboards_test')

    ckpt_red = opt.ckpt # Update this function if you need to reduce the size of the name
    opt.tb_folder = os.path.join(opt.tb_path, 'lr.{}'.format(opt.learning_rate)).strip()
    
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

   
    opt.img_size = 128
    opt.n_cls = 25

    return opt


def set_loader(opt):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.CenterCrop(int(opt.img_size*0.875)),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'val.x1'),
                                        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return  val_loader


def set_model(opt,ckpt_file):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls, img_size=int(opt.img_size*0.875))
    criterion = torch.nn.CrossEntropyLoss()

    ckpt = torch.load(ckpt_file, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        # ipdb.set_trace()
        model.load_state_dict(state_dict,  strict=False)

    return model, criterion


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    meanAPmetric = mAPMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1))


    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()
    print(opt)

    # build data loader
    val_loader = set_loader(opt)

    # build model and criterion
    ckpt_folder="./logger/SupCE/imagenet100_models/SupCE_imagenet100_steer_resnet50_ncontrast.1_ratiodata.1.0_lr_0.03_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine_data"
    ep=10
    while ep<=300:
        ckpt_file=Path(ckpt_folder).joinpath(f"ckpt_epoch_{ep}.pth")
        model, criterion = set_model(opt,ckpt_file)

        loss, val_acc = validate(val_loader, model, criterion, opt)
        print('Epoch {} acc: {:.2f}'.format(ep,val_acc))
        
        ep+=10

if __name__ == '__main__':
    main()