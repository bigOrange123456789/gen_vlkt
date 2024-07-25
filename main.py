import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate_hoi
from models import build_model
import os


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_clip', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of stage1 decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--hoi', action='store_true',
                        help="Train for HOI if the flag is provided")
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--with_mimic', action='store_true',
                        help="Use clip feature mimic")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=2.5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")
    parser.add_argument('--set_cost_hoi', default=1, type=float,
                        help="Hoi class coefficient")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=2, type=float)
    parser.add_argument('--hoi_loss_coef', default=2, type=float)
    parser.add_argument('--mimic_loss_coef', default=20, type=float)
    parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # hoi eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--json_file', default='results.json', type=str)

    # clip
    parser.add_argument('--ft_clip_with_small_lr', action='store_true',
                        help='Use smaller learning rate to finetune clip weights')
    parser.add_argument('--with_clip_label', action='store_true', help='Use clip to classify HOI')
    parser.add_argument('--early_stop_mimic', action='store_true', help='stop mimic after step')
    parser.add_argument('--with_obj_clip_label', action='store_true', help='Use clip to classify object')
    parser.add_argument('--clip_model', default='ViT-B/32',
                        help='clip pretrained model path')
    parser.add_argument('--fix_clip', action='store_true', help='')
    parser.add_argument('--clip_embed_dim', default=512, type=int)

    # zero shot type
    parser.add_argument('--zero_shot_type', default='default',
                        help='default, rare_first, non_rare_first, unseen_object, unseen_verb')
    parser.add_argument('--del_unseen', action='store_true', help='')

    return parser


def main(args):
    utils.init_distributed_mode(args) #args: Namespace(lr=0.0001, lr_backbone=1e-05, ...)
    print("git:\n  {}\n".format(utils.get_sha())) #应该是要判断是否为gitbub上的最新版
    
    if args.frozen_weights is not None: # False
        assert args.masks, "Frozen training is meant for segmentation only" #冻结训练仅用于细分
    #print(args)

    device = torch.device(args.device)#[args.device: cuda, device: cuda]

    # fix the seed for reproducibility # 固定种子以提高可重复性
    seed = args.seed + utils.get_rank()#seed: 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    print("开始构建模型...")
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    print("完成构建模型...\n")
    if False: # 这里可以输出模型的结构
        print('****************')
        print(model)
        print('****************')

    print("开始初始化optimizer...")
    model_without_ddp = model
    if args.distributed: # distributed=False # 不进行分布式计算
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print('number of params:', n_parameters) # number of params: 41877603 # 参数总量:4千万

    for name, p in model.named_parameters(): # 遍历模型的每一层
        if 'eval_visual_projection' in name: # 似乎没有找到这样的层
            p.requires_grad = False
            print("lzc flag: main.py: (1) if 'eval_visual_projection' in name")

    if args.fix_clip: # fix_clip: False
        for name, p in model.named_parameters():
            if 'obj_visual_projection' in name or 'visual_projection' in name:
                p.requires_grad = False
                print("lzc flag: main.py: (2) if 'eval_visual_projection' in name")

    if args.ft_clip_with_small_lr: #ft_clip_with_small_lr: True
        if args.with_obj_clip_label and args.with_clip_label: # args.with_obj_clip_label=True  with_clip_label=True
            param_dicts = [ # 感觉这里应该是获取了模型中全部参数
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'visual_projection' not in n and 'obj_visual_projection' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               ('visual_projection' in n or 'obj_visual_projection' in n) and p.requires_grad],
                    "lr": args.lr_clip,
                },
            ]
        elif args.with_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'visual_projection' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               'visual_projection' in n and p.requires_grad],
                    "lr": args.lr_clip,
                },
            ]
        elif args.with_obj_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'obj_visual_projection' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               'obj_visual_projection' in n and p.requires_grad],
                    "lr": args.lr_clip,
                },
            ]
        else:
            raise

    else:
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, # lr=0.0001
                                  weight_decay=args.weight_decay) # weight_decay=0.0001
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop) # args.lr_drop=60
    # lr_scheduler: <StepLR> # 这对象是啥，有什么作用？
    print("完成初始化optimizer...\n")

    print("开始处理数据集...")
    print("1.开始加载数据集标注...")
    dataset_train = build_dataset(image_set='train', args=args) # dataset_train: <datasets.vcoco.VCOCO object>
    dataset_val = build_dataset(image_set='val', args=args)     # dataset_val:   <datasets.vcoco.VCOCO object>

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else: #不使用分布式计算
        sampler_train = torch.utils.data.RandomSampler(dataset_train) # 随机采样器
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) # 序列采样器

    batch_sampler_train = torch.utils.data.BatchSampler( # 批处理采样器
        sampler_train, args.batch_size, drop_last=True) # batch_size=2

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # dataset_train: <datasets.vcoco.VCOCO object>
    # utils.collate_fn: <function collate_fn>
    # args.num_workers: 2
    # data_loader_train: <torch.utils.data.dataloader.DataLoader>
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False,
                                  collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # dataset_val: <datasets.vcoco.VCOCO>
    # args.batch_size: 2
    # sampler_val: <torch.utils.data.sampler.SequentialSampler>
    exit(0)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    elif args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if args.eval:
            model_without_ddp.load_state_dict(checkpoint['model'])
        else:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    if args.eval:
        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val,
                                  args.subject_category_id, device, args)
        return

    print("Start training")
    start_time = time.time()
    best_performance = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if epoch == args.epochs - 1:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        if epoch < args.lr_drop and epoch % 5 != 0:  ## eval every 5 epoch before lr_drop
            continue
        elif epoch >= args.lr_drop and epoch % 2 == 0:  ## eval every 2 epoch after lr_drop
            continue

        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val,
                                  args.subject_category_id, device, args)
        if args.dataset_file == 'hico':
            performance = test_stats['mAP']
        elif args.dataset_file == 'vcoco':
            performance = test_stats['mAP_all']
        elif args.dataset_file == 'hoia':
            performance = test_stats['mAP']

        if performance > best_performance:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

            best_performance = performance

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GEN VLKT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
