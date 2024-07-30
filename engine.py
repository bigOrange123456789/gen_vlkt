import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import util.misc as utils
from datasets.hico_eval_triplet import HICOEvaluator
from datasets.vcoco_eval import VCOCOEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0): # 这个函数应该每个epoch被调用1次
    model.train()
    criterion.train() #train函数的作用不是进行训练、而是将模型切换为训练模式 # 还没有弄清楚criterion是什么
    metric_logger = utils.MetricLogger(delimiter="  ") # delimiter用于指定指标之间的分隔符
    # metric_logger函数，可称为度量记录器，可自定义。
    # 该函数是为了统计各项数据，通过调用来使用或显示各项指标，通过具体项目自定义的函数。
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # 使用metric_logger对象的add_meter方法为指标记录器添加一个名为lr的计量器
    # 计量器使用utils.SmoothedValue实例化，其中window_size指定了平滑窗口的大小,fmt指定了格式化输出的方式
    if hasattr(criterion, 'loss_labels'): # False
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    elif hasattr(criterion, 'loss_hoi_labels'): # True
        metric_logger.add_meter('hoi_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header): # 每次epoch这个循环的次数都是2
        # metric_logger.log_every(data_loader, print_freq, header): <generator object>
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets) # 执行该对象的forword函数
        # print("loss_dict:",loss_dict)
        # loss_dict: {
        # 'loss_hoi_labels': tensor(225.4516, device='cuda:0', grad_fn=<RsubBackward1>), 
        # 'hoi_class_error': tensor(85.7143, device='cuda:0'), 
        # 'loss_obj_ce': tensor(4.8103, device='cuda:0', grad_fn=<NllLoss2DBackward0>), 
        # 'obj_class_error': tensor(100., device='cuda:0'), 
        # 'loss_sub_bbox': tensor(0.4883, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_obj_bbox': tensor(0.5961, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_sub_giou': tensor(0.5452, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_obj_giou': tensor(1.0786, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_feat_mimic': tensor(0.8220, device='cuda:0', grad_fn=<MeanBackward0>), 
        # 'loss_hoi_labels_0': tensor(198.1955, device='cuda:0', grad_fn=<RsubBackward1>), 
        # 'hoi_class_error_0': tensor(100., device='cuda:0'), 
        # 'loss_obj_ce_0': tensor(4.8462, device='cuda:0', grad_fn=<NllLoss2DBackward0>), 
        # 'loss_sub_bbox_0': tensor(0.5139, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_obj_bbox_0': tensor(0.5457, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_sub_giou_0': tensor(0.5053, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_obj_giou_0': tensor(1.0609, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_feat_mimic_0': tensor(0.8143, device='cuda:0', grad_fn=<MeanBackward0>), 
        # 'loss_hoi_labels_1': tensor(255.3134, device='cuda:0', grad_fn=<RsubBackward1>), 
        # 'hoi_class_error_1': tensor(85.7143, device='cuda:0'), 
        # 'loss_obj_ce_1': tensor(4.8006, device='cuda:0', grad_fn=<NllLoss2DBackward0>), 
        # 'loss_sub_bbox_1': tensor(0.5094, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_obj_bbox_1': tensor(0.5345, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_sub_giou_1': tensor(0.6176, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_obj_giou_1': tensor(1.0357, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_feat_mimic_1': tensor(0.8157, device='cuda:0', grad_fn=<MeanBackward0>  )}
        
        weight_dict = criterion.weight_dict
        # print("weight_dict:",weight_dict)
        # weight_dict: {
        # 'loss_hoi_labels': 2, 
        # 'loss_obj_ce': 1, 
        # 'loss_sub_bbox': 2.5, 
        # 'loss_obj_bbox': 2.5, 
        # 'loss_sub_giou': 1, 
        # 'loss_obj_giou': 1, 
        # 'loss_feat_mimic': 20.0, 
        # 'loss_hoi_labels_0': 2, 
        # 'loss_obj_ce_0': 1, 
        # 'loss_sub_bbox_0': 2.5, 
        # 'loss_obj_bbox_0': 2.5, 
        # 'loss_sub_giou_0': 1, 
        # 'loss_obj_giou_0': 1, 
        # 'loss_feat_mimic_0': 20.0, 
        # 'loss_hoi_labels_1': 2, 
        # 'loss_obj_ce_1': 1, 
        # 'loss_sub_bbox_1': 2.5, 
        # 'loss_obj_bbox_1': 2.5, 
        # 'loss_sub_giou_1': 1, 
        # 'loss_obj_giou_1': 1, 
        # 'loss_feat_mimic_1': 20.0     }
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # losses: tensor(2042.8176, device='cuda:0', grad_fn=<AddBackward0>)

        # reduce losses over all GPUs for logging purposes # 减少所有GPU的损耗以用于日志记录
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # utils.reduce_dict将多个loss进行合并。分布式训练中，每个进程都会计算一个loss值。
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item() # loss_value: 1434.2313232421875

        if not math.isfinite(loss_value): # 如果loss不是有限数,则执行后面的代码并中断程序
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad() # 清空模型参数的梯度，以确保每次迭代的梯度计算都是基于当前小批量数据的，而不会受之前迭代的影响。
        losses.backward() 
        # 反向传播，计算当前梯度
        # losses: tensor(1434.2313, device='cuda:0', grad_fn=<AddBackward0>)
        print("losses:",losses)
        if max_norm > 0: # max_norm:0.1
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            # torch.nn.utils.clip_grad_norm_(parameters, max_norm)
            # 剪切参数迭代对象的梯度范数。 范数是在所有梯度上一起计算的，就像它们被连接成一个向量一样。坡度已适当修改。
            # clip_grad_norm: 对所有的梯度乘以一个clip_coef，且clip_coef<1，
            # clip_grad_norm只解决梯度爆炸问题，不解决梯度消失问题
        optimizer.step()
        # optimizer.step()根据每个参数的梯度和学习率等超参数来更新模型的参数，从而最小化损失函数。
        # 通常使用随机梯度下降（SGD）或其变体（如Adam）来更新参数。
        # 更新后的参数可以通过模型对象的.parameters()方法来获取。
        # 需要注意的是，每次调用step()方法之前，我们需要手动将每个参数的梯度清零，以避免梯度累加。这可以通过优化器的.zero_grad()方法来实现。

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # loss_value: 1434.2313232421875
        
        # loss_dict_reduced_scaled: {
        # 'loss_hoi_labels': tensor(622.0366, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_obj_ce': tensor(4.7259, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_sub_bbox': tensor(0.7508, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_obj_bbox': tensor(0.8745, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_sub_giou': tensor(0.5672, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_obj_giou': tensor(0.7860, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_feat_mimic': tensor(16.1306, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_hoi_labels_0': tensor(545.4234, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_obj_ce_0': tensor(4.8390, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_sub_bbox_0': tensor(0.8367, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_obj_bbox_0': tensor(0.9977, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_sub_giou_0': tensor(0.5355, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_obj_giou_0': tensor(1.1490, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_feat_mimic_0': tensor(16.0072, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_hoi_labels_1': tensor(803.2125, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_obj_ce_1': tensor(4.7170, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_sub_bbox_1': tensor(0.6487, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_obj_bbox_1': tensor(1.0559, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_sub_giou_1': tensor(0.4935, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_obj_giou_1': tensor(0.9696, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'loss_feat_mimic_1': tensor(16.0606, device='cuda:0', grad_fn=<MulBackward0>)     }
        
        # loss_dict_reduced_unscaled: {
        # 'loss_hoi_labels_unscaled': tensor(311.0183, device='cuda:0', grad_fn=<RsubBackward1>), 
        # 'hoi_class_error_unscaled': tensor(100., device='cuda:0'), 
        # 'loss_obj_ce_unscaled': tensor(4.7259, device='cuda:0', grad_fn=<NllLoss2DBackward0>), 
        # 'obj_class_error_unscaled': tensor(100., device='cuda:0'), 
        # 'loss_sub_bbox_unscaled': tensor(0.3003, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_obj_bbox_unscaled': tensor(0.3498, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_sub_giou_unscaled': tensor(0.5672, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_obj_giou_unscaled': tensor(0.7860, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_feat_mimic_unscaled': tensor(0.8065, device='cuda:0', grad_fn=<MeanBackward0>), 
        # 'loss_hoi_labels_0_unscaled': tensor(272.7117, device='cuda:0', grad_fn=<RsubBackward1>), 
        # 'hoi_class_error_0_unscaled': tensor(100., device='cuda:0'), 
        # 'loss_obj_ce_0_unscaled': tensor(4.8390, device='cuda:0', grad_fn=<NllLoss2DBackward0>), 
        # 'loss_sub_bbox_0_unscaled': tensor(0.3347, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_obj_bbox_0_unscaled': tensor(0.3991, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_sub_giou_0_unscaled': tensor(0.5355, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_obj_giou_0_unscaled': tensor(1.1490, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_feat_mimic_0_unscaled': tensor(0.8004, device='cuda:0', grad_fn=<MeanBackward0>), 
        # 'loss_hoi_labels_1_unscaled': tensor(401.6062, device='cuda:0', grad_fn=<RsubBackward1>), 
        # 'hoi_class_error_1_unscaled': tensor(100., device='cuda:0'), 
        # 'loss_obj_ce_1_unscaled': tensor(4.7170, device='cuda:0', grad_fn=<NllLoss2DBackward0>), 
        # 'loss_sub_bbox_1_unscaled': tensor(0.2595, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_obj_bbox_1_unscaled': tensor(0.4224, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_sub_giou_1_unscaled': tensor(0.4935, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_obj_giou_1_unscaled': tensor(0.9696, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_feat_mimic_1_unscaled': tensor(0.8030, device='cuda:0', grad_fn=<MeanBackward0>)    }
        if hasattr(criterion, 'loss_labels'): # False
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        elif hasattr(criterion, 'loss_hoi_labels'): # True
            metric_logger.update(hoi_class_error=loss_dict_reduced['hoi_class_error'])
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes # 从所有进程收集统计数据
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad() # 在进行模型评估或验证时，常见的做法是将model.eval()与torch.no_grad()结合使用，以关闭梯度计算。这样可以提高评估的效率，因为在评估阶段不需要进行梯度更新。
def evaluate_hoi(dataset_file, model, postprocessors, data_loader,
                 subject_category_id, device, args): # evaluate:评价
    model.eval() # 将模型设置为评估模式，评估模式下一些层的行为会发生变化，例如Dropout层和BatchNorm层等。

    metric_logger = utils.MetricLogger(delimiter="  ") #MetricLogger度量记录器，要用来打印输出训练的时候产生的一些数据
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    counter = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        # targets: ({'orig_size': tensor([427, 640]), 'size': tensor([427, 640]), 'filename': 'COCO_val2014_000000311300.jpg', 'boxes': tensor(), 'labels': tensor()},..
        outputs = model(samples, is_training=False)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        # counter += 1


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    if dataset_file == 'hico':
        evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)

    stats = evaluator.evaluate()

    return stats
