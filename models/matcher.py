import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcherHOI(nn.Module): # Hungarian匈牙利 # DETR中的匈牙利算法

    def __init__(self, cost_obj_class: float = 1, cost_verb_class: float = 1, cost_bbox: float = 1,
                 cost_giou: float = 1, cost_hoi_class: float = 1):
        super().__init__()
        self.cost_obj_class = cost_obj_class   # 物体种类 #1
        self.cost_verb_class = cost_verb_class # 动作种类 #1
        self.cost_hoi_class = cost_hoi_class   # ?       #2.5
        self.cost_bbox = cost_bbox # 包围盒位置  #1
        self.cost_giou = cost_giou # 包围盒范围？#1
        assert cost_obj_class != 0 or cost_verb_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0' #不能所有损失都为0

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_sub_boxes'].shape[:2]
        if 'pred_hoi_logits' in outputs.keys():
            out_hoi_prob = outputs['pred_hoi_logits'].flatten(0, 1).sigmoid()
            tgt_hoi_labels = torch.cat([v['hoi_labels'] for v in targets])
            tgt_hoi_labels_permute = tgt_hoi_labels.permute(1, 0)
            cost_hoi_class = -(out_hoi_prob.matmul(tgt_hoi_labels_permute) / \
                               (tgt_hoi_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                               (1 - out_hoi_prob).matmul(1 - tgt_hoi_labels_permute) / \
                               ((1 - tgt_hoi_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
            cost_hoi_class = self.cost_hoi_class * cost_hoi_class
        else:

            out_verb_prob = outputs['pred_verb_logits'].flatten(0, 1).sigmoid()
            tgt_verb_labels = torch.cat([v['verb_labels'] for v in targets])
            tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
            cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / \
                                (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                                (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / \
                                ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2

            cost_hoi_class = self.cost_verb_class * cost_verb_class
        tgt_obj_labels = torch.cat([v['obj_labels'] for v in targets])
        out_obj_prob = outputs['pred_obj_logits'].flatten(0, 1).softmax(-1)
        cost_obj_class = -out_obj_prob[:, tgt_obj_labels]
        out_sub_bbox = outputs['pred_sub_boxes'].flatten(0, 1)
        out_obj_bbox = outputs['pred_obj_boxes'].flatten(0, 1)

        tgt_sub_boxes = torch.cat([v['sub_boxes'] for v in targets])
        tgt_obj_boxes = torch.cat([v['obj_boxes'] for v in targets])

        cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)
        if cost_sub_bbox.shape[1] == 0:
            cost_bbox = cost_sub_bbox
        else:
            cost_bbox = torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]

        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
                        cost_sub_giou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)
        if cost_sub_giou.shape[1] == 0:
            cost_giou = cost_sub_giou
        else:
            cost_giou = torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]

        C = self.cost_hoi_class * cost_hoi_class + self.cost_bbox * cost_bbox + \
            self.cost_giou * cost_giou + self.cost_obj_class * cost_obj_class

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v['sub_boxes']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcherHOI(cost_obj_class=args.set_cost_obj_class, cost_verb_class=args.set_cost_verb_class,
                               cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
                               cost_hoi_class=args.set_cost_hoi)

class HungarianMatcher(nn.Module):# HungarianMatcher这部分代码是从网上复制过来用来学习的，不参与程序的执行
  # https://shihan-ma.github.io/posts/2021-04-15-DETR_annotation
  # 对每个图像，DETR 的输出包括两部分：
  # 1.“pred_logits”：维度为 [batch_size, num_queries, num_classes]，代表每个 query 预测为某个物体的可能性；
  # 2.“pred_boxes”：维度为 [batch_size, num_queries, 4]，表示每个 query 预测到的物体边界框的位置。
  # query 的个数远大于一张图片中可能出现的物体的最大个数，那么如何将预测到的 query 与真值 targets 对应起来呢？
  # DETR 采用 匈牙利算法（Hungarian algorithm）将 query 与真实物体匹配，多余的 query 将与背景噪声空集对应。
  # 简单理解，就是计算每个 query 预测为不同类时，在概率层面的代价和在空间边界框层面的代价，找到一种组合方式，使总代价最小。
  # 匹配时预测值与真值之间的代价有以下两部分：
  # 1.cost_class 直接通过取 “pred_logits” 的相反数获得，即预测是某物体的概率越大，代价越小；
  # 2.cost_box 包括 cost_bbox 和 cost_giou。前者计算 l1 距离，后者计算 IoU cost。（ IoU 是 Intersection over Union 交并比 ）
  # 补充 IoU cost 的原因是：l1 cost 依赖于边界框的大小，IoU cost 则与边界框的尺度无关。 （l1 cost不知道是什么？）
  # 最终加入每种 cost 的权重后，有：final_cost = w1 * cost_class + w2 * cost_bbox + w3 * cost_giou

  def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
    super().__init__()
    self.cost_class = cost_class    # 预测概率 cost 对应的权重 
    self.cost_bbox = cost_bbox      # 边界框 l1 cost 的权重  
    self.cost_giou = cost_giou      # IoU cost 的权重
    assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

  def forward(self, outputs, targets):
    bs, num_queries = outputs["pred_logits"].shape[:2]
    out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
    out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
    tgt_ids = torch.cat([v["labels"] for v in targets]) #目标类别
    tgt_bbox = torch.cat([v["boxes"] for v in targets]) #目标位置
    cost_class = -out_prob[:, tgt_ids]  # 只提取目前出现过的 class 处的 prob，其余的不需要参与匹配 # 可能性越小代价越大，所以需要加负号
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)    # L1 cost
    cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))    # giou cost
    C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou  # 计算出最终损失
    C = C.view(bs, num_queries, -1).cpu()
    sizes = [len(v["boxes"]) for v in targets]    # sizes 存储每个 batch 内物体的个数
    # for i, c in enumerate(C.split(sizes, -1)) -> bs * num_queries * [num_target_boxes in image0, num_target_boxes in image1,...]
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
