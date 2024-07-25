import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import numpy as np
import clip
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
from datasets.vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label

from .backbone import build_backbone
from .matcher import build_matcher
from .gen import build_gen


def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y

class DETR(nn.Module): # DETR这部分代码是从网上复制过来用来学习的，不参与程序的执行
  # https://shihan-ma.github.io/posts/2021-04-15-DETR_annotation
  def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
    super().__init__()
    self.num_queries = num_queries
    self.transformer = transformer      # transformer 模型
    hidden_dim = transformer.d_model    # 隐层维度，一般设为 256
    self.class_embed = nn.Linear(hidden_dim, num_classes + 1)   # 全连接层，预测每类的概率
    self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)         # 多层感知机，用于预测边界框位置
    self.query_embed = nn.Embedding(num_queries, hidden_dim)    # 网络可学习的参数，含有物体抽象特征
    self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)   # 1*1 卷积，用于特征降维
    self.backbone = backbone  # 主干网络
    self.aux_loss = aux_loss

  def forward(self, samples: NestedTensor):
    if isinstance(samples, (list, torch.Tensor)):
        samples = nested_tensor_from_tensor_list(samples)   # 将 samples 包裹在 nested_tensor 中
    features, pos = self.backbone(samples)
    # mask 是所有图像 padding 后的维度（padding 到相同维度）
    src, mask = features[-1].decompose()    # src [bs, dim, H, W], mask [bs, H, W]
    assert mask is not None
    hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]      
    # transformer 返回 decoder 输出的 query embedding 以及 encoder 输出的 memory, 这里 [0] 表示取 query embedding

    outputs_class = self.class_embed(hs)    # 将 transformer 输出结果转换为 类 的预测概率
    outputs_coord = self.bbox_embed(hs).sigmoid()   # 将 transformer 输出结果转换为 框 的位置信息，sigmoid 归一化到 0-1
    out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
    if self.aux_loss:
        out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
    return out

class GEN_VLKT(nn.Module):
    def __init__(self, backbone, transformer, num_queries, aux_loss=False, args=None):
        # backbone=Joiner(...) 
        # transformer=GEN(...) 
        # num_queries=64 # ？
        # aux_loss=True  # ？
        # args=Namespace(lr=0.0001, ..)
        super().__init__()

        self.args = args 
        self.num_queries = num_queries # 64,字典中词的个数
        self.transformer = transformer
        hidden_dim = transformer.d_model #模型的嵌入隐维度
        self.query_embed_h = nn.Embedding(num_queries, hidden_dim)#num_queries=64, hidden_dim=256
        # nn.Embedding是PyTorch中的一个常用模块，其主要作用是将输入的整数序列转换为密集向量表示。
        # 在自然语言处理（NLP）任务中，可以将每个单词表示成一个向量，从而方便进行下一步的计算和处理。
        self.query_embed_o = nn.Embedding(num_queries, hidden_dim)
        self.pos_guided_embedd = nn.Embedding(num_queries, hidden_dim)
        self.hum_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) # in, hidden, out, num_layers
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1) # 1*1 卷积，用于特征降维
        # in_channels=2048 
        # out_channels=256 
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.dec_layers = self.args.dec_layers # dec_layers=3 ?

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # torch.nn.Parameter()将一个不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面。
        # 即在定义网络时这个tensor就是一个可以训练的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        # 在训练网络的时候，可以使用nn.Parameter()来转换一个固定的权重数值，使其可以在反向传播时进行参数更新，从而学习到一个最适合的权重值。

        if self.args.dataset_file == 'hico':
            hoi_text_label = hico_text_label
            obj_text_label = hico_obj_text_label
            unseen_index = hico_unseen_index
        elif self.args.dataset_file == 'vcoco':
            hoi_text_label = vcoco_hoi_text_label # hoi_text_label: {(0, 41): 'a photo of a person holding a cup', (16, 80): 'a photo of a person cutting with something', ... 
            obj_text_label = vcoco_obj_text_label # obj_text_label: [(0, 'a photo of a person and a person'), (1, 'a photo of a person and a bicycle'), ...
            unseen_index = None

        clip_label, obj_clip_label, v_linear_proj_weight, hoi_text, obj_text, train_clip_label = \
            self.init_classifier_with_CLIP(hoi_text_label, obj_text_label, unseen_index) # 获取文本的clip嵌入向量
        num_obj_classes = len(obj_text) - 1  # del nothing # 最后一个标签是'图片里什么都没有' # 'a photo of nothing'
        # obj_text_label: [(0, 'a photo of a person and a person'), (1, 'a photo of a person and a bicycle'), ...
        # len(obj_text_label): 82
        # len(obj_text): 82
        # obj_text_label[81]: (81, 'a photo of nothing')

        self.hoi_class_fc = nn.Sequential(
            nn.Linear(hidden_dim, args.clip_embed_dim),
            nn.LayerNorm(args.clip_embed_dim), # hidden=256 clip_embed=512
        )

        if args.with_clip_label: # with_clip_label=True
            self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text))
            self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)
            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default': # dataset_file=vcoco zero_shot_type=default # False
                self.eval_visual_projection = nn.Linear(args.clip_embed_dim, 600)
                self.eval_visual_projection.weight.data = clip_label / clip_label.norm(dim=-1, keepdim=True)
        else:
            self.hoi_class_embedding = nn.Linear(args.clip_embed_dim, len(hoi_text))

        if args.with_obj_clip_label: # with_clip_label: True
            self.obj_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, args.clip_embed_dim),
                nn.LayerNorm(args.clip_embed_dim),
            )
            self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1)
            self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
        else:
            self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)

        self.hidden_dim = hidden_dim
        self.reset_parameters() # 随机初始化参数值

    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embedd.weight)

    def init_classifier_with_CLIP(self, hoi_text_label, obj_text_label, unseen_index):
        # hoi_text_label: {(0, 41): 'a photo of a person holding a cup', (16, 80): 'a photo of a person cutting with something', ... 
        # obj_text_label: [(0, 'a photo of a person and a person'), (1, 'a photo of a person and a bicycle'), ...
        # unseen_index = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_inputs = torch.cat( # 将hoi_text_label里面的文本进行clip编码
            [clip.tokenize(hoi_text_label[id]) for id in hoi_text_label.keys()])
        # clip.tokenize('a photo of a person holding a cup'): tensor([[49406,   320,  1125, ...  
        # torch.Size([1, 77])
        # torch.cat : 把多个tensor进行拼接。
        # hoi_text_label.keys() : [(0, 41), (16, 80), (17, 53), ...
        # [x**2 for x in range(10)] => [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        if self.args.del_unseen and unseen_index is not None: # del_unseen=False unseen_index=None # False
            hoi_text_label_del = {}
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
            for idx, k in enumerate(hoi_text_label.keys()):
                if idx in unseen_index_list:
                    continue
                else:
                    hoi_text_label_del[k] = hoi_text_label[k]
        else:
            hoi_text_label_del = hoi_text_label.copy()
        text_inputs_del = torch.cat( # 【当前参数下】结果与text_inputs相同
            [clip.tokenize(hoi_text_label[id]) for id in hoi_text_label_del.keys()])

        obj_text_inputs = torch.cat( # 将obj_text_label里面的文本进行clip编码
            [clip.tokenize(obj_text[1]) for obj_text in obj_text_label])
        clip_model, preprocess = clip.load(self.args.clip_model, device=device) # 加载clip的预训练模型
        with torch.no_grad(): # Python 中的 with 语句用于异常判断
            text_embedding = clip_model.encode_text(text_inputs.to(device))
            # text_inputs    = tensor([[494,    320,  11,    ... # torch.Size([263, 77])
            # text_embedding = tensor([[-0.13,  0.51, -0.26, ... # torch.Size([263, 512])
            text_embedding_del = clip_model.encode_text(text_inputs_del.to(device))
            obj_text_embedding = clip_model.encode_text(obj_text_inputs.to(device))
            # obj_text_inputs    = tensor([[49406,   320,  1125, ... # torch.Size([82, 77])
            # obj_text_embedding = tensor([[ 0.11,  0.20, -0.29, ... # torch.Size([82, 512])
            v_linear_proj_weight = clip_model.visual.proj.detach()
            # v_linear_proj_weight = tensor([[-2.62e-03,  5.09e-05,  2.74e-02, ... # torch.Size([768, 512])

        del clip_model # 获得完编码向量后就可以删除clip模型了

        return text_embedding.float(), obj_text_embedding.float(), v_linear_proj_weight.float(), \
               hoi_text_label_del, obj_text_inputs, text_embedding_del.float() #返回内容：【hoi嵌入 obj嵌入 权重 ; ? obj编码 ?】

    def forward(self, samples: NestedTensor, is_training=True): # 这个前向传播过程看起来似乎有点古怪
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        h_hs, o_hs, inter_hs = self.transformer(self.input_proj(src), mask,
                                                self.query_embed_h.weight,
                                                self.query_embed_o.weight,
                                                self.pos_guided_embedd.weight,
                                                pos[-1])[:3]

        outputs_sub_coord = self.hum_bbox_embed(h_hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(o_hs).sigmoid()

        if self.args.with_obj_clip_label:
            obj_logit_scale = self.obj_logit_scale.exp()
            o_hs = self.obj_class_fc(o_hs)
            o_hs = o_hs / o_hs.norm(dim=-1, keepdim=True)
            outputs_obj_class = obj_logit_scale * self.obj_visual_projection(o_hs)
        else:
            outputs_obj_class = self.obj_class_embed(o_hs)

        if self.args.with_clip_label:
            logit_scale = self.logit_scale.exp()
            inter_hs = self.hoi_class_fc(inter_hs)
            outputs_inter_hs = inter_hs.clone()
            inter_hs = inter_hs / inter_hs.norm(dim=-1, keepdim=True)
            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default' \
                    and (self.args.eval or not is_training):
                outputs_hoi_class = logit_scale * self.eval_visual_projection(inter_hs)
            else:
                outputs_hoi_class = logit_scale * self.visual_projection(inter_hs)
        else:
            inter_hs = self.hoi_class_fc(inter_hs)
            outputs_inter_hs = inter_hs.clone()
            outputs_hoi_class = self.hoi_class_embedding(inter_hs)

        out = {'pred_hoi_logits': outputs_hoi_class[-1], 'pred_obj_logits': outputs_obj_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}

        if self.args.with_mimic:
            out['inter_memory'] = outputs_inter_hs[-1]
        if self.aux_loss:
            if self.args.with_mimic:
                aux_mimic = outputs_inter_hs
            else:
                aux_mimic = None

            out['aux_outputs'] = self._set_aux_loss_triplet(outputs_hoi_class, outputs_obj_class,
                                                            outputs_sub_coord, outputs_obj_coord,
                                                            aux_mimic)

        return out

    @torch.jit.unused
    def _set_aux_loss_triplet(self, outputs_hoi_class, outputs_obj_class,
                              outputs_sub_coord, outputs_obj_coord, outputs_inter_hs=None):

        aux_outputs = {'pred_hoi_logits': outputs_hoi_class[-self.dec_layers: -1],
                       'pred_obj_logits': outputs_obj_class[-self.dec_layers: -1],
                       'pred_sub_boxes': outputs_sub_coord[-self.dec_layers: -1],
                       'pred_obj_boxes': outputs_obj_coord[-self.dec_layers: -1]}
        if outputs_inter_hs is not None:
            aux_outputs['inter_memory'] = outputs_inter_hs[-self.dec_layers: -1]
        outputs_auxes = []
        for i in range(self.dec_layers - 1):
            output_aux = {}
            for aux_key in aux_outputs.keys():
                output_aux[aux_key] = aux_outputs[aux_key][i]
            outputs_auxes.append(output_aux)
        return outputs_auxes


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module): #criterion标准 # 这个对象的作用应该是计算损失loss

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        # num_obj_classes:81  # 对象种类数
        # num_queries:64      # 查询目标数
        # num_verb_classes:29 # 动词种类数
        # matcher:HungarianMatcherHOI() # 匈牙利匹配算法
        # weight_dict:{'loss_hoi_labels': 2, 'loss_obj_ce': 1, ... # 权重
        # eos_coef:0.1 # eos的系数 # eos是什么？
        # losses:['hoi_labels', 'obj_labels', 'sub_obj_boxes', 'feats_mimic'] #用到的损失函数
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1) # tensor([1., 1., 1., 1., ... 1.])
        empty_weight[-1] = self.eos_coef                    # tensor([1., 1., 1., 1., ... 0.1])
        self.register_buffer('empty_weight', empty_weight)
        # 这个tensor注册到模型的 buffers() 属性中，
        # 并命名为'empty_weight',对应的是一个持久态，不会有梯度传播给它，但是能被模型的state_dict记录下来。
        # 可以理解为模型的常数。
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load(args.clip_model, device=device) 
        # args.clip_model: ViT-B/32
        # self.clip_model: CLIP(...)
        self.alpha = args.alpha # alpha: 0.5

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_hoi_labels(self, outputs, targets, indices, num_interactions, topk=5):
        assert 'pred_hoi_logits' in outputs
        src_logits = outputs['pred_hoi_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        src_logits = _sigmoid(src_logits)
        loss_hoi_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_hoi_labels': loss_hoi_ce}

        _, pred = src_logits[idx].topk(topk, 1, True, True)
        acc = 0.0
        for tid, target in enumerate(target_classes_o):
            tgt_idx = torch.where(target == 1)[0]
            if len(tgt_idx) == 0:
                continue
            acc_pred = 0.0
            for tgt_rel in tgt_idx:
                acc_pred += (tgt_rel in pred[tid])
            acc += acc_pred / len(tgt_idx)
        rel_labels_error = 100 - 100 * acc / max(len(target_classes_o), 1)
        losses['hoi_class_error'] = torch.from_numpy(np.array(
            rel_labels_error)).to(src_logits.device).float()
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                    exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def mimic_loss(self, outputs, targets, indices, num_interactions):
        src_feats = outputs['inter_memory']
        src_feats = torch.mean(src_feats, dim=1)

        target_clip_inputs = torch.cat([t['clip_inputs'].unsqueeze(0) for t in targets])
        with torch.no_grad():
            target_clip_feats = self.clip_model.encode_image(target_clip_inputs)
        loss_feat_mimic = F.l1_loss(src_feats, target_clip_feats)
        losses = {'loss_feat_mimic': loss_feat_mimic}
        return losses

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        if 'pred_hoi_logits' in outputs.keys():
            loss_map = {
                'hoi_labels': self.loss_hoi_labels,
                'obj_labels': self.loss_obj_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
                'feats_mimic': self.mimic_loss
            }
        else:
            loss_map = {
                'obj_labels': self.loss_obj_labels,
                'obj_cardinality': self.loss_obj_cardinality,
                'verb_labels': self.loss_verb_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
            }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['hoi_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOITriplet(nn.Module): # HOI三元组的后处理

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id # subject_category_id: 0

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_hoi_logits = outputs['pred_hoi_logits']
        out_obj_logits = outputs['pred_obj_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']

        assert len(out_hoi_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        hoi_scores = out_hoi_logits.sigmoid()
        obj_scores = out_obj_logits.sigmoid()
        obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)[1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(hoi_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(hoi_scores)):
            hs, os, ol, sb, ob = hoi_scores[index], obj_scores[index], obj_labels[index], sub_boxes[index], obj_boxes[
                index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            ids = torch.arange(b.shape[0])

            results[-1].update({'hoi_scores': hs.to('cpu'), 'obj_scores': os.to('cpu'),
                                'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:]})

        return results


def build(args):
    device = torch.device(args.device)#[args.device: cuda , device: cuda]    

    print("1.build_backbone...")
    backbone = build_backbone(args) # 这是一个添加了余弦嵌入的标准主干网，对应论文中的 CNN+PositionalEncoding
    print("2.build_gen...")
    gen = build_gen(args) # 对应论文中图1的内容

    print("3.GEN_VLKT...")
    model = GEN_VLKT(
        backbone,
        gen,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    print("4.build_matcher...")
    matcher = build_matcher(args) #匈牙利匹配算法，使用DETR论文中的思想
    print("5.SetCriterionHOI...")
    weight_dict = {}
    if args.with_clip_label: # with_clip_label: True #这个判断没有什么意义，因为后面的两段代码完全一致
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef # hoi_loss_coef: 2
        weight_dict['loss_obj_ce'] = args.obj_loss_coef     # obj_loss_coef: 1
    else:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef

    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef # bbox_loss_coef: 2.5
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef # giou_loss_coef: 1
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    if args.with_mimic: # with_mimic: True
        weight_dict['loss_feat_mimic'] = args.mimic_loss_coef # mimic_loss_coef: 20.0

    if args.aux_loss: # aux_loss: True # aux是什么？
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1): # dec_layers: 3
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            # weight_dict.items(): 
            #                dict_items([('loss_hoi_labels', 2), ('loss_obj_ce', 1), ('loss_sub_bbox', 2.5), 
            #                          ('loss_obj_bbox', 2.5), ('loss_sub_giou', 1), ('loss_obj_giou', 1), ('loss_feat_mimic', 20.0)])
            # {k + f'_0': v for k, v in weight_dict.items()}: 
            #               {'loss_hoi_labels_0': 2, 'loss_obj_ce_0': 1, 'loss_sub_bbox_0': 2.5, 
            #                   'loss_obj_bbox_0': 2.5, 'loss_sub_giou_0': 1, 'loss_obj_giou_0': 1, 'loss_feat_mimic_0': 20.0}
        weight_dict.update(aux_weight_dict)
    losses = ['hoi_labels', 'obj_labels', 'sub_obj_boxes']
    if args.with_mimic: # with_mimic: True
        losses.append('feats_mimic')

    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)
    criterion.to(device)
    print("6.PostProcessHOITriplet...")
    postprocessors = {'hoi': PostProcessHOITriplet(args)}

    return model, criterion, postprocessors
