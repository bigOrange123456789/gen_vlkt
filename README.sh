

requirements:

cython
pycocotools
torch==1.7.1
torchvision==0.8.2
scipy==1.3.1
opencv-python
ftfy
regex
tqdm


Training:
[HICO-DET]
python  main.py --pretrained params/detr-r50-pre-2branch-hico.pth  --output_dir exps/hico_gen_vlkt_s_r50_dec_3layers  --dataset_file hico  --hoi_path data/hico_20160224_det --num_obj_classes 80 --num_verb_classes 117 --backbone resnet50 --num_queries 64 --dec_layers 3 --epochs 90 --lr_drop 60 --use_nms_filter --ft_clip_with_small_lr --with_clip_label --with_obj_clip_label --with_mimic --mimic_loss_coef 20
[V-COCO]
python  main.py --pretrained params/detr-r50-pre-2branch-vcoco.pth --output_dir exps/vcoco_gen_vlkt_s_r50_dec_3layers --dataset_file vcoco --hoi_path data/v-coco            --num_obj_classes 81 --num_verb_classes 29  --backbone resnet50 --num_queries 64 --dec_layers 3 --epochs 90 --lr_drop 60 --use_nms_filter --ft_clip_with_small_lr --with_clip_label --with_obj_clip_label --with_mimic --mimic_loss_coef 20
python  main.py --pretrained params/detr-r50-pre-2branch-vcoco.pth --output_dir exps/vcoco_gen_vlkt_s_r50_dec_3layers --dataset_file vcoco --hoi_path data/v-coco            --num_obj_classes 81 --num_verb_classes 29  --backbone resnet50 --num_queries 64 --dec_layers 3 --epochs  1 --lr_drop 60 --use_nms_filter --ft_clip_with_small_lr --with_clip_label --with_obj_clip_label --with_mimic --mimic_loss_coef 20

python main.py \
        --pretrained params/detr-r50-pre-2branch-vcoco.pth \
        预训练的DETR模型 \
        pth文件是PyTorch模型保存和加载的格式之一 \
        --output_dir exps/vcoco_gen_vlkt_s_r50_dec_3layers \
        训练后的模型参数 \
        --dataset_file vcoco \
        数据集名称 \
        --hoi_path data/v-coco \
        预训练后的模型参数 \
        --num_obj_classes 81 \
        对象的种类数量 \
        --num_verb_classes 29 \
        关系动词的种类数量 \
        --backbone resnet50 \
        主干网的类型 \
        --num_queries 64 \
        transformer所处理矩阵的行数 \
        --dec_layers 3 \
        InstanceDecoder和InteractionDecoder 的层数 \
        --epochs 90 \
        数据被重复使用的次数 \
        --lr_drop 60 \
        数据集名称 \
        Step Decay \
        lr = lr0 * drop^floor(epoch / epochs_drop) \
        drop = 0.5 ; epochs_drop = 10.0 逐步衰减时间表会使学习率每隔几个时期下降一半 \
        --use_nms_filter \
        ? \
        --ft_clip_with_small_lr \
        ? \
        --with_clip_label \
        ? \
        --with_obj_clip_label \
        ? \
        --with_mimic \
        ? \
        --mimic_loss_coef 20 \
        论文中公式6的系数 \