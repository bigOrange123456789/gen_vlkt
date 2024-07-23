

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

