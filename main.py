import argparse
import datetime
import json
import random
import time
from pathlib import Path
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets.hico_text_label import hico_text_label
from build_modules import *

import datasets
import util.misc as utils
from datasets import build_dataset
# from engine import train_one_epoch, evaluate_hoi
from engine import *
from models import build_model
import os

from util.scheduler import CosineAnnealingLRWarmup, MultiStepLRWarmup
from torch.utils.tensorboard import SummaryWriter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_bbox', default=3e-5, type=float)
    parser.add_argument('--lr_clip', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval_each', default=4, type=int)
    parser.add_argument('--eval_each_lr_drop', default=2, type=int)

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
    parser.add_argument('--num_workers', default=0, type=int)

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
    parser.add_argument('--with_obj_clip_label', action='store_true', help='Use clip to classify object')
    parser.add_argument('--clip_model', default='ViT-B/32',
                        help='clip pretrained model path')
    parser.add_argument('--fix_clip', action='store_true', help='')
    parser.add_argument('--clip_embed_dim', default=512, type=int)

    # zero/few shot type
    parser.add_argument('--zero_shot_type', default='default',
                        help='default, rare_first, non_rare_first, unseen_object, unseen_verb')
    parser.add_argument('--del_unseen', action='store_true', help='')
    # old parameter
    parser.add_argument('--fix_backbone_mode', nargs='+', default=[], help='fix (part of) backbone')

    # others
    parser.add_argument('--use_ddp', default=1, type=int)
    parser.add_argument('--with_random_shuffle', default=2, type=int, help='Time of random shuffle of annotation')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--opt_sched', default='multiStep', type=str, help='type of scheduler')
    parser.add_argument('--no_clip_cls_init', action='store_true',
                        help='not init classifier weight with clip text encoder')
    parser.add_argument('--enable_amp', action='store_true', help='')
    parser.add_argument('--opt_level', default='O2', help='half precision optimization level', choices=('O1', 'O2'))
    parser.add_argument('--fix_clip_label', action='store_true', help='')
    parser.add_argument('--with_rec_loss', action='store_true', help='')
    parser.add_argument('--rec_loss_coef', default=2, type=float)
    parser.add_argument('--no_training', action='store_true', help='')
    parser.add_argument('--dataset_root', default='GEN', help='')
    parser.add_argument('--model_name', default='GEN', help='')
    parser.add_argument('--eval_location', action='store_true', help='')
    # DAB
    parser.add_argument('--enable_cp', action='store_true',
                        help="use checkpoint to save memory")
    parser.add_argument('--no_fix_clip_linear', action='store_true',
                        help="")
    parser.add_argument('--analysis', action='store_true')

    # tmp args
    parser.add_argument('--alternative', default=1, type=int)
    parser.add_argument('--eval_each_ap', action='store_true')
    parser.add_argument('--topk_hoi', default=10, type=int)
    parser.add_argument('--inter_dec_layers', default=3, type=int)

    # verb setting
    parser.add_argument('--verb_pth', default='', help='location for predefined verb feature', type=str)
    parser.add_argument('--verb_weight', default=0.5, type=float)
    # fractional training
    parser.add_argument('--frac', default=-1., type=float)

    # validation split
    parser.add_argument('--validation_split', default=-1., type=int)
    parser.add_argument('--lr_drop_gamma', default=0.1, type=float)

    # zero shot enhancement
    parser.add_argument('--training_free_enhancement_path', default='', type=str)
    parser.add_argument('--loss_mae', default=1, type=float)
    parser.add_argument('--loss_domain_bac', default=1, type=float)
    parser.add_argument('--loss_domain_enc', default=1, type=float)
    parser.add_argument('--loss_domain_dec', default=1, type=float)
    parser.add_argument('--coef_target', default=1, type=float)
    parser.add_argument('--threshold', default=0.3, type=float)
    parser.add_argument('--num_classes', default=600, type=int)
    parser.add_argument('--flush', default=True, type=bool)
    parser.add_argument("--mode", default="cross_domain_mae", type=str,
                        help="'single_domain' for single domain training, "
                             "'cross_domain_mae' for cross domain training with mae, "
                             "'teaching' for teaching process, 'eval' for evaluation only.")
    parser.add_argument('--val_batch', default=20, type=int)
    parser.add_argument('--alpha_ema', default=0.999, type=float)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--mask_ratio', default=0.8, type=float)
    parser.add_argument('--epoch_mae_decay', default=10, type=float)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--control', default='[sum]', type=str, nargs="+")
    return parser

def write_loss(epoch, prefix, total_loss, loss_dict):
    writer.add_scalar(prefix + '/total_loss', total_loss, epoch)
    for k, v in loss_dict.items():
        writer.add_scalar(prefix + '/' + k, v, epoch)


def write_ap50(epoch, prefix, m_ap, ap_per_class, idx_to_class):
    writer.add_scalar(prefix + '/mAP_rare', m_ap, epoch)
    writer.add_scalar(prefix + '/mAP_no_rare', ap_per_class, epoch)
    writer.add_scalar(prefix + '/mAP50', idx_to_class, epoch)


# Teaching
def teaching(model_stu, postprocessors, device, args):
    start_time = time.time()
    # Build dataloaders
    source_train = build_dataset(image_set='train', args=args)
    target_train = build_dataset(image_set='gen_train_teacher', args=args)
    genfea=build_dataset(image_set='genfea', args=args)
    # target_train = build_dataset(image_set='gen_train', args=args)
    target_val = build_dataset(image_set='val', args=args)
    print("source_train:", len(source_train))
    print("target_train:", len(target_train))
    print("target_val:", len(target_val))
    if args.dataset_file=='hico':
        rare_triplets = target_val.non_rare_triplets
        rare_1 = [(i[2],i[1]) for i in rare_triplets]
        hico_triplet_labels = list(hico_text_label.keys())
        rare=[hico_triplet_labels.index(i) for i in rare_1]
    else:
        norare=[22, 60, 103, 233, 73, 134, 169, 187, 207, 211, 221, 222, 225, 227, 230, 239, 242, 243, 244, 248, 252, 12]
        rare=[]
        mm=range(263)
        for i in mm:
            if i not in norare:
                rare.append(i)
        


    if args.distributed:
        sampler_source = DistributedSampler(source_train)
        sampler_target = DistributedSampler(target_train)
        sampler_val = DistributedSampler(target_val, shuffle=False)
        sampler_genfea = DistributedSampler(genfea, shuffle=False)
    else:
        sampler_source = torch.utils.data.RandomSampler(source_train)
        sampler_target = torch.utils.data.RandomSampler(target_train)
        sampler_val = torch.utils.data.SequentialSampler(target_val)
        sampler_genfea = torch.utils.data.SequentialSampler(genfea)

    batch_source_train = torch.utils.data.BatchSampler(
        sampler_source, 9, drop_last=True)
    # batch_source_train = torch.utils.data.BatchSampler(
        # sampler_source, 3, drop_last=True)
    # batch_source_train = torch.utils.data.BatchSampler(
    #     sampler_source, 6, drop_last=True)
    batch_target_train = torch.utils.data.BatchSampler(
        sampler_target, 2, drop_last=True)

    source_loader = DataLoader(source_train, batch_sampler=batch_source_train,
                               collate_fn=utils.collate_fn, num_workers=args.num_workers)
    target_loader = DataLoader(target_train, batch_sampler=batch_target_train,
                               collate_fn=utils.collate_fn_tea, num_workers=args.num_workers)
    # target_loader = DataLoader(target_train, batch_sampler=batch_target_train,
    #                            collate_fn=utils.collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(target_val, args.val_batch, sampler=sampler_val,
                            drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    genfea_loader = DataLoader(genfea, args.val_batch, sampler=sampler_genfea,
                            drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # Build teacher model
    model_tch, postprocessors_teacher = build_teacher(args, model_stu, device)
    # Build discriminators
    model_stu.build_discriminators(device)
    # Build MAE branch
    image_size = target_loader.dataset.__getitem__(0)[0].shape[-2:]
    model_stu.transformer.build_mae_decoder(image_size, device, model_stu.backbone.num_channels)
    # Prepare model for optimization
    model_stu.to(device)
    model_tch.to(device)
    if args.distributed:
        model_stu = torch.nn.parallel.DistributedDataParallel(model_stu, device_ids=[args.gpu],
                                                              find_unused_parameters=True)
        model_tch = torch.nn.parallel.DistributedDataParallel(model_tch, device_ids=[args.gpu])
        model_stu_without_ddp = model_stu.module
        model_tch_without_ddp = model_tch.module

    for name, p in model_stu.named_parameters():
        if 'eval_visual_projection' in name:
            p.requires_grad = False

    if args.fix_clip:
        for name, p in model_stu.named_parameters():
            if 'obj_visual_projection' in name or 'visual_projection' in name or 'clip_model' in name:
                p.requires_grad = False
    if args.no_fix_clip_linear:
        for name, p in model_stu.named_parameters():
            if 'obj_visual_projection' in name or 'visual_projection' in name:
                p.requires_grad = True
    for name, p in model_stu.named_parameters():
        if 'transformer.instance_decoder' in name:
            p.requires_grad = False
    # for name, p in model_stu.named_parameters():
    #     if 'hoi_class_fc' in name:
    #         p.requires_grad = True
    #     else:
    #         p.requires_grad = False
    # for name, p in model_tch.named_parameters():
    #     p.requires_grad = False
    # param_dicts = [
    #     {"params": [p for n, p in model_stu.named_parameters() if p.requires_grad]}
    # ]
    # optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # for name, p in model_stu.named_parameters():
    #     if 'sub_bbox_embed' in name or 'obj_bbox_embed' in name:
    #         p.requires_grad = False
    criterion = build_criterion(args, device)
    criterion_pseudo = build_criterion(args, device)
    # criterion_pseudo = build_criterion_pesudo(args, device)
    optimizer = build_optimizer(args, model_stu)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)
    if args.opt_sched == 'multiStep':
        lr_scheduler = MultiStepLRWarmup(optimizer, [args.lr_drop], warmup_iter=0, warmup_ratio=0.01,
                                         gamma=args.lr_drop_gamma)
    elif args.opt_sched == 'cosine':
        lr_scheduler = CosineAnnealingLRWarmup(optimizer, verbose=False,
                                               warmup_iter=500,
                                               warmup_ratio=0.01,
                                               T_max=args.epoch - 1,
                                               eta_min=0.01)
    else:
        raise KeyError('Unsupported scheduler type')
    # Reinitialize checkpoint for selective retraining
    # reinit_ckpt = copy.deepcopy(model_tch.state_dict())
    # Initialize thresholds
    thresholds = [args.threshold] * args.num_classes
    # print(rare)
    for j in rare:
        thresholds[j] = 2
    # thresholds[rare]=1
    # Record the best mAP
    ap50_best = -1.0
    best_performance_rare = -1.0
    best_performance_norare = -1.0
    increase_per_epoch = (0.9999 - args.alpha_ema) / (args.epoch - 1)
    for epoch in range(args.epoch):
        # Set the epoch for the sampler
        if args.distributed and hasattr(source_loader.sampler, 'set_epoch'):
            source_loader.sampler.set_epoch(epoch)
            target_loader.sampler.set_epoch(epoch)
        # alpha_ema = args.alpha_ema + epoch * increase_per_epoch
        # print(alpha_ema)
        alpha_ema = args.alpha_ema
        gen_fea(args.dataset_file, model_stu, postprocessors, genfea_loader,
                                          args.subject_category_id, device, args)
        loss_train, loss_source_dict, loss_target_dict, vis = train_one_epoch_teaching(
            student_model=model_stu,
            teacher_model=model_tch,
            criterion=criterion,
            criterion_pseudo=criterion_pseudo,
            source_loader=source_loader,
            target_loader=target_loader,
            optimizer=optimizer,
            thresholds=thresholds,
            coef_target=args.coef_target,
            mask_ratio=args.mask_ratio,
            alpha_ema=alpha_ema,
            device=device,
            epoch=epoch,
            image_size=image_size,
            postprocessors=postprocessors_teacher,
            correct_mat=val_loader.dataset.correct_mat,
            args=args,
            enable_mae=(epoch < args.epoch_mae_decay),
            clip_max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # JSON 文件路径
        json_file_path = os.path.join(vis_dir, 'vis_{}.json'.format(epoch))
        utils.all_gather(vis)
        # 将列表保存到 JSON 文件
        if utils.is_main_process():
            with open(json_file_path, 'w') as json_file:
                json.dump(vis, json_file)
        # Renew thresholds
        # thresholds = criterion.dynamic_threshold(thresholds)
        # criterion.clear_positive_logits()
        # for j in rare:
        #     thresholds[j] = 1
        # Write the losses to tensorboard
        write_loss(epoch, 'teaching_source', loss_train, loss_source_dict)
        write_loss(epoch, 'teaching_target', loss_train, loss_target_dict)
        lr_scheduler.step()
        # Selective Retraining
        # if (epoch + 1) % args.epoch_retrain == 0:
        #     model_stu = selective_reinitialize(model_stu, reinit_ckpt, args.keep_modules)

        # Evaluate teacher and student model
        test_stats_teacher = evaluate_hoi(args.dataset_file, model_tch, postprocessors_teacher, val_loader,
                                          args.subject_category_id, device, args)
        test_stats_student = evaluate_hoi(args.dataset_file, model_stu, postprocessors, val_loader,
                                          args.subject_category_id, device, args)
        # test_stats_student={}
        # test_stats_student['mAP_all']=0
        # test_stats_student['mAP_rare_thesis']=0
        # test_stats_student['mAP_norare_thesis']=0
        # ap50_per_class_teacher, loss_val_teacher = evaluate(
        #     model=model_tch,
        #     criterion=criterion,
        #     data_loader_val=val_loader,
        #     device=device,
        #     print_freq=args.print_freq,
        #     flush=args.flush
        # )
        # ap50_per_class_student, loss_val_student = evaluate(
        #     model=model_stu,
        #     criterion=criterion,
        #     data_loader_val=val_loader,
        #     device=device,
        #     print_freq=args.print_freq,
        #     flush=args.flush
        # )
        # Save the best checkpoint
        if args.dataset_file=='hico':
            map50_tch = test_stats_teacher['mAP rare']
            map50_stu = test_stats_student['mAP rare']
            tch = test_stats_teacher['mAP']
            stu = test_stats_student['mAP']
            norare_tch = test_stats_teacher['mAP non-rare']
            norare_stu = test_stats_student['mAP non-rare']
            write_ap50(epoch, 'teaching_teacher', map50_tch, test_stats_teacher['mAP non-rare'], test_stats_teacher['mAP'])
            write_ap50(epoch, 'teaching_student', map50_stu, test_stats_student['mAP non-rare'], test_stats_student['mAP'])
            if max(map50_tch, map50_stu) > ap50_best:
                ap50_best = max(map50_tch, map50_stu)
                best_performance = max(tch, stu)
                best_performance_norare = max(norare_tch, norare_stu)
                checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
                if map50_tch < map50_stu:
                    utils.save_on_master({
                        'model': model_stu_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
                else:
                    utils.save_on_master({
                        'model': model_tch_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

            if epoch == args.epoch - 1:
                checkpoint_path_tch = os.path.join(output_dir, 'checkpoint_last_tch.pth')
                checkpoint_path_stu = os.path.join(output_dir, 'checkpoint_last_stu.pth')
                utils.save_on_master({
                    'model': model_stu_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path_tch)
                utils.save_on_master({
                    'model': model_tch_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path_stu)
        else:
            map50_tch = test_stats_teacher['mAP_rare_thesis']
            map50_stu = test_stats_student['mAP_rare_thesis']
            tch = test_stats_teacher['mAP_all']
            stu = test_stats_student['mAP_all']
            norare_tch = test_stats_teacher['mAP_norare_thesis']
            norare_stu = test_stats_student['mAP_norare_thesis']
            write_ap50(epoch, 'teaching_teacher', map50_tch, test_stats_teacher['mAP_norare_thesis'], test_stats_teacher['mAP_all'])
            write_ap50(epoch, 'teaching_student', map50_stu, test_stats_student['mAP_norare_thesis'], test_stats_student['mAP_all'])
            if max(map50_tch, map50_stu) > ap50_best:
                ap50_best = max(map50_tch, map50_stu)
                best_performance = max(tch, stu)
                best_performance_norare = max(norare_tch, norare_stu)
                checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
                if map50_tch < map50_stu:
                    utils.save_on_master({
                        'model': model_stu_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
                else:
                    utils.save_on_master({
                        'model': model_tch_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

            if epoch == args.epoch - 1:
                checkpoint_path_tch = os.path.join(output_dir, 'checkpoint_last_tch.pth')
                checkpoint_path_stu = os.path.join(output_dir, 'checkpoint_last_stu.pth')
                utils.save_on_master({
                    'model': model_stu_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path_tch)
                utils.save_on_master({
                    'model': model_tch_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path_stu)

    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching finished. Time cost: ' + total_time_str + ' . Best mAP50: ' + str(best_performance) +
          ' . Best mAP_rare: ' + str(ap50_best) +
          ' . Best mAP_norare: ' + str(best_performance_norare), flush=args.flush)


def main(args):
    # Initialize distributed mode
    utils.init_distributed_mode(args)

    # if args.frozen_weights is not None:
    #     assert args.masks, "Frozen training is meant for segmentation only"

    # Set random seed
    seed = args.seed + utils.get_rank()
    setup_seed(seed)
    # setup_seed(233)
    # Print args
    print('-------------------------------------', flush=args.flush)
    print('Logs will be written to ' + str(logs_dir))
    print('Checkpoints will be saved to ' + str(output_dir))
    print('-------------------------------------', flush=args.flush)
    with open(Path(args.output_dir).joinpath('args.txt'), 'a') as file0:
        for key, value in args.__dict__.items():
            print(key, value, file=file0)
    # Build model
    device = torch.device(args.device)
    model, postprocessors = build_model(args)
    # for name, p in model.named_parameters():
    #     print(name)
    if args.resume != "":
        # model = resume_and_load(model, args.resume, device)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    # Training or evaluation
    print('-------------------------------------', flush=args.flush)
    
    if args.mode == "teaching":
        teaching(model, postprocessors, device, args)
    elif args.mode == "eval":
        target_val = build_dataset(image_set='val', args=args)
        if args.distributed:
            sampler_val = DistributedSampler(target_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(target_val)
        model.to(device)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            
        val_loader = DataLoader(target_val, args.val_batch, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        # eval_only(model, device)
        evaluate_hoi(args.dataset_file, model, postprocessors, val_loader, args.subject_category_id, device, args)
    elif args.mode == "genfea":
        target_val = build_dataset(image_set='val', args=args)
        if args.distributed:
            sampler_val = DistributedSampler(target_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(target_val)
        model.to(device)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            
        val_loader = DataLoader(target_val, args.val_batch, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        # eval_only(model, device)
        save_features(args.dataset_file, model, postprocessors, val_loader, args.subject_category_id, device, args)
    else:
        raise ValueError('Invalid mode: ' + args.mode)


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser('GEN VLKT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    logs_dir = output_dir / 'data_logs'
    vis_dir = output_dir / 'vis_logs'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    Path(vis_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logs_dir))
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    main(args)


# def main(args):
#     if args.use_ddp == 1:
#         utils.init_distributed_mode(args)
#     else:
#         args.distributed = False

#     # args.save_points = [int(i) for i in args.save_points]

#     print('setting up seeds')
#     setup_seed(233)

#     # sys.exit(0)

#     print("git:\n  {}\n".format(utils.get_sha()))

#     if args.frozen_weights is not None:
#         assert args.masks, "Frozen training is meant for segmentation only"
#     print(args)

#     device = torch.device(args.device)

#     # fix the seed for reproducibility
#     seed = args.seed + utils.get_rank()
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

#     model, criterion, postprocessors = build_model(args)
#     model.to(device)
#     print('****************')
#     # print(model)
#     print(args.model_name)
#     print('****************')

#     model_without_ddp = model
#     if args.distributed:
#         if args.enable_amp:
#             raise NotImplementedError
#         else:
#             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
#             model_without_ddp = model.module

#     # model = convert_syncbn_model(model)

#     for name, p in model.named_parameters():
#         if 'eval_visual_projection' in name:
#             p.requires_grad = False

#     if args.fix_clip:
#         for name, p in model.named_parameters():
#             if 'obj_visual_projection' in name or 'visual_projection' in name or 'clip_model' in name:
#                 p.requires_grad = False

#     if args.no_fix_clip_linear:
#         for name, p in model.named_parameters():
#             if 'obj_visual_projection' in name or 'visual_projection' in name:
#                 p.requires_grad = True

#     if args.ft_clip_with_small_lr:
#         if args.with_obj_clip_label and args.with_clip_label:
#             param_dicts = [
#                 {"params": [p for n, p in model_without_ddp.named_parameters() if
#                             "backbone" not in n and 'visual_projection' not in n and 'obj_visual_projection' not in n
#                             and 'clip_model' not in n and p.requires_grad and 'T5_model' not in n]},
#                 {
#                     "params": [p for n, p in model_without_ddp.named_parameters() if
#                                "backbone" in n and p.requires_grad],
#                     "lr": args.lr_backbone,
#                 },
#                 {
#                     "params": [p for n, p in model_without_ddp.named_parameters() if
#                                (
#                                        'visual_projection' in n or 'obj_visual_projection' in n or 'clip_model' in n) and p.requires_grad],
#                     "lr": args.lr_clip,
#                 },
#                 {
#                     "params": [p for n, p in model_without_ddp.named_parameters() if
#                                (
#                                        'T5_model' in n or 'llm' in n) and p.requires_grad],
#                     "lr": args.lr_llm,
#                 },
#             ]
#         elif args.with_clip_label:
#             param_dicts = [
#                 {"params": [p for n, p in model_without_ddp.named_parameters() if
#                             "backbone" not in n and 'visual_projection' not in n and 'clip_model' not in n and p.requires_grad]},
#                 {
#                     "params": [p for n, p in model_without_ddp.named_parameters() if
#                                "backbone" in n and p.requires_grad],
#                     "lr": args.lr_backbone,
#                 },
#                 {
#                     "params": [p for n, p in model_without_ddp.named_parameters() if
#                                ('visual_projection' in n or 'clip_model' in n) and p.requires_grad],
#                     "lr": args.lr_clip,
#                 },
#             ]
#         elif args.with_obj_clip_label:
#             param_dicts = [
#                 {"params": [p for n, p in model_without_ddp.named_parameters() if
#                             "backbone" not in n and 'obj_visual_projection' not in n and 'clip_model' not in n and p.requires_grad]},
#                 {
#                     "params": [p for n, p in model_without_ddp.named_parameters() if
#                                "backbone" in n and p.requires_grad],
#                     "lr": args.lr_backbone,
#                 },
#                 {
#                     "params": [p for n, p in model_without_ddp.named_parameters() if
#                                ('obj_visual_projection' in n or 'clip_model' in n) and p.requires_grad],
#                     "lr": args.lr_clip,
#                 },
#             ]
#         else:
#             raise
#     else:
#         param_dicts = [
#             {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
#             {
#                 "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
#                 "lr": args.lr_backbone,
#             },
#         ]
#     optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
#                                   weight_decay=args.weight_decay)

#     n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print('number of params:', n_parameters)

#     if args.opt_sched == 'multiStep':
#         lr_scheduler = MultiStepLRWarmup(optimizer, [args.lr_drop], warmup_iter=0, warmup_ratio=0.01,
#                                          gamma=args.lr_drop_gamma)
#     elif args.opt_sched == 'cosine':
#         lr_scheduler = CosineAnnealingLRWarmup(optimizer, verbose=False,
#                                                warmup_iter=500,
#                                                warmup_ratio=0.01,
#                                                T_max=args.epoch - 1,
#                                                eta_min=0.01)
#     else:
#         raise KeyError('Unsupported scheduler type')

#     print('init dataloader')
#     # train dataloader initialization
#     dataset_train = build_dataset(image_set='train', args=args)

#     if args.distributed:
#         sampler_train = DistributedSampler(dataset_train)

#     else:
#         sampler_train = torch.utils.data.RandomSampler(dataset_train)

#     batch_sampler_train = torch.utils.data.BatchSampler(
#         sampler_train, args.batch_size, drop_last=True)

#     data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
#                                    collate_fn=utils.collate_fn, num_workers=args.num_workers)

#     # test and val dataloader initialization

#     test_split = 'val'
#     dataset_val = build_dataset(image_set='val', args=args)
#     dataset_test = build_dataset(image_set=test_split, args=args)
#     if args.distributed:
#         sampler_val = DistributedSampler(dataset_val, shuffle=False)
#         sampler_test = DistributedSampler(dataset_test, shuffle=False)
#     else:
#         sampler_val = torch.utils.data.SequentialSampler(dataset_val)
#         sampler_test = torch.utils.data.SequentialSampler(dataset_test)

#     data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
#                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
#     data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
#                                   drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
#     print('dataloader finished')

#     if args.frozen_weights is not None:
#         checkpoint = torch.load(args.frozen_weights, map_location='cpu')
#         model_without_ddp.detr.load_state_dict(checkpoint['model'])

#     output_dir = Path(args.output_dir)

#     # init logging
#     _LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
#     _DATE_FMT = '%m/%d/%Y %H:%M:%S'
#     logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
#     LOGGER = logging.getLogger('__main__')  # this is the global logger
#     fh = logging.FileHandler(os.path.join(output_dir, 'training_log.txt'))
#     formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
#     fh.setFormatter(formatter)
#     LOGGER.addHandler(fh)

#     if args.resume and os.path.exists(args.resume):
#         if args.resume.startswith('https'):
#             checkpoint = torch.hub.load_state_dict_from_url(
#                 args.resume, map_location='cpu', check_hash=True)
#         else:
#             checkpoint = torch.load(args.resume, map_location='cpu')
#         model_without_ddp.load_state_dict(checkpoint['model'])
#         if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#             args.start_epoch = checkpoint['epoch'] + 1

#         # if args.enable_amp:
#         #     amp.load_state_dict(checkpoint['amp'])

#     elif args.pretrained:
#         checkpoint = torch.load(args.pretrained, map_location='cpu')
#         if args.eval:
#             # model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
#             model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
#         else:
#             model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

#     if args.eval:
#         if not os.path.exists(output_dir / "log.txt"):
#             with open(output_dir / "log.txt", 'w') as f:
#                 f.write('')
#         with open(output_dir / "log.txt", 'r') as f:
#             previous_log = f.read()

#         if 'Test result:' not in previous_log:
#             print('Evaluating in test split!')
#             test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_test,
#                                       args.subject_category_id, device, args)

#             if args.output_dir and utils.is_main_process():
#                 #  add eval in log for my convenience
#                 with (output_dir / "log.txt").open("a") as f:
#                     f.write('Test result:' + json.dumps(test_stats) + "\n")
#                 LOGGER.info('Epoch Test: [{}] '.format('eval') + json.dumps(test_stats))

#         # if 'Val result:' not in previous_log:
#         #     print('Evaluating in val split!')
#         #     test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val,
#         #                               args.subject_category_id, device, args)

#         #     if args.output_dir and utils.is_main_process():
#         #         #  add eval in log for my convenience
#         #         with (output_dir / "log.txt").open("a") as f:
#         #             f.write('Val result:' + json.dumps(test_stats) + "\n")
#         #         LOGGER.info('Epoch Val: [{}] '.format('eval') + json.dumps(test_stats))
#         return

#     best_performance = 0
#     if args.resume and os.path.exists(args.resume):
#         try:
#             with open(output_dir / "log.txt", 'r') as f:
#                 previous_log = f.read().split('\n')
#             previous_log.remove('')
#             test_stats = json.loads(previous_log[-1])
#             if args.dataset_file == 'hico':
#                 performance = test_stats['mAP']
#             elif args.dataset_file == 'vcoco':
#                 performance = test_stats['mAP_all']
#             elif args.dataset_file == 'hoia':
#                 performance = test_stats['mAP']
#             best_performance = performance
#         except:
#             best_performance = 0

#     print("Start training")
#     start_time = time.time()
#     for epoch in range(args.start_epoch, args.epoch):
#         if args.distributed:
#             sampler_train.set_epoch(epoch)

#         train_stats = train_one_epoch(
#             model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, lr_scheduler,
#             args.gradient_accumulation_steps, args.enable_amp, args.no_training, args)

#         lr_scheduler.step()
#         # if epoch == args.epochs - 1:
#         checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pth')
#         utils.save_on_master({
#                                  'model': model_without_ddp.state_dict(),
#                                  'optimizer': optimizer.state_dict(),
#                                  'lr_scheduler': lr_scheduler.state_dict(),
#                                  # 'amp': amp.state_dict(),
#                                  'epoch': epoch,
#                                  'args': args,
#                              } if args.enable_amp else {
#             'model': model_without_ddp.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'lr_scheduler': lr_scheduler.state_dict(),
#             # 'amp': None,
#             'epoch': epoch,
#             'args': args,
#         }, checkpoint_path)

#         test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val,
#                                   args.subject_category_id, device, args)
#         if args.dataset_file == 'hico':
#             performance = test_stats['mAP']
#         elif args.dataset_file == 'vcoco':
#             performance = test_stats['mAP_all']
#         elif args.dataset_file == 'hoia':
#             performance = test_stats['mAP']

#         if performance > best_performance:
#             checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
#             utils.save_on_master({
#                                      'model': model_without_ddp.state_dict(),
#                                      'optimizer': optimizer.state_dict(),
#                                      'lr_scheduler': lr_scheduler.state_dict(),
#                                      # 'amp': amp.state_dict(),
#                                      'epoch': epoch,
#                                      'args': args,
#                                  } if args.enable_amp else {
#                 'model': model_without_ddp.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'lr_scheduler': lr_scheduler.state_dict(),
#                 # 'amp': None,
#                 'epoch': epoch,
#                 'args': args,
#             }, checkpoint_path)

#             best_performance = performance

#             if epoch in args.save_points and utils.is_main_process():
#                 checkpoint_path = os.path.join(output_dir, f'best_before_epoch_{epoch}.pth')
#                 print('achieve save point')
#                 if os.path.exists(os.path.join(output_dir, 'checkpoint_best.pth')):
#                     os.system(f"cp {os.path.join(output_dir, 'checkpoint_best.pth')} {checkpoint_path}")
#                 elif os.path.exists(os.path.join(output_dir, 'checkpoint_last.pth')):
#                     os.system(f"cp {os.path.join(output_dir, 'checkpoint_last.pth')} {checkpoint_path}")
#                 else:
#                     raise ValueError

#         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
#                      **{f'test_{k}': v for k, v in test_stats.items()},
#                      'epoch': epoch,
#                      'n_parameters': n_parameters}

#         if args.output_dir and utils.is_main_process():
#             with (output_dir / "log.txt").open("a") as f:
#                 f.write(json.dumps(log_stats) + "\n")
#             LOGGER.info('Epoch: [{}] '.format(epoch) + json.dumps(log_stats))

#             #  add eval in log for my convenience
#             with (output_dir / "log.txt").open("a") as f:
#                 f.write(json.dumps(test_stats) + "\n")
#             LOGGER.info('Epoch: [{}] '.format(epoch) + json.dumps(test_stats))

#         if epoch == args.epoch - 1 and os.path.exists(os.path.join(output_dir, 'checkpoint_best.pth')):
#             print('Loading best val checkpoint!')
#             checkpoint = torch.load(os.path.join(output_dir, 'checkpoint_best.pth'), map_location='cpu')
#             model_without_ddp.load_state_dict(checkpoint['model'])
#             if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
#                 optimizer.load_state_dict(checkpoint['optimizer'])
#                 lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#                 args.start_epoch = -1
#             model.to(device)
#             print('Final evaluating in test split!')
#             test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_test,
#                                       args.subject_category_id, device, args)

#             if args.output_dir and utils.is_main_process():
#                 #  add eval in log for my convenience
#                 with (output_dir / "log.txt").open("a") as f:
#                     f.write('Test result:' + json.dumps(test_stats) + "\n")
#                 LOGGER.info('Epoch Test: [{}] '.format(epoch) + json.dumps(test_stats))

#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     print('Training time {}'.format(total_time_str))


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('GEN VLKT training and evaluation script', parents=[get_args_parser()])
#     args = parser.parse_args()
#     if args.output_dir:
#         Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#     main(args)
