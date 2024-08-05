from models.models_hoiclip.hoiclip import SetCriterionHOI
from models.matcher import build_matcher
import torch
from models import build_model


# def build_criterion_pesudo(args, device, annotations=None, enable_mae=None, domain_label=None):
#     # matcher = build_matcher(args)
#     matcher = build_matcher_pesudo(args)
#     weight_dict = {}
#
#     weight_dict['loss_obj_ce'] = args.obj_loss_coef
#     weight_dict['loss_verb_ce'] = args.verb_loss_coef
#     weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
#     weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
#     weight_dict['loss_sub_giou'] = args.giou_loss_coef
#     weight_dict['loss_obj_giou'] = args.giou_loss_coef
#     weight_dict['loss_mae'] = args.loss_mae
#     weight_dict['loss_domain_bac'] = args.loss_domain_bac
#     weight_dict['loss_domain_enc'] = args.loss_domain_enc
#     weight_dict['loss_domain_dec'] = args.loss_domain_dec
#
#     losses = []
#
#     criterion = SetCriterionHOI_Pesudo(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
#                                        weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
#                                        args=args,
#                                        alpha_dt=args.alpha_dt,
#                                        gamma_dt=args.gamma_dt,
#                                        max_dt=args.max_dt,
#                                        device=device)
#
#     criterion.to(device)
#     return criterion
#

def build_criterion(args, device, annotations=None, enable_mae=None, domain_label=None):
    matcher = build_matcher(args)
    weight_dict = {}

    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_hoi_labels'] = args.verb_loss_coef
    # weight_dict['loss_verb_ce'] = args.verb_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    weight_dict['loss_mae'] = args.loss_mae
    weight_dict['loss_domain_bac'] = args.loss_domain_bac
    weight_dict['loss_domain_enc'] = args.loss_domain_enc
    weight_dict['loss_domain_dec'] = args.loss_domain_dec
    # if args.use_matching:
    #     weight_dict['loss_matching'] = args.matching_loss_coef
    if args.with_mimic:
        weight_dict['loss_feat_mimic'] = args.mimic_loss_coef
    if args.with_rec_loss:
        weight_dict['loss_rec'] = args.rec_loss_coef
    if args.aux_loss:
        # min_dec_layers_num = min(args.dec_layers_hopd, args.dec_layers_interaction)
        aux_weight_dict = {}
        # for i in range(min_dec_layers_num - 1):
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = []
    # if annotations is not None:
    #     losses += ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality']
    # if enable_mae:
    #     losses += ['loss_mae']
    # if domain_label is not None:
    #     losses += ['loss_domains']

    # if args.use_matching:
    #     losses.append('matching_labels')

    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)

    criterion.to(device)
    return criterion


# def build_optimizer(args, model, enable_mae=False):
#     params_backbone = [param for name, param in model.named_parameters()
#                        if 'backbone' in name]
#     params = [param for name, param in model.named_parameters()
#               if 'backbone' not in name]
#     param_dicts = [
#         {'params': params, 'lr': args.lr},
#         {'params': params_backbone, 'lr': 0.0 if enable_mae else args.lr_backbone},
#     ]
#     if args.sgd:
#         optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
#     else:
#         optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
#     return optimizer


def build_optimizer(args, model_without_ddp, enable_mae=False):
    # params_backbone = [param for name, param in model.named_parameters()
    #                    if 'backbone' in name]
    # params_bbox = [param for name, param in model.named_parameters()
    #                if 'sub_bbox_embed' in name or 'obj_bbox_embed' in name]
    # params = [param for name, param in model.named_parameters()
    #           if 'backbone' not in name and 'sub_bbox_embed' not in name and 'obj_bbox_embed' not in name]
    # param_dicts = [
    #     {'params': params, 'lr': args.lr},
    #     {'params': params_backbone, 'lr': 0.0 if enable_mae else args.lr_backbone},
    #     {'params': params_bbox, 'lr': args.lr_bbox},
    # ]
    # if args.sgd:
    #     optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # else:
    #     optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # return optimizer
    # if args.ft_clip_with_small_lr:
    #     if args.with_obj_clip_label and args.with_clip_label:
    #         param_dicts = [
    #             {"params": [p for n, p in model_without_ddp.named_parameters() if
    #                         "backbone" not in n and 'visual_projection' not in n and 'obj_visual_projection' not in n and p.requires_grad]},
    #             {
    #                 "params": [p for n, p in model_without_ddp.named_parameters() if
    #                            "backbone" in n and p.requires_grad],
    #                 "lr": 0.0 if enable_mae else args.lr_backbone,
    #             },
    #             {
    #                 "params": [p for n, p in model_without_ddp.named_parameters() if
    #                            ('visual_projection' in n or 'obj_visual_projection' in n) and p.requires_grad],
    #                 "lr": args.lr_clip,
    #             },
    #         ]
    #     elif args.with_clip_label:
    #         param_dicts = [
    #             {"params": [p for n, p in model_without_ddp.named_parameters() if
    #                         "backbone" not in n and 'visual_projection' not in n and p.requires_grad]},
    #             {
    #                 "params": [p for n, p in model_without_ddp.named_parameters() if
    #                            "backbone" in n and p.requires_grad],
    #                 "lr": 0.0 if enable_mae else args.lr_backbone,
    #             },
    #             {
    #                 "params": [p for n, p in model_without_ddp.named_parameters() if
    #                            'visual_projection' in n and p.requires_grad],
    #                 "lr": args.lr_clip,
    #             },
    #         ]
    #     elif args.with_obj_clip_label:
    #         param_dicts = [
    #             {"params": [p for n, p in model_without_ddp.named_parameters() if
    #                         "backbone" not in n and 'obj_visual_projection' not in n and p.requires_grad]},
    #             {
    #                 "params": [p for n, p in model_without_ddp.named_parameters() if
    #                            "backbone" in n and p.requires_grad],
    #                 "lr": 0.0 if enable_mae else args.lr_backbone,
    #             },
    #             {
    #                 "params": [p for n, p in model_without_ddp.named_parameters() if
    #                            'obj_visual_projection' in n and p.requires_grad],
    #                 "lr": args.lr_clip,
    #             },
    #         ]
    #     else:
    #         raise

    # else:
        
    #     param_dicts = [
    #         {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and "obj_bbox_embed" not in n and "hum_bbox_embed" not in n and p.requires_grad]},
    #         {
    #             "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
    #             "lr": 0.0 if enable_mae else args.lr_backbone,
    #         },
    #         {
    #             "params":[param for name, param in model_without_ddp.named_parameters() \
    #                    if 'hum_bbox_embed' in name or 'obj_bbox_embed' in name],
    #             "lr": args.lr_bbox,
    #         }
    #     ]

    # optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
    #                               weight_decay=args.weight_decay)
    if args.ft_clip_with_small_lr:
        if args.with_obj_clip_label and args.with_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'visual_projection' not in n and 'obj_visual_projection' not in n
                            and 'clip_model' not in n and p.requires_grad and 'T5_model' not in n]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               (
                                       'visual_projection' in n or 'obj_visual_projection' in n or 'clip_model' in n) and p.requires_grad],
                    "lr": args.lr_clip,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               (
                                       'T5_model' in n or 'llm' in n) and p.requires_grad],
                    "lr": args.lr_llm,
                },
            ]
        elif args.with_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'visual_projection' not in n and 'clip_model' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               ('visual_projection' in n or 'clip_model' in n) and p.requires_grad],
                    "lr": args.lr_clip,
                },
            ]
        elif args.with_obj_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'obj_visual_projection' not in n and 'clip_model' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               ('obj_visual_projection' in n or 'clip_model' in n) and p.requires_grad],
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
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return optimizer
def build_teacher(args, student_model, device):
    teacher_model, postprocessors = build_model(args)
    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
    for key, value in state_dict.items():
        state_dict[key] = student_state_dict[key].clone().detach()
    teacher_model.load_state_dict(state_dict)
    return teacher_model, postprocessors