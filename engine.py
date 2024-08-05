import math
import os
import random
import sys
from typing import Iterable
import numpy as np
import copy
import itertools
from typing import List
import torch
from collections import defaultdict
import util.misc as utils
from datasets.datasets_gen.hico_eval_triplet import HICOEvaluator as HICOEvaluator_gen
# from datasets.datasets_gen.hico_eval_triplet_ko import HICOEvaluatorKO as HICOEvaluator_gen
from datasets.datasets_gen.vcoco_eval import VCOCOEvaluator as VCOCOEvaluator_gen
import json
import torch.nn.functional as F
from tqdm import tqdm
import datetime
import time
from torch.utils.data import DataLoader
from datasets.vcoco_text_label import vcoco_hoi_text_label
from datasets.hico_text_label import hico_text_label
from util.box_ops import box_cxcywh_to_xyxy
from util.topk import top_k
import gc

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, lr_scheduler=None,
                    gradient_accumulation_steps=1, enable_amp=False, no_training=False, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels') and False:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    elif hasattr(criterion, 'loss_hoi_labels'):
        metric_logger.add_meter('hoi_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    if enable_amp:
        print('\nEnable half precision training\n')

    # scaler = GradScaler()
    # debug
    debug_count = 0
    step = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        if no_training:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items() if k != 'filename' and k != 'raw_img'} for t in targets]
            clip_img = torch.stack([v['clip_inputs'] for v in targets])
            # with autocast():
            obj_feature, hoi_feature, verb_feature = model(samples, clip_input=clip_img, targets=targets)

            metric_logger.update(loss=0)
            if hasattr(criterion, 'loss_labels'):
                metric_logger.update(class_error=0)
            elif hasattr(criterion, 'loss_hoi_labels'):
                metric_logger.update(hoi_class_error=0)
            else:
                metric_logger.update(obj_class_error=0)
            metric_logger.update(lr=0)
            continue

        samples = samples.to(device)
        file_names = [{'filename': i['filename']} for i in targets]
        targets = [{k: v.to(device) for k, v in t.items() if k != 'filename' and k != 'raw_img'} for t in targets]
        for t, f in zip(targets, file_names):
            t.update(f)
        clip_img = torch.stack([v['clip_inputs'] for v in targets])

        outputs = model(samples, clip_input=clip_img, targets=targets)
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        # print(loss_value)
        # sys.exit()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        delay_unscale = (step + 1) % gradient_accumulation_steps != 0
        losses = losses / gradient_accumulation_steps
        if enable_amp:
            raise NotImplementedError
            # with amp.scale_loss(losses, optimizer, delay_unscale=delay_unscale) as scaled_loss:
            #     scaled_loss.backward()
        else:
            losses.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            if max_norm > 0:
                if enable_amp:
                    pass
                    # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        if lr_scheduler:
            lr_scheduler.iter_step()

        step += 1

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels') and False:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        elif hasattr(criterion, 'loss_hoi_labels'):
            if 'hoi_class_error' in loss_dict_reduced:
                metric_logger.update(hoi_class_error=loss_dict_reduced['hoi_class_error'])
            else:
                metric_logger.update(hoi_class_error=-1)
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # trick for generate verb
    if no_training:
        from datasets.static_hico import HOI_IDX_TO_ACT_IDX, HOI_IDX_TO_OBJ_IDX
        hoi_feature = hoi_feature / hoi_feature.norm(dim=1, keepdim=True)
        obj_feature = obj_feature / obj_feature.norm(dim=1, keepdim=True)

        y_verb = [HOI_IDX_TO_ACT_IDX[i] for i in range(600)]
        y_obj = [HOI_IDX_TO_OBJ_IDX[i] for i in range(600)]

        # composite image feature verb + text feature object
        obj_human = []
        for i in range(600):
            obj_human.append(obj_feature[y_obj[i]])
        obj_human = torch.stack(obj_human)
        verb_human = hoi_feature - obj_human

        verb_feature = torch.zeros(117, 512)
        for idx, v in zip(y_verb, verb_human):
            verb_feature[idx] += v

        for i in range(117):
            verb_feature[i] /= y_verb.count(i)

        v_feature = verb_feature / verb_feature.norm(dim=-1, keepdim=True)
        torch.save(v_feature, f'./verb_{args.dataset_file}.pth')
        exit()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
                                                       

@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader,
                 subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    counter = 0

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        clip_img = torch.stack([v['clip_inputs'] for v in targets]).to(device)

        outputs = model(samples, is_training=False, clip_input=clip_img, targets=targets)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        counter += 1
        if counter >= 20 and args.no_training:
            break
        del clip_img
        del samples
        gc.collect()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    # print(preds)
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    """
    For zero-shot enhancement
    args.training_free_enhancement_path is the path to store performance for different hyper-parameter
    """
    root = os.path.join(args.output_dir, args.training_free_enhancement_path)
    if args.training_free_enhancement_path:

        with open(os.path.join(root, 'log.txt'), 'a') as f:
            log = f'\n=========The great hyperparameter tuning begins============\n'
            print(log)
            f.write(log)

        test_pred = copy.deepcopy(preds)

        # testing
        if dataset_file == 'hico':
            evaluator = HICOEvaluator_gen(test_pred, gts, data_loader.dataset.rare_triplets,
                                          data_loader.dataset.non_rare_triplets,
                                          data_loader.dataset.correct_mat, args=args)
        else:
            evaluator = VCOCOEvaluator_gen(preds, gts, data_loader.dataset.correct_mat,
                                           use_nms_filter=args.use_nms_filter)
        stats = evaluator.evaluate()

        text_hoi_feature = model.module.transformer.hoi_cls.to(model.device)
        spatial_feature = torch.cat([i['clip_visual'].unsqueeze(0) for i in preds]).to(model.device)
        spatial_feature /= spatial_feature.norm(dim=-1, keepdim=True)
        spatial_cls = spatial_feature[:, 0, :]  # M, c
        # print(spatial_cls.shape)
        # print(text_hoi_feature.shape)
        cls_scores = spatial_cls @ text_hoi_feature.t()
        with open(os.path.join(root, 'log.txt'), 'a') as f:
            log = f'\n=========Baseline Performance============\n{stats}\n============================\n'
            print(log)
            f.write(log)

        best_performance_1 = 0
        for a in [1]:
            for co in [1.0]:
                for topk in [10]:
                    print(f'current at topk: {topk} as: {a}')
                    test_pred = copy.deepcopy(preds)
                    clip_hoi_score = cls_scores
                    # clip_hoi_score /= (1 + alpha + beta)
                    clip_hoi_score_ori = clip_hoi_score.clone()

                    ignore_idx = clip_hoi_score.sort(descending=True).indices[:, topk:]
                    for idx, igx in enumerate(ignore_idx):
                        clip_hoi_score[idx][igx] *= 0
                    clip_hoi_score = clip_hoi_score.unsqueeze(1)

                    # update logits
                    co=torch.tensor(co).to(model.device)
                    # print(clip_hoi_score[0])
                    # print(co)
                    # print(test_pred[0]['hoi_scores'])
                    for i in range(len(test_pred)):
                        # print(test_pred)
                        # print(co)
                        test_pred[i]['hoi_scores'] += (clip_hoi_score[i].sigmoid() * co).cpu()
                    # testing
                    if dataset_file == 'hico':
                        evaluator = HICOEvaluator_gen(test_pred, gts, data_loader.dataset.rare_triplets,
                                                      data_loader.dataset.non_rare_triplets,
                                                      data_loader.dataset.correct_mat, args=args)

                    else:
                        evaluator = VCOCOEvaluator_gen(test_pred, gts, data_loader.dataset.correct_mat,
                                                       use_nms_filter=args.use_nms_filter)
                    stats = evaluator.evaluate()
                    if dataset_file == 'hico':
                        re_map = stats['mAP']
                    elif dataset_file == 'vcoco':
                        re_map = stats['mAP_all']
                    elif dataset_file == 'hoia':
                        re_map = stats['mAP']
                    else:
                        raise NotImplementedError

                    if best_performance_1 < re_map:
                        best_performance_1 = re_map

                        with open(os.path.join(root, 'log.txt'), 'a') as f:
                            log = f'sigmoid after topk: {topk} as: {a} co: {co}' \
                                  f'\n performance: {stats}\n'
                            print(log)
                            f.write(log)
                        return stats
                        

    if dataset_file == 'hico':
        if args.dataset_root == 'GEN':
            evaluator = HICOEvaluator_gen(preds, gts, data_loader.dataset.rare_triplets,
                                          data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat,
                                          args=args)
    elif dataset_file == 'vcoco':
        if args.dataset_root == 'GEN':
            evaluator = VCOCOEvaluator_gen(preds, gts, data_loader.dataset.correct_mat,
                                           use_nms_filter=args.use_nms_filter)
    else:
        raise NotImplementedError
    start_time = time.time()
    stats = evaluator.evaluate()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time computing mAP: {}'.format(total_time_str))

    return stats
@torch.no_grad()    
def gen_fea(dataset_file, model, postprocessors, data_loader,
                 subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Genfeature:'

    preds = []
    gts = []
    counter = 0
    result={}
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        # clip_img = torch.stack([v['clip_inputs'] for v in targets]).to(device)

        clip_img = torch.stack([v['clip_inputs'] for v in targets]).to(device)

        outputs = model(samples, is_training=False, clip_input=clip_img, targets=targets)
        for i,ta in enumerate(targets):
            # if ta['hoilabel'] in result.keys():
            #     print(ta['hoilabel'])
            # result[ta['hoilabel']]=outputs['encoder'][i].cpu().detach()
            result[ta['hoilabel']]=outputs['encoder'][i]
            # result=result.cpu()
    result_reduced = utils.all_gather(result)
    result1={}
    for mm in result_reduced:
        result1.update(mm)
    for k,v in result1.items():
        result1[k]=v.cpu().detach()
    torch.save(result1, '/public/home/zlj/HOICLIP/genxing_fea.pt')
@torch.no_grad()    
def save_features(dataset_file, model, postprocessors, data_loader,
                 subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Savefeature:'

    preds = []
    gts = []
    counter = 0
    # result_origin_my_cnn=[]
    # result_origin_my_encoder=[]
    # result_origin_my_decoder=[]
    result_origin_hoiclip_cnn=[]
    result_origin_hoiclip_encoder=[]
    result_origin_hoiclip_decoder=[]
    for samples, targets in metric_logger.log_every(data_loader, 10, header):

        samples = samples.to(device)
        # clip_img = torch.stack([v['clip_inputs'] for v in targets]).to(device)

        clip_img = torch.stack([v['clip_inputs'] for v in targets]).to(device)

        outputs = model(samples, is_training=False, clip_input=clip_img, targets=targets)
        
        cnn = torch.mean(outputs['cnn'].cpu().detach(), dim=(-2, -1))
        encoder = torch.mean(outputs['encoder'].cpu().detach().squeeze(dim=1)[:,:1000,:], dim=1)
        decoder = torch.mean(outputs['decoder'].cpu().detach(), dim=1)
        # print(cnn.shape)
        # print(encoder.shape)
        # print(decoder.shape)
        # result_origin_my_cnn.append(cnn)
        # result_origin_my_encoder.append(encoder)
        # result_origin_my_decoder.append(decoder)

        result_origin_hoiclip_cnn.append(cnn)
        result_origin_hoiclip_encoder.append(encoder)
        result_origin_hoiclip_decoder.append(decoder)

        
    # result_reduced = utils.all_gather(result)
    result_origin_hoiclip_cnn_reduced=utils.all_gather(result_origin_hoiclip_cnn)
    result_origin_hoiclip_encoder_reduced=utils.all_gather(result_origin_hoiclip_encoder)
    result_origin_hoiclip_decoder_reduced=utils.all_gather(result_origin_hoiclip_decoder)

    # for k,v in result_origin_my_cnn_reduced.items():
    #     result_origin_my_cnn_reduced[k]=v.cpu().detach()
    # for k,v in result_origin_my_encoder_reduced.items():
    #     result_origin_my_encoder_reduced[k]=v.cpu().detach()
    # for k,v in result_origin_my_decoder_reduced.items():
    #     result_origin_my_decoder_reduced[k]=v.cpu().detach()
    torch.save(result_origin_hoiclip_cnn_reduced, '/public/home/zlj/HOICLIP/result_origin_hoiclip_cnn_reduced.pt')
    torch.save(result_origin_hoiclip_encoder_reduced, '/public/home/zlj/HOICLIP/result_origin_hoiclip_encoder_reduced.pt')
    torch.save(result_origin_hoiclip_decoder_reduced, '/public/home/zlj/HOICLIP/result_origin_hoiclip_decoder_reduced.pt')
    # torch.save(result_origin_my_cnn, '/public/home/zlj/HOICLIP/result_origin_my_cnn.pt')
    # torch.save(result_origin_my_encoder, '/public/home/zlj/HOICLIP/result_origin_my_encoder.pt')
    # torch.save(result_origin_my_decoder, '/public/home/zlj/HOICLIP/result_origin_my_decoder.pt')
def triplet_nms_filter(preds, args, bs, device):
    preds_filtered = []
    for img_preds in preds:
        pred_bboxes = img_preds['predictions']
        pred_hois = img_preds['hoi_prediction']
        all_triplets = {}
        for index, pred_hoi in enumerate(pred_hois):
            triplet = pred_hoi['category_id']

            if triplet not in all_triplets:
                all_triplets[triplet] = {'subs': [], 'objs': [], 'scores': [], 'indexes': []}
            all_triplets[triplet]['subs'].append(pred_bboxes[pred_hoi['subject_id']]['bbox'])
            all_triplets[triplet]['objs'].append(pred_bboxes[pred_hoi['object_id']]['bbox'])
            all_triplets[triplet]['scores'].append(pred_hoi['score'])
            all_triplets[triplet]['indexes'].append(index)

        all_keep_inds = []
        for triplet, values in all_triplets.items():
            subs, objs, scores = values['subs'], values['objs'], values['scores']
            keep_inds = pairwise_nms(np.array(subs), np.array(objs), np.array(scores), args, bs, device)

            # if self.use_score_thres:
            #     sorted_scores = np.array(scores)[keep_inds]
            #     keep_inds = np.array(keep_inds)[sorted_scores > self.thres_score]

            keep_inds = list(np.array(values['indexes'])[keep_inds])
            all_keep_inds.extend(keep_inds)

        preds_filtered.append({
            'filename': img_preds['filename'],
            'predictions': pred_bboxes,
            'hoi_prediction': list(np.array(img_preds['hoi_prediction'])[all_keep_inds])
        })

    return preds_filtered

def pairwise_nms(subs, objs, scores, args, bs, device):
    target_sizes = torch.tensor([512, 512], device=device).expand(subs.shape[0], 2)
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(device)
    # scale_fct = torch.stack(torch.tensor([512, 512, 512, 512]), dim=1).to(scores.device)
    sub_boxes = box_cxcywh_to_xyxy(torch.tensor(subs, device=device))
    sub_boxes = sub_boxes * 512
    obj_boxes = box_cxcywh_to_xyxy(torch.tensor(objs, device=device))
    obj_boxes = obj_boxes * 512
    sub_boxes = sub_boxes.cpu().numpy()
    obj_boxes = obj_boxes.cpu().numpy()
    sx1, sy1, sx2, sy2 = sub_boxes[:, 0], sub_boxes[:, 1], sub_boxes[:, 2], sub_boxes[:, 3]
    ox1, oy1, ox2, oy2 = obj_boxes[:, 0], obj_boxes[:, 1], obj_boxes[:, 2], obj_boxes[:, 3]

    sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
    obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

    order = scores.argsort()[::-1]

    keep_inds = []
    while order.size > 0:
        i = order[0]
        keep_inds.append(i)

        sxx1 = np.maximum(sx1[i], sx1[order[1:]])
        syy1 = np.maximum(sy1[i], sy1[order[1:]])
        sxx2 = np.minimum(sx2[i], sx2[order[1:]])
        syy2 = np.minimum(sy2[i], sy2[order[1:]])

        sw = np.maximum(0.0, sxx2 - sxx1 + 1)
        sh = np.maximum(0.0, syy2 - syy1 + 1)
        sub_inter = sw * sh
        sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

        oxx1 = np.maximum(ox1[i], ox1[order[1:]])
        oyy1 = np.maximum(oy1[i], oy1[order[1:]])
        oxx2 = np.minimum(ox2[i], ox2[order[1:]])
        oyy2 = np.minimum(oy2[i], oy2[order[1:]])

        ow = np.maximum(0.0, oxx2 - oxx1 + 1)
        oh = np.maximum(0.0, oyy2 - oyy1 + 1)
        obj_inter = ow * oh
        obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

        ovr = np.power(sub_inter / sub_union, args.nms_alpha) * np.power(obj_inter / obj_union, args.nms_beta)
        # inds = np.where(ovr <= args.thres_nms)[0]
        inds = np.where(ovr <= 0.5)[0]

        order = order[inds + 1]
    return keep_inds


def get_pseudo_labels(outputs, target, thresholds, h, w, postprocessors, label_leng, correct_mat, args,
                      nms_threshold=0.7):
    preds = []
    bs = outputs['pred_obj_logits'].shape[0]
    device = outputs['pred_obj_logits'].device
    orig_target_sizes = torch.tensor([h, w], device=device).expand(bs, 2)
    # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    results = postprocessors['hoi'](outputs, orig_target_sizes, pseudo_labels=True)
    # preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
    preds = results
    # print(preds)
    ans = []
    
    if args.dataset_file=='hico':

        hico_triplet_labels = list(hico_text_label.keys())
    else:
        hico_triplet_labels = list(vcoco_hoi_text_label.keys())
    hoi_obj_list = []
    for hoi_pair in hico_triplet_labels:
        hoi_obj_list.append(hoi_pair[1])


    for index, img_preds in enumerate(preds):
        img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
        bboxes = [{'bbox': list(bbox)} for bbox in img_preds['boxes']]
        obj_scores = img_preds['obj_scores'] *  img_preds['obj_scores']
        hoi_scores = img_preds['hoi_scores'] + obj_scores[:, hoi_obj_list]

        hoi_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
        subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
        object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

        hoi_scores = hoi_scores.ravel()
        hoi_labels = hoi_labels.ravel()
        subject_ids = subject_ids.ravel()
        object_ids = object_ids.ravel()

        topk_hoi_scores = top_k(list(hoi_scores), 100)
        topk_indexes = np.array([np.where(hoi_scores == score)[0][0] for score in topk_hoi_scores])

        if len(subject_ids) > 0:
            hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score}
                    for
                    subject_id, object_id, category_id, score in
                    zip(subject_ids[topk_indexes], object_ids[topk_indexes], hoi_labels[topk_indexes], topk_hoi_scores)]
            hois = hois[:100]
        else:
            hois = []


   
        predsdict = []
        predsdict.append({
            'filename': target[index]['filename'],
            'predictions': bboxes,
            'hoi_prediction': hois
        })
        predsdict = triplet_nms_filter(predsdict, args, bs, device)
        hois = predsdict[0]['hoi_prediction']
        hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        # print(hois)
        indic = []
        jihao = True
        for i, hoi in enumerate(hois):
            if hoi['category_id'] == torch.argmax(target[index]['hoi_labels'], dim=1) and jihao:
                indic.append(i)
                jihao = False
            if hoi['score'] >= thresholds[hoi['category_id']]:
                indic.append(i)
        if len(indic) == 0:
            indic.append(0)
        temp = {}
        verb = []
        labels = []
        boxes = []
        obj_labels = []
        obj_boxes = []
        sub_boxes = []
        verb_score = []
        j = 0
        for i in indic:
            # for i in range(label_leng):
            sub_id = hois[i]['subject_id']
            obj_id = hois[i]['object_id']
            verb_id = hois[i]['category_id']
            verb_score.append(torch.tensor(hois[i]['score']).unsqueeze(0))
            # verb_score=hois[i]['score']
            verb.append(F.one_hot(torch.tensor(verb_id, device=device), num_classes=args.num_classes).unsqueeze(0))
            boxes.append(torch.tensor(img_preds['boxes'][sub_id], device=device).unsqueeze(0))
            boxes.append(torch.tensor(img_preds['boxes'][obj_id], device=device).unsqueeze(0))
            # labels.append(torch.tensor(img_preds['labels'][sub_id],device=device))
            # labels.append(torch.tensor(img_preds['labels'][obj_id],device=device))
            labels.append(img_preds['labels'][sub_id])
            labels.append(img_preds['labels'][obj_id])
            # obj_labels.append(torch.tensor(img_preds['labels'][obj_id],device=device))
            obj_labels.append(img_preds['labels'][obj_id])
            obj_boxes.append(torch.tensor(img_preds['boxes'][obj_id], device=device).unsqueeze(0))
            sub_boxes.append(torch.tensor(img_preds['boxes'][sub_id], device=device).unsqueeze(0))
            j = j + 2
        # temp['verb_score'] = torch.cat(verb_score).numpy()
        # temp['verb_labels'] = torch.cat(verb).to(dtype=torch.float32)
        temp['hoi_score'] = torch.cat(verb_score).numpy()
        temp['hoi_labels'] = torch.cat(verb).to(dtype=torch.float32)
        temp['boxes'] = torch.cat(boxes)
        temp['labels'] = torch.tensor(labels, device=device)
        temp['obj_labels'] = torch.tensor(obj_labels, device=device)
        temp['obj_boxes'] = torch.cat(obj_boxes)
        temp['sub_boxes'] = torch.cat(sub_boxes)
        temp['file_name'] = target[index]['filename']
        temp['clip_inputs'] = target[index]['clip_inputs'] 
        ans.append(temp)
    return ans

def get_pseudo_labels_vcoco(outputs, target, thresholds, h, w, postprocessors, label_leng, correct_mat, args,
                      nms_threshold=0.7):
    preds = []
    bs = outputs['pred_obj_logits'].shape[0]
    device = outputs['pred_obj_logits'].device
    orig_target_sizes = torch.tensor([h, w], device=device).expand(bs, 2)
    # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    results = postprocessors['hoi'](outputs, orig_target_sizes, pseudo_labels=True)
    # preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
    preds = results
    # print(preds)
    ans = []
    
    hico_triplet_labels = list(hico_text_label.keys())
    hoi_obj_list = []
    for hoi_pair in hico_triplet_labels:
        hoi_obj_list.append(hoi_pair[1])


    for index, img_preds in enumerate(preds):
        img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
        bboxes = [{'bbox': list(bbox)} for bbox in img_preds['boxes']]
        obj_scores = img_preds['obj_scores'] *  img_preds['obj_scores']
        hoi_scores = img_preds['hoi_scores'] + obj_scores[:, hoi_obj_list]

        hoi_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
        subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
        object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

        hoi_scores = hoi_scores.ravel()
        hoi_labels = hoi_labels.ravel()
        subject_ids = subject_ids.ravel()
        object_ids = object_ids.ravel()

        topk_hoi_scores = top_k(list(hoi_scores), 100)
        topk_indexes = np.array([np.where(hoi_scores == score)[0][0] for score in topk_hoi_scores])

        if len(subject_ids) > 0:
            hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score}
                    for
                    subject_id, object_id, category_id, score in
                    zip(subject_ids[topk_indexes], object_ids[topk_indexes], hoi_labels[topk_indexes], topk_hoi_scores)]
            hois = hois[:100]
        else:
            hois = []


        predsdict = []
        predsdict.append({
            'filename': target[index]['filename'],
            'predictions': bboxes,
            'hoi_prediction': hois
        })
        # predsdict = triplet_nms_filter(predsdict, args, bs, device)
        hois = predsdict[0]['hoi_prediction']
        hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        # print(hois)
        indic = []
        jihao = True
        for i, hoi in enumerate(hois):
            if hoi['category_id'] == torch.argmax(target[index]['hoi_labels'], dim=1) and jihao:
                indic.append(i)
                jihao = False
            if hoi['score'] >= thresholds[hoi['category_id']]:
                indic.append(i)
        if len(indic) == 0:
            indic.append(0)
        temp = {}
        verb = []
        labels = []
        boxes = []
        obj_labels = []
        obj_boxes = []
        sub_boxes = []
        verb_score = []
        j = 0
        for i in indic:
            # for i in range(label_leng):
            sub_id = hois[i]['subject_id']
            obj_id = hois[i]['object_id']
            verb_id = hois[i]['category_id']
            verb_score.append(torch.tensor(hois[i]['score']).unsqueeze(0))
            # verb_score=hois[i]['score']
            verb.append(F.one_hot(torch.tensor(verb_id, device=device), num_classes=600).unsqueeze(0))
            boxes.append(torch.tensor(img_preds['boxes'][sub_id], device=device).unsqueeze(0))
            boxes.append(torch.tensor(img_preds['boxes'][obj_id], device=device).unsqueeze(0))
            # labels.append(torch.tensor(img_preds['labels'][sub_id],device=device))
            # labels.append(torch.tensor(img_preds['labels'][obj_id],device=device))
            labels.append(img_preds['labels'][sub_id])
            labels.append(img_preds['labels'][obj_id])
            # obj_labels.append(torch.tensor(img_preds['labels'][obj_id],device=device))
            obj_labels.append(img_preds['labels'][obj_id])
            obj_boxes.append(torch.tensor(img_preds['boxes'][obj_id], device=device).unsqueeze(0))
            sub_boxes.append(torch.tensor(img_preds['boxes'][sub_id], device=device).unsqueeze(0))
            j = j + 2
        # temp['verb_score'] = torch.cat(verb_score).numpy()
        # temp['verb_labels'] = torch.cat(verb).to(dtype=torch.float32)
        temp['hoi_score'] = torch.cat(verb_score).numpy()
        temp['hoi_labels'] = torch.cat(verb).to(dtype=torch.float32)
        temp['boxes'] = torch.cat(boxes)
        temp['labels'] = torch.tensor(labels, device=device)
        temp['obj_labels'] = torch.tensor(obj_labels, device=device)
        temp['obj_boxes'] = torch.cat(obj_boxes)
        temp['sub_boxes'] = torch.cat(sub_boxes)
        temp['file_name'] = target[index]['filename']
        temp['clip_inputs'] = target[index]['clip_inputs'] 
        ans.append(temp)
    return ans

def train_one_epoch_teaching(student_model: torch.nn.Module,
                             teacher_model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             criterion_pseudo: torch.nn.Module,
                             source_loader: DataLoader,
                             target_loader: DataLoader,
                             optimizer: torch.optim.Optimizer,
                             thresholds: List[float],
                             coef_target: float,
                             mask_ratio: float,
                             alpha_ema: float,
                             device: torch.device,
                             epoch: int,
                             image_size,
                             postprocessors,
                             correct_mat,
                             args,
                             enable_mae: bool = False,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True):
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    criterion.train()
    criterion_pseudo.train()
    source_loader = iter(source_loader)
    target_loader = iter(target_loader)
    # source_sample, source_annotations = source_loader.next()
    # target_sample, _ = target_loader.next()

    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    # Training data statistics
    epoch_source_loss_dict = defaultdict(float)
    epoch_target_loss_dict = defaultdict(float)
    print("source_loader:", len(source_loader))
    print("target_loader:", len(target_loader))
    total_iters = min(len(source_loader), len(target_loader))
    vis = []
    for i in range(total_iters):
        # Data pre-fetch
        source_sample, source_annotations = next(source_loader)
        # target_student_images, target_annotations1 = next(target_loader)
        # target_teacher_images=target_student_images
        target_student_images,target_teacher_images, target_annotations1 = next(target_loader)
        
        source_sample = source_sample.to(device)
        target_teacher_images = target_teacher_images.to(device)
        target_student_images = target_student_images.to(device)
        file_names = [{'filename': i['filename']} for i in source_annotations]
        source_annotations = [{k: v.to(device) for k, v in t.items() if k != 'filename' and k != 'raw_img'} for t in source_annotations]
        for t, f in zip(source_annotations, file_names):
            t.update(f)
        # source_annotations = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in source_annotations]
        target_annotations = []
        for t in target_annotations1:
            temp = {}
            for k, v in t.items():
                if k != 'filename':
                    temp[k] = v.to(device)
                else:
                    temp[k] = v
            target_annotations.append(temp)
        # target_annotations= [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in target_annotations]
        clip_img = torch.stack([v['clip_inputs'] for v in source_annotations])
        clip_img_teach = torch.stack([v['clip_inputs'] for v in target_annotations])
        # print(clip_img.shape)
        # print(clip_img_teach.shape)
        # Source forward
        
        source_out = student_model(source_sample, clip_input=clip_img)
        source_loss, source_loss_dict = criterion(source_out, source_annotations, domain_label=0)
        # source_loss, source_loss_dict = criterion(source_out, source_annotations)
        
        source_weight_dict = criterion.weight_dict
        target_weight_dict = criterion_pseudo.weight_dict
        # Target teacher forward
        with torch.no_grad():
            # teacher_out = teacher_model(target_teacher_images)
            teacher_out = teacher_model(target_teacher_images, clip_input=clip_img_teach)

            pseudo_labels = get_pseudo_labels(teacher_out, target_annotations, thresholds, image_size[0], image_size[1],
                                              postprocessors=postprocessors, label_leng=1, correct_mat=correct_mat,
                                              args=args)
            for pseudo_label in pseudo_labels:
                tempp = {}
                # print(pseudo_label['boxes'].shape)
                tempp['boxes'] = pseudo_label['boxes'].cpu().numpy().tolist()
                tempp['labels'] = pseudo_label['labels'].cpu().numpy().tolist()
                tempp['file_name'] = pseudo_label['file_name']
                tempp['score'] = pseudo_label['hoi_score'].tolist()
                # print((tempp))
                vis.append(tempp)
            # print([i['verb_score'] for i in pseudo_labels])
        # Target student forward
        target_student_out = student_model(target_student_images, targets=target_annotations,clip_input=clip_img_teach, enable_mae=enable_mae, mask_ratio=mask_ratio)
        # target_student_out = student_model(target_student_images)
        # print(len(target_student_out))

        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels, domain_label=1,
                                                         enable_mae=enable_mae)
        # target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels, domain_label=1)
        # target_loss, target_loss_dict =criterion_pseudo(target_student_out, pseudo_labels,enable_mae=enable_mae)
        # Backward
        optimizer.zero_grad()
        # print(source_loss)
        # print(target_loss)
        loss = source_loss + coef_target * target_loss
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()
        # Record epoch losses
        epoch_loss += loss.item()
        # update loss_dict
        for k, v in source_loss_dict.items():
            if k in source_weight_dict:
                epoch_source_loss_dict[k] += v.detach().cpu().item()
        epoch_source_loss_dict['source_loss'] += source_loss.detach().cpu().item()
        for k, v in target_loss_dict.items():
            if k in target_weight_dict:
                epoch_target_loss_dict[k] += v.detach().cpu().item()
        epoch_target_loss_dict['target_loss'] += target_loss.detach().cpu().item()
        # EMA update teacher
        with torch.no_grad():
            state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
            for key, value in state_dict.items():
                state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
            teacher_model.load_state_dict(state_dict)
        # Data pre-fetch
        # source_sample, source_annotations = source_loader.next()
        # target_sample, _ = target_loader.next()
        # if target_sample is not None:
        #     target_teacher_images, target_student_images = target_sample, target_sample
        # Log
        if utils.is_main_process() and (i + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    # Final process of loss dict
    # JSON 文件路径
    # json_file_path = '/home/zlj/CDN_MRT/data/vis_json/vis_{}.json'.format(epoch)
    # utils.all_gather(vis)
    # # 将列表保存到 JSON 文件
    # if utils.is_main_process():
    #     with open(json_file_path, 'w') as json_file:
    #         json.dump(vis, json_file)
    epoch_loss /= total_iters
    for k, v in epoch_source_loss_dict.items():
        epoch_source_loss_dict[k] /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_source_loss_dict, epoch_target_loss_dict, vis

