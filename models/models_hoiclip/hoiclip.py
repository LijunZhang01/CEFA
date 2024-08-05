import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import numpy as np
from ModifiedCLIP import clip
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
from datasets.vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label
from datasets.static_hico import HOI_IDX_TO_ACT_IDX

from ..backbone import build_backbone
from ..matcher import build_matcher
from .gen import build_gen
from .gcn import GraphConvolution
from .transformer_encoder import graphtiencoder
import copy

def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eta=1.0):
        ctx.eta = eta
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.eta), None


def grad_reverse(x, eta=1.0):
    return GradReverse.apply(x, eta)


class HOICLIP(nn.Module):
    def __init__(self, backbone, transformer, num_queries, aux_loss=False, args=None):
        super().__init__()

        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed_h = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_o = nn.Embedding(num_queries, hidden_dim)
        self.pos_guided_embedd = nn.Embedding(num_queries, hidden_dim)
        self.hum_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.inter2verb = MLP(args.clip_embed_dim, args.clip_embed_dim // 2, args.clip_embed_dim, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.dec_layers = self.args.dec_layers

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_model, self.preprocess = clip.load(self.args.clip_model)

        if self.args.dataset_file == 'hico':
            hoi_text_label = hico_text_label
            obj_text_label = hico_obj_text_label
            unseen_index = hico_unseen_index
        elif self.args.dataset_file == 'vcoco':
            hoi_text_label = vcoco_hoi_text_label
            obj_text_label = vcoco_obj_text_label
            unseen_index = None

        clip_label, obj_clip_label, v_linear_proj_weight, hoi_text, obj_text, train_clip_label = \
            self.init_classifier_with_CLIP(hoi_text_label, obj_text_label, unseen_index, args.no_clip_cls_init)
        num_obj_classes = len(obj_text) - 1  # del nothing
        self.clip_visual_proj = v_linear_proj_weight

        self.hoi_class_fc = nn.Sequential(
            nn.Linear(hidden_dim, args.clip_embed_dim),
            nn.LayerNorm(args.clip_embed_dim),
        )

        if unseen_index:
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
        else:
            unseen_index_list = []

        if self.args.dataset_file == 'hico':
            verb2hoi_proj = torch.zeros(117, 600)
            select_idx = list(set([i for i in range(600)]) - set(unseen_index_list))
            for idx, v in enumerate(HOI_IDX_TO_ACT_IDX):
                verb2hoi_proj[v][idx] = 1.0
            self.verb2hoi_proj = nn.Parameter(verb2hoi_proj[:, select_idx], requires_grad=False)
            self.verb2hoi_proj_eval = nn.Parameter(verb2hoi_proj, requires_grad=False)

            self.verb_projection = nn.Linear(args.clip_embed_dim, 117, bias=False)
            self.verb_projection.weight.data = torch.load(args.verb_pth, map_location='cpu')
            self.verb_weight = args.verb_weight
        else:
            verb2hoi_proj = torch.zeros(29, 263)
            for i in vcoco_hoi_text_label.keys():
                verb2hoi_proj[i[0]][i[1]] = 1

            self.verb2hoi_proj = nn.Parameter(verb2hoi_proj, requires_grad=False)
            self.verb_projection = nn.Linear(args.clip_embed_dim, 29, bias=False)
            self.verb_projection.weight.data = torch.load(args.verb_pth, map_location='cpu')
            self.verb_weight = args.verb_weight

        if args.with_clip_label:
            if args.fix_clip_label:
                self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text), bias=False)
                self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)
                for i in self.visual_projection.parameters():
                    i.require_grads = False
            else:
                self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text))
                self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)

            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default':
                self.eval_visual_projection = nn.Linear(args.clip_embed_dim, 600, bias=False)
                self.eval_visual_projection.weight.data = clip_label / clip_label.norm(dim=-1, keepdim=True)
        else:
            self.hoi_class_embedding = nn.Linear(args.clip_embed_dim, len(hoi_text))

        if args.with_obj_clip_label:
            self.obj_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, args.clip_embed_dim),
                nn.LayerNorm(args.clip_embed_dim),
            )
            if args.fix_clip_label:
                self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1, bias=False)
                self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
                for i in self.obj_visual_projection.parameters():
                    i.require_grads = False
            else:
                self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1)
                self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
        else:
            self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)

        self.transformer.hoi_cls = clip_label / clip_label.norm(dim=-1, keepdim=True)

        self.hidden_dim = hidden_dim
        self.reset_parameters()
        self.domain_pred_bac, self.domain_pred_enc, self.domain_pred_dec = None, None, None
        self.args=args

    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embedd.weight)

    def init_classifier_with_CLIP(self, hoi_text_label, obj_text_label, unseen_index, no_clip_cls_init=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_inputs = torch.cat([clip.tokenize(hoi_text_label[id]) for id in hoi_text_label.keys()])
        if self.args.del_unseen and unseen_index is not None:
            hoi_text_label_del = {}
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
            for idx, k in enumerate(hoi_text_label.keys()):
                if idx in unseen_index_list:
                    continue
                else:
                    hoi_text_label_del[k] = hoi_text_label[k]
        else:
            hoi_text_label_del = hoi_text_label.copy()
        text_inputs_del = torch.cat(
            [clip.tokenize(hoi_text_label[id]) for id in hoi_text_label_del.keys()])

        obj_text_inputs = torch.cat([clip.tokenize(obj_text[1]) for obj_text in obj_text_label])
        clip_model = self.clip_model
        clip_model.to(device)
        with torch.no_grad():
            text_embedding = clip_model.encode_text(text_inputs.to(device))
            text_embedding_del = clip_model.encode_text(text_inputs_del.to(device))
            obj_text_embedding = clip_model.encode_text(obj_text_inputs.to(device))
            v_linear_proj_weight = clip_model.visual.proj.detach()

        if not no_clip_cls_init:
            print('\nuse clip text encoder to init classifier weight\n')
            return text_embedding.float(), obj_text_embedding.float(), v_linear_proj_weight.float(), \
                   hoi_text_label_del, obj_text_inputs, text_embedding_del.float()
        else:
            print('\nnot use clip text encoder to init classifier weight\n')
            return torch.randn_like(text_embedding.float()), torch.randn_like(
                obj_text_embedding.float()), torch.randn_like(v_linear_proj_weight.float()), \
                   hoi_text_label_del, obj_text_inputs, torch.randn_like(text_embedding_del.float())
    @staticmethod
    def get_mask_list(mask_list, mask_ratio):
    # def get_mask_list(mask_list, mask_ratio,maskpt):
        mae_mask_list = copy.deepcopy(mask_list)
        mae_mask_list = torch.rand(mask_list.shape).to(mask_list.device) < mask_ratio
        return mae_mask_list

        # maskpt1 = copy.deepcopy(maskpt)
        # maskpt1 = torch.rand(maskpt.shape).to(maskpt.device) < mask_ratio
        # for i in range(len(mask_list)):
        #     for j in range(len(mask_list[0])):
        #         for k in range(len(mask_list[1])):
        #             if j*len(mask_list[0])+k in maskpt[i] and maskpt1[i][maskpt[i].tolist().index(j*len(mask_list[0])+k)]:
        #                 mask_list[i][j][k]=True   
        # return mask_list
        
    @staticmethod
    def get_mask_list1(mask_list, mask_ratio):
        mask = torch.zeros_like(mask_list)
        center_x = mask_list.shape[1] // 2
        center_y = mask_list.shape[2] // 2
        max_distance=torch.sqrt(torch.pow(0 - center_x, torch.tensor(2)) + torch.pow(0 - center_y, torch.tensor(2)))
        temp=[max_distance/3,max_distance/3*2,max_distance/3*3]
        # 计算每个像素点到中心点的距离
        for i in range(mask_list.shape[1]):
            for j in range(mask_list.shape[2]):
                distance = torch.sqrt(torch.pow(i - center_x, torch.tensor(2)) + torch.pow(j - center_y, torch.tensor(2)))
                if distance>=0 and distance<temp[0]:
                    mask[:, i, j] = torch.rand(1) < 0.7
                elif distance>=temp[0] and distance<temp[1]:
                    mask[:, i, j] = torch.rand(1) < 0.8
                elif distance>=temp[1] and distance<=temp[2]:
                    mask[:, i, j] = torch.rand(1) < 0.9
                # elif distance>=temp[2] and distance<=temp[3]:
                #     mask[:, i, j] = torch.rand(1) < 0.7
        # mae_mask_list = copy.deepcopy(mask_list)
        # mae_mask_list = torch.rand(mask_list.shape).to(mask_list.device) < mask_ratio
        mae_mask_list = mask.bool().to(mask_list.device)
        return mae_mask_list

    def build_discriminators(self, device):
        if self.domain_pred_bac is None and self.domain_pred_enc is None and self.domain_pred_dec is None:
            self.domain_pred_bac = MultiConv2d(self.backbone.num_channels, self.hidden_dim, 2, 3)
            self.domain_pred_bac.to(device)
            self.domain_pred_enc = MultiConv2d(self.hidden_dim, self.hidden_dim, 2, 3)
            self.domain_pred_enc.to(device)
            # print(self.hidden_dim)
            self.domain_pred_dec = MLP(self.hidden_dim*2, self.hidden_dim*2, 2, 3)
            self.domain_pred_dec.to(device)
            self.domain_prototype_dec_gcn = GraphConvolution(self.hidden_dim*2, self.hidden_dim*2)
            self.domain_prototype_dec_gcn.to(device)
            # self.graphtiencoder = graphtiencoder(
            #                             hidden_dim=self.hidden_dim*2,
            #                             feedforward_dim=1024,
            #                             num_heads=8,
            #                             dropout=0.1,
            #                             activation="relu",
            #                             total_spatial_shapes=[64,self.hidden_dim*2],
            #                     )
            # self.graphtiencoder.to(device)
    def discriminator_forward(self, features, inter_memory, inter_object_query,indince):
        def apply_dis(memory, discriminator):
            return discriminator(grad_reverse(memory))

        # Conv discriminator
        outputs_domains_bac = apply_dis(features[-1], self.domain_pred_bac).permute(0, 2, 3, 1)

        outputs_domains_enc = []
        b, _, h, w = features[-1].shape
        lvl_domains_enc = []
        for hda_idx in range(inter_memory.shape[1]):
            lvl_inter_memory = inter_memory[:, hda_idx, :, :] \
                .transpose(1, 2).reshape(b, -1, h, w)  # (b, c, h, w)
            lvl_hda_domains_enc = apply_dis(lvl_inter_memory, self.domain_pred_enc)  # (b, 2, h, w)
            lvl_hda_domains_enc = lvl_hda_domains_enc.reshape(b, 2, h * w).transpose(1, 2)  # (b, h * w, 2)
            lvl_domains_enc.append(lvl_hda_domains_enc)
        outputs_domains_enc.append(torch.stack(lvl_domains_enc, dim=1))  # (b, hda, h * w, 2)

        outputs_domains_enc = torch.cat(outputs_domains_enc, dim=2)
        # outputs_domains_dec=0
        # outputs_domains_bac=0
        # outputs_domains_enc=0
        # ----------------------------------------------------------------------------------------
        # decoder_outputs=inter_object_query[:,-1,:,:]
        # mask_flatten=torch.zeros((decoder_outputs.shape[0],decoder_outputs.shape[1]),device=decoder_outputs.device)
        # decoder_outputs=decoder_outputs.transpose(0,1)
        # graph_output=self.graphtiencoder(decoder_outputs,mask_flatten).transpose(0,1)
        # decoder_outputs_prototype=torch.stack([graph_output[i].index_select(0,indince[i]).clone() for i in range(len(indince))],dim=0).unsqueeze(1)
        # outputs_domains_dec = apply_dis(decoder_outputs_prototype, self.domain_pred_dec)
        # ----------------------------------------------------------------------------------------
        decoder_outputs=inter_object_query[:,-1,:,:]
        adj_mask = torch.zeros(decoder_outputs.shape[0], decoder_outputs.shape[1], decoder_outputs.shape[1],
                                             device=inter_memory.device)
        for i in range(len(indince)):
            # 消融
            adj_mask[i][indince[i]]=1
            # 消融
            adj_mask[i][:,indince[i]]=1
            # 消融
            # adj_mask[i]=1
        graph_output = self.domain_prototype_dec_gcn(decoder_outputs, adj_mask).transpose(1,2)
        decoder_outputs_prototype=torch.stack([graph_output[i].index_select(0,indince[i]).clone() for i in range(len(indince))],dim=0).unsqueeze(1)
        outputs_domains_dec = apply_dis(decoder_outputs_prototype, self.domain_pred_dec)
        # outputs_domains_dec = apply_dis(inter_object_query, self.domain_pred_dec)
        return outputs_domains_bac, outputs_domains_enc, outputs_domains_dec


    def forward(self, samples: NestedTensor, is_training=True, clip_input=None, targets=None, enable_mae=False, mask_ratio=0.8):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        mm=self.input_proj(src)
        h_hs, o_hs, inter_hs, inter_memory,inter_hs_oral,clip_cls_feature, clip_hoi_score, clip_visual = self.transformer(mm, mask,
                                                self.query_embed_h.weight,
                                                self.query_embed_o.weight,
                                                self.pos_guided_embedd.weight,
                                                pos[-1], self.clip_model, self.clip_visual_proj, clip_input)
        
        eno=inter_hs
        # print(eno.shape)
        # if self.args.mode=='eval':
        #     torch.save(inter_memory, 'oral_encoder.pt')
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
            # inter_hs = self.hoi_class_fc(inter_hs)
            outputs_inter_hs = inter_hs.clone()
            verb_hs = self.inter2verb(inter_hs)
            inter_hs = inter_hs / inter_hs.norm(dim=-1, keepdim=True)
            verb_hs = verb_hs / verb_hs.norm(dim=-1, keepdim=True)
            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default' \
                    and (self.args.eval or not is_training):
                outputs_hoi_class = logit_scale * self.eval_visual_projection(inter_hs)
                outputs_verb_class = logit_scale * self.verb_projection(verb_hs) @ self.verb2hoi_proj_eval
                outputs_hoi_class = outputs_hoi_class + outputs_verb_class * self.verb_weight
            else:
                outputs_hoi_class = logit_scale * self.visual_projection(inter_hs)
                outputs_verb_class = logit_scale * self.verb_projection(verb_hs) @ self.verb2hoi_proj
                outputs_hoi_class = outputs_hoi_class + outputs_verb_class * self.verb_weight
        else:
            inter_hs = self.hoi_class_fc(inter_hs)
            outputs_inter_hs = inter_hs.clone()
            outputs_hoi_class = self.hoi_class_embedding(inter_hs)
        
        out = {'pred_hoi_logits': outputs_hoi_class[-1], 'pred_obj_logits': outputs_obj_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1], 'clip_visual': clip_visual,
               'clip_cls_feature': clip_cls_feature, 'hoi_feature': inter_hs[-1], 'clip_logits': clip_hoi_score}
        # out['encoder']=inter_memory.squeeze(1)[:,0,:]
        out['encoder']=inter_memory
        # out['decoder']=inter_hs_oral[:,0,:]
        out['decoder']=inter_hs_oral
        out['cnn']=mm
        # out['resnet']=src
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
        # MAE branch
        if enable_mae:
            assert self.transformer.mae_decoder is not None
            # maskp=[]
            # for i in targets:
            #     maskp.append(i['mask_patch'])
            # maskpt=torch.stack(maskp)
            # mae_mask_list = self.get_mask_list(mask,mask_ratio,maskpt)
            mae_mask_list = self.get_mask_list(mask,mask_ratio)
            mae_src_list = self.input_proj(src)
            mae_output = self.transformer(mae_src_list, mae_mask_list,
                                        self.query_embed_h.weight,
                                        self.query_embed_o.weight,
                                        self.pos_guided_embedd.weight,
                                        pos[-1], self.clip_model, self.clip_visual_proj, clip_input,enable_mae=enable_mae,target=targets
                                        )
            output_tensor = torch.zeros((2, 512, 512), dtype=torch.bool,device=mae_mask_list.device)
            for i in range(mae_mask_list.shape[0]):
                for j in range(mae_mask_list.shape[1]):
                    for k in range(mae_mask_list.shape[2]):
                        # 获取当前元素的值
                        value = mae_mask_list[i, j, k]
                        # 在输出张量中对应位置扩展为32x32
                        e1=int(512/mae_mask_list.shape[1])
                        e2=int(512/mae_mask_list.shape[2])
                        output_tensor[i, j*e1:(j+1)*e1, k*e2:(k+1)*e2] = value
            out['features'] = [src.detach()]
            out['mae_output'] = mae_output
            out['img'] = samples.decompose()[0]
            out['mask']=output_tensor
            
        # Discriminators
        # print(inter_memory.shape)
        
        if self.domain_pred_bac is not None and self.domain_pred_enc is not None and self.domain_pred_dec is not None:
        # if self.domain_pred_dec is not None:
            decoder_query=torch.max(outputs_hoi_class[-1],dim=2)[0]
            values, indince = torch.topk(decoder_query, 6, dim=1)
            # indince=torch.tensor([1,2,3,4,5,6],device=inter_memory.device).expand(inter_memory.shape[0],6)
            outputs_domains_bac, outputs_domains_enc, outputs_domains_dec = self.discriminator_forward(
                [src], inter_memory, eno.transpose(0, 1)[:, -1, :, :].unsqueeze(1),indince
            )
            out['domain_bac_all'] = outputs_domains_bac
            out['domain_enc_all'] = outputs_domains_enc
            out['domain_dec_all'] = outputs_domains_dec
        return out

    @torch.jit.unused
    def _set_aux_loss_triplet(self, outputs_hoi_class, outputs_obj_class,
                              outputs_sub_coord, outputs_obj_coord, outputs_inter_hs=None):

        if outputs_hoi_class.shape[0] == 1:
            outputs_hoi_class = outputs_hoi_class.repeat(self.dec_layers, 1, 1, 1)
        aux_outputs = {'pred_hoi_logits': outputs_hoi_class[-self.dec_layers: -1],
                       'pred_obj_logits': outputs_obj_class[-self.dec_layers: -1],
                       'pred_sub_boxes': outputs_sub_coord[-self.dec_layers: -1],
                       'pred_obj_boxes': outputs_obj_coord[-self.dec_layers: -1],
                       }
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


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.with_mimic:
            self.clip_model, _ = clip.load(args.clip_model, device=device)
        else:
            self.clip_model = None
        self.alpha = args.alpha
        self.with_rec_loss = args.with_rec_loss
        self.with_mimic=args.with_mimic

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, domain_label=None, log=True):
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
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions, domain_label=None):
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

    def loss_hoi_labels(self, outputs, targets, indices, num_interactions, domain_label=None, topk=5):
        assert 'pred_hoi_logits' in outputs
        src_logits = outputs['pred_hoi_logits']
        dtype = src_logits.dtype

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)]).to(dtype)
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

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions, domain_label=None):
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

    def mimic_loss(self, outputs, targets, indices, num_interactions, domain_label=None):
        src_feats = outputs['inter_memory']
        src_feats = torch.mean(src_feats, dim=1)

        target_clip_inputs = torch.cat([t['clip_inputs'].unsqueeze(0) for t in targets])
        with torch.no_grad():
            target_clip_feats = self.clip_model.encode_image(target_clip_inputs)
        loss_feat_mimic = F.l1_loss(src_feats, target_clip_feats)
        losses = {'loss_feat_mimic': loss_feat_mimic}
        return losses
    def reconstruction_loss(self, outputs, targets, indices, num_interactions, domain_label=None):
        raw_feature = outputs['clip_cls_feature']
        hoi_feature = outputs['hoi_feature']

        loss_rec = F.l1_loss(raw_feature, hoi_feature)
        return {'loss_rec': loss_rec}
    def loss_domains(self, out, targets, indices, num_interactions,domain_label):
        domain_pred_bac = out['domain_bac_all']
        domain_pred_enc = out['domain_enc_all']  # (hda, batch_size, len_enc+1, 2)
        domain_pred_dec = out['domain_dec_all']  # (hda, batch_size, len_enc+1, 2)
        batch_size, len_hda, len_enc, len_domain = domain_pred_enc.shape
        batch_size, len_hda, len_dec, len_domain = domain_pred_dec.shape
        # Permute
        domain_pred_bac = domain_pred_bac.permute(0, 3, 1, 2)
        domain_pred_enc = domain_pred_enc.permute(0, 3, 1, 2)
        domain_pred_dec = domain_pred_dec.permute(0, 3, 1, 2)
        # Generate domain label for domain_pred_enc_token, domain_pred_dec_token, and domain_query
        domain_label = torch.tensor(domain_label, dtype=torch.long, device=domain_pred_enc.device)  # (batch_size,)
        # domain_label = torch.tensor(domain_label, dtype=torch.long, device=domain_pred_dec.device) 
        domain_label_bac = domain_label.expand(batch_size, domain_pred_bac.shape[2], domain_pred_bac.shape[3])
        domain_label_enc = domain_label.expand(batch_size, len_hda, len_enc)
        domain_label_dec = domain_label.expand(batch_size, len_hda, len_dec)
        # cross-entropy
        loss_domain_bac = nn.CrossEntropyLoss()(domain_pred_bac, domain_label_bac)
        loss_domain_enc = nn.CrossEntropyLoss()(domain_pred_enc, domain_label_enc)
        loss_domain_dec = nn.CrossEntropyLoss()(domain_pred_dec, domain_label_dec)
        loss_dict = {
            'loss_domain_bac': loss_domain_bac,
            'loss_domain_enc': loss_domain_enc,
            'loss_domain_dec': loss_domain_dec,
        }
        return loss_dict
        # loss = loss_domain_enc + loss_domain_dec + self.coef_domain_bac * loss_domain_bac
        # return loss, loss_dict

    @staticmethod
    def loss_mae(out, targets, indices, num_interactions,domain_label=None):
        num_layers = len(out['mae_output'])
        mae_loss = 0.0
        expanded_mask = out['mask'].unsqueeze(dim=1).expand(-1,3,512,512)
        # mae_loss = F.mse_loss(out['mae_output'][-1].reshape(-1,3,512,512)*expanded_mask, out['img']*expanded_mask)
        for layer_idx in range(num_layers):
            if out['mae_output'][layer_idx].shape[1] > 0:
                # mae_loss += F.mse_loss(out['mae_output'][layer_idx], out['features'][layer_idx])
                mae_loss += F.mse_loss(out['mae_output'][layer_idx].reshape(-1,3,512,512), out['img'])
        mae_loss /= num_layers
        losses = {'loss_mae': mae_loss}
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

    def get_loss(self, loss, outputs, targets, indices, num, domain_label=None, **kwargs):
        if 'pred_hoi_logits' in outputs.keys():
            loss_map = {
                'hoi_labels': self.loss_hoi_labels,
                'obj_labels': self.loss_obj_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
                'feats_mimic': self.mimic_loss,
                'rec_loss': self.reconstruction_loss,
                'loss_mae': self.loss_mae,
                'loss_domains': self.loss_domains
            }
        else:
            loss_map = {
                'obj_labels': self.loss_obj_labels,
                'obj_cardinality': self.loss_obj_cardinality,
                'verb_labels': self.loss_verb_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
            }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, domain_label, **kwargs)

    def forward(self, outputs, targets=None, domain_label=None, enable_mae=False):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        if targets is not None:
        # Retrieve the matching between the outputs of the last layer and the targets
            indices = self.matcher(outputs_without_aux, targets)

            num_interactions = sum(len(t['hoi_labels']) for t in targets)
            num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                            device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_interactions)
            num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()
        else:
            indices = None
            num_interactions = None
        # Compute all the requested losses
        losses_me = []
        if targets is not None:
            losses_me += ['hoi_labels', 'obj_labels', 'sub_obj_boxes']
            if self.with_mimic:
                losses_me+=['feats_mimic']
            if self.with_rec_loss:
                losses_me+=['rec_loss']
            # losses_me += ['hoi_labels']
            
        if domain_label is not None:
            losses_me += ['loss_domains']
        if enable_mae:
            losses_me += ['loss_mae']

        losses = {}
        for loss in losses_me:
            # losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))
            if domain_label is not None:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions, domain_label))
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if targets is not None:
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    indices = self.matcher(aux_outputs, targets)
                    for loss in self.losses:
                        kwargs = {}
                        if loss == 'loss_mae' or loss == 'loss_domains':
                            continue
                        if loss =='rec_loss':
                            continue
                        if loss == 'obj_labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
        weight_dict = self.weight_dict
        losses_sum = sum(losses[k] * weight_dict[k] for k in losses.keys() if k in weight_dict)

        return losses_sum, losses


class PostProcessHOITriplet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes,pseudo_labels=False):
        out_hoi_logits = outputs['pred_hoi_logits']
        out_obj_logits = outputs['pred_obj_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']
        clip_visual = outputs['clip_visual']
        clip_logits = outputs['clip_logits']

        assert len(out_hoi_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        hoi_scores = out_hoi_logits.sigmoid()
        obj_scores = out_obj_logits.sigmoid()
        obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)[1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(hoi_scores.device)
        # sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        # sub_boxes = sub_boxes * scale_fct[:, None, :]
        # obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        # obj_boxes = obj_boxes * scale_fct[:, None, :]
        if pseudo_labels:
            sub_boxes = out_sub_boxes
            obj_boxes = out_obj_boxes
        else:
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

            results[-1].update({'hoi_scores': hs.to('cpu'), 'obj_scores': os.to('cpu'), 'clip_visual': clip_visual[index].to('cpu'),
                                'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:], 'clip_logits': clip_logits[index].to('cpu')})

        return results

class MultiConv2d(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv2d(n, k, kernel_size=(3, 3), padding=1) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    gen = build_gen(args)

    model = HOICLIP(
        backbone,
        gen,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    matcher = build_matcher(args)
    weight_dict = {}
    if args.with_clip_label:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef
    else:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef

    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    if args.with_mimic:
        weight_dict['loss_feat_mimic'] = args.mimic_loss_coef

    if args.with_rec_loss:
        weight_dict['loss_rec'] = args.rec_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['hoi_labels', 'obj_labels', 'sub_obj_boxes']
    if args.with_mimic:
        losses.append('feats_mimic')

    if args.with_rec_loss:
        losses.append('rec_loss')

    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOITriplet(args)}

    return model, postprocessors
