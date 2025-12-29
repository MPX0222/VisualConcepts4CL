import math
import open_clip
from open_clip.tokenizer import HFTokenizer
import numpy as np
import torch
import torch.nn as nn
from model.CLIP_Base_Net import CLIP_Base_Net
import copy
from model.backbone.clip.clip import load, tokenize
from model.backbone.MedCLIP.model import MedCLIPModel, MedCLIPVisionModelViT
from model.backbone.Adapter import Adapter
from utils.functions import *


import torch
import torch.nn as nn

class ShiftScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 初始参数
        self.register_parameter('scale', nn.Parameter(torch.ones(dim)))
        self.register_parameter('shift', nn.Parameter(torch.zeros(dim)))
        # 旧参数（合并后的冻结参数）
        self.old_scale = None
        self.old_shift = None

    def expand(self):
        """
        扩展模块：新建一组参数，并与旧参数加权合并，生成新的旧参数。
        """
        # 获取当前参数数据
        current_scale_data = self.scale.data.clone()
        current_shift_data = self.shift.data.clone()

        # 计算新参数的初始值（全1和全0）
        new_scale_data = torch.ones_like(self.scale).data
        new_shift_data = torch.zeros_like(self.shift).data

        # 计算合并后的参数数据
        if self.old_scale is not None:
            combined_scale_data = 0.9 * self.old_scale.data + 0.1 * current_scale_data
            combined_shift_data = 0.9 * self.old_shift.data + 0.1 * current_shift_data
        else:
            combined_scale_data = current_scale_data
            combined_shift_data = current_shift_data

        # 删除旧参数（如果存在）
        if hasattr(self, 'old_scale'):
            del self.old_scale
        if hasattr(self, 'old_shift'):
            del self.old_shift

        # 删除当前参数
        del self.scale
        del self.shift

        # 注册新的旧参数（冻结）
        self.register_parameter('old_scale', nn.Parameter(combined_scale_data, requires_grad=False))
        self.register_parameter('old_shift', nn.Parameter(combined_shift_data, requires_grad=False))

        # 注册新的当前参数（可训练）
        self.register_parameter('scale', nn.Parameter(new_scale_data))
        self.register_parameter('shift', nn.Parameter(new_shift_data))

    def forward(self, x, add_residual=False):
        if self.old_scale is not None:
            scale = 0.9 * self.old_scale + 0.1 * self.scale
            shift = 0.9 * self.old_shift + 0.1 * self.shift
        else:
            scale = self.scale
            shift = self.shift
        # return 0.1*(x * scale + shift)
        return x * scale + shift


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=False)
        # self.linear_q = nn.Identity()
        self.linear_k = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, dim, bias=False)
        # self.linear_output = nn.Linear(all_head_size, hidden_size)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, Q_input, KV_input):
        B, N, C = Q_input.shape

        q = self.linear_q(Q_input).view(B, -1, self.num_heads, self.head_dim).transpose(2, 1)
        k = self.linear_k(KV_input).view(B, -1, self.num_heads, self.head_dim).transpose(2, 1)
        v = self.linear_v(KV_input).view(B, -1, self.num_heads, self.head_dim).transpose(2, 1)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        # q = q / q.norm(dim=-1, keepdim=True)
        # k = k / k.norm(dim=-1, keepdim=True)
        attn = (q @ k.transpose(-2, -1))*self.scale
        # print(attn.shape)
        # print(attn)
        attn_softmaxed = attn.softmax(dim=-1)
        # print(attn_softmaxed)
        # attn = self.attn_drop(attn)

        x = (attn_softmaxed @ v).transpose(2, 1).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x, attn_softmaxed

class CLIP_Concept_Net(CLIP_Base_Net):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.alpha = config.alpha
        self.inc = config.increment_steps[0]
        self.img_adapter_list = nn.ModuleList([])

    def model_init(self):
        if self.config.backbone == "CLIP":
            self.backbone, _ = load(self.config.pretrained_path, jit=False)
        elif self.config.backbone == "OpenCLIP":
            # self.backbone, _ = open_clip.create_model_from_pretrained("ViT-B-16", pretrained=self.config.pretrained_path+"/open_clip_pytorch_model.bin")
            self.backbone = open_clip.create_model("ViT-B-16-SigLIP")
            self.backbone.load_state_dict(
                torch.load(self.config.pretrained_path+"/open_clip_pytorch_model.bin")
                )
            self.backbone.float()
        elif self.config.backbone == "MedCLIP":
            self.backbone = MedCLIPModel(MedCLIPVisionModelViT)
            self.backbone.from_pretrained(self.config.pretrained_path)
        self.output_dim = self.backbone.output_dim
        # self.output_dim = 512
        self.logger.info("model loaded!")

        # self.feature_fusion_module = nn.Identity()
        self.aux_fc = None
        self.fc = None
        self.cross_attn = Attention(self.output_dim, 1)
        self.ln_layer = nn.LayerNorm(self.output_dim)


    def update_model(self, task_id):
        new_classes = self.config.increment_steps[task_id]
        known_classes = sum(self.config.increment_steps[:task_id])
        cur_classes = new_classes + known_classes

        new_fc = nn.Linear(self.output_dim, cur_classes)
        if self.fc is not None:
            new_fc.weight.data[:known_classes, :] = copy.deepcopy(self.fc.weight.data)
            new_fc.bias.data[:known_classes] = copy.deepcopy(self.fc.bias.data)
            self.logger.info('Updated classifier head output dim from {} to {}'.format(known_classes, cur_classes))
        else:
            self.logger.info('Created classifier head with output dim {}'.format(cur_classes))
        del self.fc
        self.fc = new_fc

        new_fc1 = nn.Linear(self.output_dim, cur_classes)
        if self.aux_fc is not None:
            new_fc1.weight.data[:known_classes, :] = copy.deepcopy(self.aux_fc.weight.data)
            new_fc1.bias.data[:known_classes] = copy.deepcopy(self.aux_fc.bias.data)
        del self.aux_fc
        self.aux_fc = new_fc1



        # return image_features, qk_loss
    def text_tokenize(self, class_names, prompt_template, descs=None):
        if self.config.backbone == "CLIP":
            if descs is None:
                text = [prompt_template.format(c) for c in class_names]
            else:
                text = []
                for class_name in class_names:
                    text.extend([i for i in descs[class_name]])
            text_tokens = tokenize(text)
        elif self.config.backbone == "OpenCLIP":
            # tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
            tokenizer = HFTokenizer(self.config.pretrained_path)
            text_tokens = tokenizer([prompt_template.format(c) for c in class_names], context_length=self.backbone.context_length)
        elif self.config.backbone == "MedCLIP":
            text_tokens = self.backbone.tokenize([prompt_template.format(c) for c in class_names])
            del text_tokens["token_type_ids"]

        return text_tokens


    def forward_train(self, image, text_tokens=None, task_id=None, labels=None):
        class_num = text_tokens.shape[0] // 3
        B = image.shape[0]

        # 获取图像特征
        image_features, _ = self.backbone.encode_image(image, adapter_list=self.cur_img_adapter,
                                                                    ret_all=True)
        image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)  # [B, 512]
        text_features = self.backbone.encode_text(text_tokens, adapter_list=None, prompt=None)  # [class_num*3, dim]
        Q_input = image_features.unsqueeze(1)
        KV_input = text_features.unsqueeze(0).expand(B, -1, -1)
        fused_img_features, attn_matrix = self.cross_attn(self.ln_layer(Q_input), self.ln_layer(KV_input))

        logits = self.fc(image_features)
        logits1 = self.aux_fc(fused_img_features.squeeze(1))
        logits = self.alpha*logits+(1-self.alpha)*logits1

        return {"logits": logits, "features": image_features_normed, "attn": attn_matrix.squeeze(1).squeeze(1)}

    def forward_test(self, image, text_tokens=None, task_id=None, img_proto=None, labels=None):
        class_num = text_tokens.shape[0] // 3
        B = image.shape[0]
        image_features, _ = self.backbone.encode_image(image, adapter_list=self.cur_img_adapter,
                                                                    ret_all=True)
        image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.backbone.encode_text(text_tokens, adapter_list=None, prompt=None)  # [class_num*3, dim]
        Q_input = image_features.unsqueeze(1)
        KV_input = text_features.unsqueeze(0).expand(B, -1, -1)
        fused_img_features, attn_matrix = self.cross_attn(self.ln_layer(Q_input), self.ln_layer(KV_input))
        logits = self.fc(image_features)
        logits1 = self.aux_fc(fused_img_features.squeeze(1))
        logits = self.alpha*logits+(1-self.alpha)*logits1

        return {"logits": logits, "features": image_features_normed, "attn":attn_matrix.squeeze(1).squeeze(1)}

    def forward(self, image, text_tokens=None, train=False, task_id=None, img_proto=None, labels=None):
        if text_tokens is None:
            image_features = self.backbone.encode_image(image, adapter_list=self.cur_img_adapter)
            return {"features": image_features}
        else:
            if train:
                out = self.forward_train(image, text_tokens, task_id, labels=labels)
            else:
                out = self.forward_test(image, text_tokens, task_id, img_proto, labels=labels)

            return out

    def forward_with_vectors(self, image_features, text_tokens, labels=None):
        class_num = text_tokens.shape[0] // 3  # 每个类别3个文本描述
        B = image_features.shape[0]
        image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)  # [B, 512]

        text_features = self.backbone.encode_text(text_tokens, adapter_list=None, prompt=None)  # [class_num*3, dim]
        Q_input = image_features.unsqueeze(1)

        KV_input = text_features.unsqueeze(0).expand(B, -1, -1)
        fused_img_features, attn_matrix = self.cross_attn(self.ln_layer(Q_input), self.ln_layer(KV_input))

        logits = self.fc(image_features)
        logits1 = self.aux_fc(fused_img_features.squeeze(1))
        logits = self.alpha*logits+(1-self.alpha)*logits1

        return {"logits": logits, "features": image_features_normed, "attn":attn_matrix.squeeze(1).squeeze(1)}

    def stage1_param(self, task_id):
        if task_id == 0:
            for name, param in self.backbone.visual.transformer.resblocks[-2:].named_parameters():
                param.requires_grad = True



