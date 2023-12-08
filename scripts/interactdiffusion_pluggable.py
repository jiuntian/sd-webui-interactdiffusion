from collections import OrderedDict

import torch
from torch import nn
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel
import itertools
from natsort import natsorted
import numpy as np
from modules import shared
from scripts.modules.model import GatedSelfAttentionDense, FourierEmbedder

import logging 

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]


class HOIPositionNet(nn.Module):
    """
    This is version 3
    Added: Absolute Positional Embedding to each interaction triplet
    This is version 4
    Added: Positional Embedding to each interaction token, ie. subject, action, object
    max_interactions: maximum interactions in one image
    This is version 5
    Added: between operation for subject and object bounding boxes
    """
    def __init__(self, in_dim, out_dim, fourier_freqs=8, max_interactions=30):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.interaction_embedding = AbsolutePositionalEmbedding(dim=out_dim, max_seq_len=max_interactions)
        self.position_embedding = AbsolutePositionalEmbedding(dim=out_dim, max_seq_len=3)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        self.linears = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.linear_action = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_action_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def get_between_box(self, bbox1, bbox2):
        """ Between Set Operation
        Operation of Box A between Box B from Prof. Jiang
        """
        all_x = torch.cat([bbox1[:, :, 0::2], bbox2[:, :, 0::2]],dim=-1)
        all_y = torch.cat([bbox1[:, :, 1::2], bbox2[:, :, 1::2]],dim=-1)
        all_x, _ = all_x.sort()
        all_y, _ = all_y.sort()
        return torch.stack([all_x[:,:,1], all_y[:,:,1], all_x[:,:,2], all_y[:,:,2]],2)

    def forward(self, subject_boxes, object_boxes, masks,
                subject_positive_embeddings, object_positive_embeddings, action_positive_embeddings):
        B, N, _ = subject_boxes.shape
        masks = masks.unsqueeze(-1)

        # embedding position (it may include padding as placeholder)
        action_boxes = self.get_between_box(subject_boxes, object_boxes)
        subject_xyxy_embedding = self.fourier_embedder(subject_boxes)  # B*N*4 --> B*N*C
        object_xyxy_embedding = self.fourier_embedder(object_boxes)  # B*N*4 --> B*N*C
        action_xyxy_embedding = self.fourier_embedder(action_boxes)  # B*N*4 --> B*N*C

        # learnable null embedding
        positive_null = self.null_positive_feature.view(1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)
        action_null = self.null_action_feature.view(1, 1, -1)

        # replace padding with learnable null embedding
        subject_positive_embeddings = subject_positive_embeddings * masks + (1 - masks) * positive_null
        object_positive_embeddings = object_positive_embeddings * masks + (1 - masks) * positive_null

        subject_xyxy_embedding = subject_xyxy_embedding * masks + (1 - masks) * xyxy_null
        object_xyxy_embedding = object_xyxy_embedding * masks + (1 - masks) * xyxy_null
        action_xyxy_embedding = action_xyxy_embedding * masks + (1 - masks) * xyxy_null

        action_positive_embeddings = action_positive_embeddings * masks + (1 - masks) * action_null

        objs_subject = self.linears(torch.cat([subject_positive_embeddings, subject_xyxy_embedding], dim=-1))
        objs_object = self.linears(torch.cat([object_positive_embeddings, object_xyxy_embedding], dim=-1))
        objs_action = self.linear_action(torch.cat([action_positive_embeddings, action_xyxy_embedding], dim=-1))

        objs_subject = objs_subject + self.interaction_embedding(objs_subject)
        objs_object = objs_object + self.interaction_embedding(objs_object)
        objs_action = objs_action + self.interaction_embedding(objs_action)

        objs_subject = objs_subject + self.position_embedding.emb(torch.tensor(0).to(objs_subject.device))
        objs_object = objs_object + self.position_embedding.emb(torch.tensor(1).to(objs_object.device))
        objs_action = objs_action + self.position_embedding.emb(torch.tensor(2).to(objs_action.device))

        objs = torch.cat([objs_subject, objs_action, objs_object], dim=1)

        assert objs.shape == torch.Size([B, N*3, self.out_dim])
        return objs


class ProxyBasicTransformerBlock(object):
    def __init__(self, controller, org_module: torch.nn.Module = None):
        super().__init__()
        self.org_module = org_module
        self.org_forward = None
        self.fuser = None
        self.attached = False
        self.controller = controller
        self.objs = None


    def __getattr__(self, attr):
        if attr not in ['org_module', 'org_forward', 'fuser', 'attached', 'controller', 'objs'] and self.attached:
            return getattr(self.org_module, attr)

    def initialize_fuser(self, fuser_state_dict):
        query_dim = self.org_module.attn1.to_q.in_features
        key_dim = self.org_module.attn2.to_k.in_features
        n_heads = self.org_module.attn1.heads
        d_head = int(self.org_module.attn2.to_q.out_features / n_heads)
        self.fuser = GatedSelfAttentionDense(query_dim, key_dim, n_heads, d_head)
        self.fuser.load_state_dict(fuser_state_dict)

    def apply_to(self):
        if self.org_forward is not None:
            return
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        self.attached = True

    def detach(self):
        if self.org_forward is None:
            return
        self.org_module.forward = self.org_forward
        self.org_forward = None
        self.attached = False

    def forward(self, x, context):
        x = self.attn1( self.norm1(x) ) + x
        x = self.fuser(x,  self.controller.batch_objs_input) # identity mapping in the beginning
        x = self.attn2(self.norm2(x), context, context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class ProxyUNetModel(object):
    def __init__(self, controller, org_module: torch.nn.Module = None):
        super().__init__()
        self.org_module = org_module
        self.org_forward = None
        self.attached = False
        self.controller = controller
        self.objs = None

    def __getattr__(self, attr):
        if attr not in ['org_module', 'org_forward', 'fuser', 'attached', 'controller', 'objs'] and self.attached:
            return getattr(self.org_module, attr)

    def apply_to(self):
        if self.org_forward is not None:
            return
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        self.attached = True

    def detach(self):
        if self.org_forward is None:
            return
        self.org_module.forward = self.org_forward
        self.org_forward = None
        self.attached = False

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        self.controller.unet_signal(timesteps=timesteps, x=x)
        return self.org_forward(x, timesteps=timesteps, context=context, y=y, **kwargs)



known_block_prefixes = [
    'input_blocks.0.',
    'input_blocks.1.',
    'input_blocks.2.',
    'input_blocks.3.',
    'input_blocks.4.',
    'input_blocks.5.',
    'input_blocks.6.',
    'input_blocks.7.',
    'input_blocks.8.',
    'input_blocks.9.',
    'input_blocks.10.',
    'input_blocks.11.',
    'middle_block.',
    'output_blocks.0.',
    'output_blocks.1.',
    'output_blocks.2.',
    'output_blocks.3.',
    'output_blocks.4.',
    'output_blocks.5.',
    'output_blocks.6.',
    'output_blocks.7.',
    'output_blocks.8.',
    'output_blocks.9.',
    'output_blocks.10.',
    'output_blocks.11.',
]


def timestep_to_alpha(timestep, stage_one=0.2, stage_two=0.5, stength=1.0):
    timestep = timestep[0]
    if timestep >= (1 - stage_one) * 1000.0:
        return 1.0 * stength
    elif (1 - stage_two) * 1000.0 <= timestep < (1 - stage_one) * 1000.0:
        # linear
        return (timestep - (1 - stage_two) * 1000.0) / ((stage_two - stage_one) * 1000.0) * stength
    elif 0 <= timestep < (1 - stage_two) * 1000.0:
        return 0.0


class PluggableInteractDiffusion:
    def __init__(self, ori_unet:UNetModel, interactdiffusion_state_dict):
        super().__init__()
        self.proxy_blocks=[]
        self.gated_self_attention_modules = []
        self.empty_objs = None
        self.objs=None
        self.batch_objs_input = None
        self.batch_size = 1
        self.strength = 1.0
        self.stage_one = 0.8
        self.stage_two = 0.0

        interactdiffusion_state_dict_keys = interactdiffusion_state_dict.keys()  # try without sorted
        interactdiffusion_sorted_dict = interactdiffusion_state_dict
        
        for block_idx, unet_block in enumerate(itertools.chain(ori_unet.input_blocks, [ori_unet.middle_block], ori_unet.output_blocks)):
            cur_block_prefix = known_block_prefixes[block_idx]
            cur_block_fuse_state_dict = {key: value for key, value in interactdiffusion_sorted_dict.items() if key.startswith(cur_block_prefix)}


            if len(cur_block_fuse_state_dict) != 0:
                if len(cur_block_fuse_state_dict) != 17:
                    raise Exception(f'State dict for block {cur_block_prefix} is not correct, have {len(cur_block_fuse_state_dict)} items')
                verify_cur_block_fuse_state_dict = cur_block_fuse_state_dict
                for key, value in verify_cur_block_fuse_state_dict.items():
                    if not key.startswith(cur_block_prefix):
                        raise Exception(f'State dict for block {cur_block_prefix} is not correct. Current key is {key}')
                    del interactdiffusion_sorted_dict[key]

                # trim state_dict keys
                key_after_fuser_pointer = list(cur_block_fuse_state_dict.keys())[0].index('fuser.') + len('fuser.')
                cur_block_fuse_state_dict = {key[key_after_fuser_pointer:]: value for key, value in cur_block_fuse_state_dict.items()}
            else:
                continue

            for module in unet_block.modules():
                if type(module) is SpatialTransformer:
                    spatial_transformer = module
                    for basic_transformer_block in spatial_transformer.transformer_blocks:
                        cur_proxy_block = ProxyBasicTransformerBlock(self, basic_transformer_block)
                        cur_proxy_block.initialize_fuser(cur_block_fuse_state_dict)
                        # cur_proxy_block.apply_to()
                        self.gated_self_attention_modules.append(cur_proxy_block.fuser)
                        self.proxy_blocks.append(cur_proxy_block)

        verify_position_net_state_dict = interactdiffusion_sorted_dict
        for key, value in verify_position_net_state_dict.items():
            if not key.startswith('position_net.'):
                raise Exception('State dict for position_net is not correct')

        # trim state_dict keys
        key_after_position_net_pointer = list(interactdiffusion_sorted_dict.keys())[0].index('position_net.') + len(
            'position_net.')
        position_net_state_dict = {key[key_after_position_net_pointer:]: value for key, value in interactdiffusion_sorted_dict.items()}
        
        self.position_net = HOIPositionNet(in_dim=768, out_dim=768)
        self.position_net.load_state_dict(position_net_state_dict)

        self.unet_proxy = ProxyUNetModel(self, ori_unet)
        # generate placeholder objs for unconditional generation
        max_objs = 30
        boxes = torch.zeros(max_objs, 4).unsqueeze(0)
        masks = torch.zeros(max_objs).unsqueeze(0)
        text_embeddings = torch.zeros(max_objs, 768).unsqueeze(0)
        self.empty_objs = self.position_net(boxes, boxes, masks, text_embeddings, text_embeddings, text_embeddings)


    def update_stages(self, strength, stage_one, stage_two):
        self.strength = strength
        self.stage_one = stage_one
        self.stage_two = stage_two
    def update_objs(self, subject_boxes, object_boxes, masks,
                    subject_text_embeddings, object_text_embeddings, action_text_embeddings, batch_size):
        self.objs = self.position_net(subject_boxes, object_boxes, masks,
                                      subject_text_embeddings, object_text_embeddings, action_text_embeddings)
        self.batch_size = batch_size
        for module in self.gated_self_attention_modules:
            module.to(device=shared.device, dtype=shared.sd_model.dtype)

    def attach_all(self):
        for proxy_block in self.proxy_blocks:
            proxy_block.apply_to()
        self.unet_proxy.apply_to()

    def detach_all(self):
        for proxy_block in self.proxy_blocks:
            proxy_block.detach()
        self.unet_proxy.detach()

    def unet_signal(self, timesteps, x):
        calculated_alpha = timestep_to_alpha(timesteps, self.stage_one, self.stage_two, self.strength)
        # calculated_alpha = torch.Tensor([calculated_alpha]).to(device=x.device, dtype=x.dtype)
        for module in self.gated_self_attention_modules:
            module.scale = calculated_alpha

        # repeat objs according to batch size-1 then append empty_objs for unconditional
        single_batch_slice_size = x.shape[0] // self.batch_size
        self.objs = self.objs.to(device=x.device, dtype=x.dtype)
        self.empty_objs = self.empty_objs.to(device=x.device, dtype=x.dtype)

        if single_batch_slice_size == 1 and self.batch_size == 1:
            # dealing with unmatched cond and uncond situation
            #TODO: need indication of whether current batch is cond or uncond
            self.batch_objs_input = self.objs
            return

        self.batch_objs_input = torch.cat([self.objs.repeat(single_batch_slice_size - 1, 1, 1), self.empty_objs], dim=0)
        # logging.warn(f"batch_objs_input: {self.batch_objs_input.shape}")
        if self.batch_size != 1:
            # self.batch_objs_input = self.batch_objs_input.repeat(self.batch_size, 1, 1)
            # for bs>1, the batch order is: cond1, cond2, uncond1, uncond2
            self.batch_objs_input = self.batch_objs_input.repeat_interleave(self.batch_size, 0)

