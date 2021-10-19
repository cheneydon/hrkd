import os
import re
import shutil
import torch
import numpy as np
from collections import OrderedDict


def save_checkpoint(state, save_dir, ckpt_name, keep_num, is_best=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, ckpt_name)
    torch.save(state, save_path)
    if is_best:
        shutil.copy(save_path, os.path.join(save_dir, 'best_model.bin'))

    ckpt_head = re.split(r'\d+', ckpt_name)[0]
    all_ckpt = np.array([file for file in os.listdir(save_dir) if re.match(ckpt_head, file) is not None])
    all_idx = np.int32([re.findall(r'\d+', ckpt)[0] for ckpt in all_ckpt])
    sorted_ckpt = all_ckpt[np.argsort(all_idx)[::-1]]
    remove_path = [os.path.join(save_dir, name) for name in sorted_ckpt[keep_num:]]
    for path in remove_path:
        os.remove(path)


def load_pretrain_state_dict(model, state_dict_path, add_module=False, load_lm_weights=False, is_finetune=False,
                             skip_name=''):
    raw_state_dict = torch.load(state_dict_path, map_location='cpu')

    if is_finetune:
        raw_state_dict = raw_state_dict['state_dict']
        new_state_dict = {}
        for n, p in raw_state_dict.items():
            if re.search(r'lm_head', n) is None:
                n = re.sub(r'module\.', '', n)
                n = 'module.' + n if add_module else n
                if skip_name and re.search(skip_name, n) is not None:
                    print('Skip loading params: {}'.format(n))
                    continue
                new_state_dict[n] = p
    else:
        new_state_dict = {}
        for n, p in raw_state_dict.items():
            # Bert & Roberta & TinyBert
            if re.match(r'bert|roberta|fit_denses', n) is not None and re.search(r'pooler', n) is None:
                if re.match(r'roberta', n) is not None and re.search(r'token_type_embeddings', n) is not None:
                    continue
                n = re.sub(r'(bert|roberta|layer|self)\.', '', n)
                n = re.sub(r'word_embeddings', 'token_embeddings', n)
                n = re.sub(r'token_type_embeddings', 'segment_embeddings', n)
                n = re.sub(r'LayerNorm', 'layernorm', n)
                n = re.sub(r'gamma', 'weight', n)
                n = re.sub(r'beta', 'bias', n)
                n = re.sub(r'attention\.output', 'attention', n)
                n = re.sub(r'intermediate\.dense', 'ffn.dense1', n)
                n = re.sub(r'output\.dense', 'ffn.dense2', n)
                n = re.sub(r'output', 'ffn', n)
                n = 'module.' + n if add_module else n
                new_state_dict[n] = p

            # Xlnet
            if re.match(r'transformer', n) is not None and re.search(r'mask_emb', n) is None:
                n = re.sub(r'layer\.', '', n)
                n = re.sub(r'transformer\.word_embedding', 'base_model.token_embeddings', n)
                n = re.sub(r'transformer', 'encoder', n)
                n = re.sub(r'seg_embed', 'seg_mat', n)
                n = re.sub(r'layer_norm', 'layernorm', n)
                n = re.sub(r'ff', 'ffn', n)
                n = re.sub(r'layer_1', 'dense1', n)
                n = re.sub(r'layer_2', 'dense2', n)
                n = 'module.' + n if add_module else n
                new_state_dict[n] = p

            # Bert LM weights
            if load_lm_weights and re.match(r'cls\.predictions', n) is not None:
                n = re.sub(r'cls\.predictions', 'lm_head', n)
                n = re.sub(r'lm_head\.bias', 'lm_head.lm_bias', n)
                n = re.sub(r'transform\.', '', n)
                n = re.sub(r'LayerNorm\.gamma', 'layernorm.weight', n)
                n = re.sub(r'LayerNorm\.beta', 'layernorm.bias', n)
                n = re.sub(r'decoder', 'lm_decoder', n)
                n = 'module.' + n if add_module else n
                new_state_dict[n] = p

    model_state_dict = model.state_dict()
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)


def load_resume_state_dict(model, resume_path, add_module=False, optimizer=None, scheduler=None, skip_name=''):
    checkpoint = torch.load(resume_path, map_location='cpu')
    model_state_dict = model.state_dict()

    new_state_dict = OrderedDict()
    for n, p in checkpoint['state_dict'].items():
        n = re.sub(r'module\.', '', n)
        n = 'module.' + n if add_module else n
        if n not in model_state_dict:
            continue
        if skip_name and re.search(skip_name, n) is not None:
            print('Skip loading params: {}'.format(n))
            continue
        new_state_dict[n] = p
    model.load_state_dict(new_state_dict)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint


def load_multi_task_state_dict(model, state_dict_path, task_id):
    raw_state_dict = torch.load(state_dict_path, map_location='cpu')['state_dict']
    new_state_dict = {}
    for n, p in raw_state_dict.items():
        res = re.search(r'classifiers.(\d+)', n)
        if res is not None:
            cur_task_id = int(res.groups()[0])
            if cur_task_id == task_id:
                n = n[:res.start()] + 'classifier' + n[res.end():]
                new_state_dict[n] = p
        else:
            new_state_dict[n] = p

    model_state_dict = model.state_dict()
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)
