import math

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import logging


class AttentionModel(nn.Module):
    def __init__(self, input_size):
        super(AttentionModel, self).__init__()
        self.main_model = nn.Sequential(
            nn.Linear(input_size, 36),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.Linear(36, 1),
        )

    def forward(self, x):
        return self.main_model(x)


def attention(q, k, v, mask=None, attn_func='learnable', return_scores=False, attn_model=None, time_embedding_dim=0, use_time_mode="concat", model_name=None, ETA_H_metirc=None, SDIM_H_metric=None, state="gsu", mode="train", history_category=None, item_embedding_dim=None, soft_search=True, top_k=-1):
    assert len(q.shape) == 2
    assert len(v.shape) == 3
    assert q.shape[0] == k.shape[0]
    assert q.shape[1] == k.shape[2]
    assert mode in ["train", "evaluate"]
    assert state in ["gsu", 'esu']

    if model_name == "TWIN_V2":
        assert torch.equal(k, v)
        assert history_category is not None
        group_centers = []
        group_masks = []
        group_representations = []
        for i in range(0, 199, 2):
            one_group = k[:, i:i+2]
            group_center = torch.mean(one_group, dim=1)
            square_distance = torch.sum((one_group - group_center.unsqueeze(1)) ** 2, dim=2)
            closest_position = torch.argmax(square_distance, dim=1)
            closest_tensor = k[torch.arange(k.shape[0]), i+closest_position]
            group_mask = mask[:, i+1]
            assert group_mask.shape == (q.shape[0],)
            assert closest_tensor.shape == (q.shape[0], k.shape[2])
            assert group_center.shape == (q.shape[0], k.shape[2])
            group_centers.append(group_center)
            group_masks.append(group_mask)
            group_representations.append(closest_tensor)

        group_centers = torch.stack(group_centers, dim=1)
        group_masks = torch.stack(group_masks, dim=1)
        group_representations = torch.stack(group_representations, dim=1)

        q = q.unsqueeze(1)
        score_for_semantics = torch.matmul(q[:, :, :-time_embedding_dim], group_centers[:, :, :-time_embedding_dim].transpose(1, 2)) / math.sqrt(float(q[:, :, :-time_embedding_dim].shape[2]))
        score_for_time = torch.matmul(q[:, :, -time_embedding_dim:], group_centers[:, :, -time_embedding_dim:].transpose(1, 2)) / math.sqrt(float(time_embedding_dim))
        center_scores = score_for_semantics + score_for_time
        center_scores = center_scores.squeeze(1)
        assert center_scores.shape == (k.shape[0], 100)
        group_masks = group_masks.type(torch.bool)
        center_scores = torch.where(group_masks, center_scores, torch.ones_like(center_scores).to(center_scores.device) * (- 2 ** 32 + 1))

        _, top_k_indices = torch.topk(center_scores, top_k, dim=1)
        mask_for_soft_search = torch.zeros_like(group_masks)
        for k_ in range(top_k_indices.shape[0]):
            mask_for_soft_search[k_, top_k_indices[k_]] = 1
        assert torch.all(torch.sum(mask_for_soft_search, dim=1) == top_k)

        score_for_semantics = torch.matmul(q[:, :, :-time_embedding_dim], group_representations[:, :, :-time_embedding_dim].transpose(1, 2)) / math.sqrt(float(q[:, :, :-time_embedding_dim].shape[2]))
        score_for_time = torch.matmul(q[:, :, -time_embedding_dim:], group_representations[:, :, -time_embedding_dim:].transpose(1, 2)) / math.sqrt(float(time_embedding_dim))
        group_repr_scores = score_for_semantics + score_for_time
        group_repr_scores = group_repr_scores.squeeze(1)
        assert group_repr_scores.shape == (k.shape[0], 100)
        group_repr_scores = torch.where(mask_for_soft_search, group_repr_scores, torch.ones_like(center_scores).to(center_scores.device) * (- 2 ** 32 + 1))

        attention_distribution = nn.functional.softmax(group_repr_scores, dim=1)

        out = torch.matmul(attention_distribution.unsqueeze(1), group_representations)
        out = out.squeeze(1)
        return out

    if model_name == "SDIM":
        q_hash = (torch.sign(torch.matmul(q[:, item_embedding_dim:-time_embedding_dim], SDIM_H_metric)) + 1) / 2   # hash to 0 and 1
        k_hash = (torch.sign(torch.matmul(k[:, :, item_embedding_dim:-time_embedding_dim], SDIM_H_metric)) + 1) / 2
        q_hash_split_ = torch.split(q_hash, 3, dim=1)
        k_hash_split_ = torch.split(k_hash, 3, dim=2)
        q_hash_split = []
        k_hash_split = []
        for one_q_hash_group in q_hash_split_:
            q_hash_split.append(torch.sum(torch.stack((one_q_hash_group[:, 0] * 4, one_q_hash_group[:, 1] * 2, one_q_hash_group[:, 2]), dim=1), dim=1))
        for one_k_hash_group in k_hash_split_:
            k_hash_split.append(torch.sum(torch.stack((one_k_hash_group[:, :, 0] * 4, one_k_hash_group[:, :, 1] * 2, one_k_hash_group[:, :, 2]), dim=2), dim=2))
        group_num = len(q_hash_split)
        final_representation = torch.zeros([v.shape[0], v.shape[2]], dtype=v.dtype, device=v.device)
        for i in range(group_num):
            one_query_hash_group = q_hash_split[i]
            one_key_hash_group = k_hash_split[i]
            one_query_hash_group = one_query_hash_group.unsqueeze(1).repeat(1, k.shape[1])
            assert one_query_hash_group.shape == one_key_hash_group.shape
            assert one_query_hash_group.shape == (q.shape[0], k.shape[1])
            selected = (one_query_hash_group == one_key_hash_group).float()
            selected_num = torch.sum(selected, dim=1).unsqueeze(1)
            selected_num = selected_num.repeat(1, v.shape[2])
            selected = selected.unsqueeze(2)
            selected = selected.repeat(1, 1, v.shape[2])
            selected_value = selected * v
            one_group_representation = torch.sum(selected_value, dim=1) / (selected_num+0.01)
            final_representation += one_group_representation
        final_representation /= group_num
        return final_representation


    q = q.unsqueeze(1)

    if attn_func == 'learnable':
        assert attn_model is not None
        q = q.repeat(1, k.shape[1], 1)
        concated_input = torch.cat((q, k, q-k, torch.mul(q, k)), dim=2)
        scores = attn_model(concated_input)
        scores = scores / q.shape[2]
        scores = scores.transpose(1, 2)
    elif attn_func == 'scaled_dot_product':
        if use_time_mode == "concat":
            if not soft_search:
                score_for_semantics = torch.matmul(q[:, :, :item_embedding_dim], k[:, :, :item_embedding_dim].transpose(1, 2)) / math.sqrt(float(q[:, :, :item_embedding_dim].shape[2]))
            else:
                score_for_semantics = torch.matmul(q[:, :, :-time_embedding_dim], k[:, :, :-time_embedding_dim].transpose(1, 2)) / math.sqrt(float(q[:, :, :-time_embedding_dim].shape[2]))
            score_for_time = torch.matmul(q[:, :, -time_embedding_dim:], k[:, :, -time_embedding_dim:].transpose(1, 2)) / math.sqrt(float(time_embedding_dim))
            scores = score_for_semantics + score_for_time
        else:  # use_time_mode: add or decoupled
            scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(q.shape[-1]))
    else:
        raise ValueError('Unknown attention function.')

    assert scores.shape == (k.shape[0], 1, k.shape[1])
    scores = scores.squeeze(1)

    if model_name == "ETA" and ETA_H_metirc is not None:
        q = torch.matmul(torch.sigmoid(q)*2-1, torch.sigmoid(ETA_H_metirc)*2-1)
        k = torch.matmul(torch.sigmoid(k)*2-1, torch.sigmoid(ETA_H_metirc)*2-1)
        q = torch.sigmoid(q)*2 - 1
        k = torch.sigmoid(k)*2 - 1
        q = q.repeat(1, k.shape[1], 1)
        assert q.shape == k.shape
        ETA_scores = -torch.sum((q-k)**2, dim=2)
        assert ETA_scores.shape == (k.shape[0], k.shape[1])
        if state == "gsu":
            scores = ETA_scores
        else:
            if mode == "train":
                scores += 0.1 * ETA_scores
            else:
                pass


    if mask is not None:
        assert mask.shape == scores.shape
        mask = mask.type(torch.bool)
        scores = torch.where(mask, scores, torch.ones_like(scores).to(scores.device) * (- 2**32 + 1))

    attention_distribution = nn.functional.softmax(scores, dim=1)

    out = torch.matmul(attention_distribution.unsqueeze(1), v)
    out = out.squeeze(1)

    if return_scores:
        return out, scores, attention_distribution
    else:
        return out
