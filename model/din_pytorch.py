import logging
import os

import torch
import torch.nn as nn
import numpy as np
import json

from .base_model_pytorch import BaseModel
from .attention_pytorch import attention


class DIN(BaseModel):
    def __init__(self,
                 *args,
                 soft_search=False,
                 top_k=10,
                 use_temporal_encoding_mode,
                 use_decoupled_embedding = True,
                 need_clustering = False,
                 **kwargs,
                 ):
        super(DIN, self).__init__(*args, **kwargs)
        self.soft_search = soft_search
        self.top_k = top_k
        self.use_temporal_encoding_mode = use_temporal_encoding_mode
        self.use_decoupled_embedding = use_decoupled_embedding
        self.need_clustering = need_clustering
        if self.use_temporal_encoding_mode == 'target':
            if need_clustering:
                self.time_id_embedding_layer_for_attention_of_target = nn.Embedding(self.cluster_cate_n, self.time_embedding_dim)
                self.time_id_embedding_layer_for_representation_of_target = nn.Embedding(self.cluster_cate_n, self.time_embedding_dim)
                with open('data/taobao/category_to_cluster_mapping.json', 'r') as f:
                    self.category_to_cluster_mapping = json.load(f)
            else:
                self.time_id_embedding_layer_for_attention_of_target = nn.Embedding(self.cate_n, self.time_embedding_dim)
                self.time_id_embedding_layer_for_representation_of_target = nn.Embedding(self.cate_n, self.time_embedding_dim)
        elif  self.use_temporal_encoding_mode == 'all':
            if need_clustering:
                self.time_id_embedding_layer_for_attention_of_target = nn.Embedding(self.cluster_cate_n * (self.max_length+1), self.time_embedding_dim)
                self.time_id_embedding_layer_for_representation_of_target = nn.Embedding(self.cluster_cate_n * (self.max_length+1), self.time_embedding_dim)
                with open('data/taobao/category_to_cluster_mapping.json', 'r') as f:
                    self.category_to_cluster_mapping = json.load(f)
                self.time_id_embedding_layer_for_attention_of_behavior = self.time_id_embedding_layer_for_attention_of_target
                self.time_id_embedding_layer_for_representation_of_behavior = self.time_id_embedding_layer_for_representation_of_target
            else:
                self.time_id_embedding_layer_for_attention_of_target = nn.Embedding(self.cate_n * (self.max_length+1), self.time_embedding_dim)
                self.time_id_embedding_layer_for_representation_of_target = nn.Embedding(self.cate_n * (self.max_length+1), self.time_embedding_dim)
                self.time_id_embedding_layer_for_attention_of_behavior = self.time_id_embedding_layer_for_attention_of_target
                self.time_id_embedding_layer_for_representation_of_behavior = self.time_id_embedding_layer_for_representation_of_target
        else:
            pass

        if self.use_decoupled_embedding:
            self.item_id_embedding_layer_for_attention_of_target_2 = nn.Embedding(self.item_n, self.item_embedding_dim)
            self.category_id_embedding_layer_for_attention_of_target_2 = nn.Embedding(self.cate_n, self.category_embedding_dim)
            self.time_id_embedding_layer_for_attention_of_target_2 = nn.Embedding(self.max_length+1, self.time_embedding_dim)
            
            self.item_id_embedding_layer_for_representation_of_target_2 = self.item_id_embedding_layer_for_attention_of_target_2
            self.category_id_embedding_layer_for_representation_of_target_2 = self.category_id_embedding_layer_for_attention_of_target_2
            self.time_id_embedding_layer_for_representation_of_target_2 = self.time_id_embedding_layer_for_attention_of_target_2

            self.item_id_embedding_layer_for_attention_of_behavior_2 = self.item_id_embedding_layer_for_attention_of_target_2
            self.category_id_embedding_layer_for_attention_of_behavior_2 = self.category_id_embedding_layer_for_attention_of_target_2
            self.time_id_embedding_layer_for_attention_of_behavior_2 = self.time_id_embedding_layer_for_attention_of_target_2

            self.item_id_embedding_layer_for_representation_of_behavior_2 = self.item_id_embedding_layer_for_attention_of_target_2
            self.category_id_embedding_layer_for_representation_of_behavior_2 = self.category_id_embedding_layer_for_attention_of_target_2
            self.time_id_embedding_layer_for_representation_of_behavior_2 = self.time_id_embedding_layer_for_attention_of_target_2
            if self.use_target_aware_target_temporal_encoding:
                self.time_id_embedding_layer_for_attention_of_target_2 = nn.Embedding(self.cate_n, self.time_embedding_dim)
                self.time_id_embedding_layer_for_representation_of_target_2 = self.time_id_embedding_layer_for_attention_of_target

    def forward(self, batch_data, mode="train", only_return_representation=False, only_return_scores=False):
        # get data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        item_ids = torch.tensor(batch_data['iid'], dtype=torch.int64).to(device)
        category_ids = torch.tensor(batch_data['cid'], dtype=torch.int64).to(device)
        if self.use_temporal_encoding_mode == 'all':
            if self.need_clustering:
                time_ids = torch.tensor(batch_data['cid'], dtype=torch.int64).to(device) 
                time_ids = torch.tensor([self.category_to_cluster_mapping[str(cid.item())] for cid in time_ids], dtype=torch.int64).to(device)
                time_ids = time_ids * self.max_length
            else:
                time_ids = torch.tensor(batch_data['cid'], dtype=torch.int64).to(device)
                time_ids = time_ids * (self.max_length+1)
        elif self.use_temporal_encoding_mode == 'target':
            if self.need_clustering:
                time_ids = torch.tensor(batch_data['cid'], dtype=torch.int64).to(device) 
                time_ids = torch.tensor([self.category_to_cluster_mapping[str(cid.item())] for cid in time_ids], dtype=torch.int64).to(device)
            else:
                time_ids = torch.tensor(batch_data['cid'], dtype=torch.int64).to(device)
        else:
            time_ids = torch.tensor(batch_data['tid'], dtype=torch.int64).to(device)
        

        history_item_ids = torch.tensor(batch_data['hist_iid'], dtype=torch.int64).to(device)
        history_category_ids = torch.tensor(batch_data['hist_cid'], dtype=torch.int64).to(device)

        if self.use_temporal_encoding_mode == 'all':
            history_time_ids = torch.tensor(batch_data['hist_tid'], dtype=torch.int64).to(device)
            time_ids_expand = time_ids.unsqueeze(1).repeat(1, self.max_length)
            history_time_ids = time_ids_expand + history_time_ids
        else:
            history_time_ids = torch.tensor(batch_data['hist_tid'], dtype=torch.int64).to(device)
        mask = batch_data['mask']
        if not self.soft_search:
            num_ones = np.sum(mask, axis=1)
            for i in range(mask.shape[0]):
                if num_ones[i] > self.top_k:
                    one_incides = np.where(mask[i] == 1)[0]
                    aim_position = one_incides[round(num_ones[i] - self.top_k)]
                    mask[i][:aim_position] = 0
                elif num_ones[i] < self.top_k:
                    zero_incides = np.where(mask[i] == 0)[0]
                    aim_position = zero_incides[round(-(self.top_k - num_ones[i]))]
                    mask[i][aim_position:] = 1
            assert np.all(np.sum(mask, axis=1) == self.top_k)
        mask = torch.tensor(mask, dtype=torch.int32).to(device)
        target = torch.tensor(batch_data['target'], dtype=torch.int32).to(device)

        # embedding
        embedded_item_for_attention_of_target = self.item_id_embedding_layer_for_attention_of_target(item_ids)
        embedded_category_for_attention_of_target = self.category_id_embedding_layer_for_attention_of_target(category_ids)
        embedded_time_for_attention_of_target = self.time_id_embedding_layer_for_attention_of_target(time_ids)

        embedded_item_for_representation_of_target = self.item_id_embedding_layer_for_representation_of_target(item_ids)
        embedded_category_for_representation_of_target = self.category_id_embedding_layer_for_representation_of_target(category_ids)
        embedded_time_for_representation_of_target = self.time_id_embedding_layer_for_representation_of_target(time_ids)

        embedded_item_for_attention_of_behavior = self.item_id_embedding_layer_for_attention_of_behavior(history_item_ids)
        embedded_category_for_attention_of_behavior = self.category_id_embedding_layer_for_attention_of_behavior(history_category_ids)
        embedded_time_for_attention_of_behavior = self.time_id_embedding_layer_for_attention_of_behavior(history_time_ids)

        embedded_item_for_representation_of_behavior = self.item_id_embedding_layer_for_representation_of_behavior(history_item_ids)
        embedded_category_for_representation_of_behavior = self.category_id_embedding_layer_for_representation_of_behavior(history_category_ids)
        embedded_time_for_representation_of_behavior = self.time_id_embedding_layer_for_representation_of_behavior(history_time_ids)

        if self.use_decoupled_embedding:
            embedded_item_for_attention_of_target_2 = self.item_id_embedding_layer_for_attention_of_target_2(item_ids)
            embedded_category_for_attention_of_target_2 = self.category_id_embedding_layer_for_attention_of_target_2(category_ids)
            embedded_time_for_attention_of_target_2 = self.time_id_embedding_layer_for_attention_of_target_2(time_ids)

            embedded_item_for_representation_of_target_2 = self.item_id_embedding_layer_for_representation_of_target_2(item_ids)
            embedded_category_for_representation_of_target_2 = self.category_id_embedding_layer_for_representation_of_target_2(category_ids)
            embedded_time_for_representation_of_target_2 = self.time_id_embedding_layer_for_representation_of_target_2(time_ids)

            embedded_item_for_attention_of_behavior_2 = self.item_id_embedding_layer_for_attention_of_behavior_2(history_item_ids)
            embedded_category_for_attention_of_behavior_2 = self.category_id_embedding_layer_for_attention_of_behavior_2(history_category_ids)
            embedded_time_for_attention_of_behavior_2 = self.time_id_embedding_layer_for_attention_of_behavior_2(history_time_ids)

            embedded_item_for_representation_of_behavior_2 = self.item_id_embedding_layer_for_representation_of_behavior_2(history_item_ids)
            embedded_category_for_representation_of_behavior_2 = self.category_id_embedding_layer_for_representation_of_behavior_2(history_category_ids)
            embedded_time_for_representation_of_behavior_2 = self.time_id_embedding_layer_for_representation_of_behavior_2(history_time_ids)

        # mlp_transform
        if not self.mlp_position_after_concat:
            if self.model_name == 'transformer_decouple_by_category':
                embedded_category_for_attention_of_target_result = self.mlp_for_category_attention_of_target(embedded_category_for_attention_of_target)
                embedded_time_for_attention_of_target_result = self.mlp_for_time_attention_of_target(embedded_time_for_attention_of_target)
                embedded_category_for_representation_of_target_result = self.mlp_for_category_representation_of_target(embedded_category_for_representation_of_target)
                embedded_time_for_representation_of_target_result = self.mlp_for_time_representation_of_target(embedded_time_for_representation_of_target)
                embedded_category_for_attention_of_behavior_result = self.mlp_for_category_attention_of_behavior(embedded_category_for_attention_of_behavior)
                embedded_time_for_attention_of_behavior_result = self.mlp_for_time_attention_of_behavior(embedded_time_for_attention_of_behavior)
                embedded_category_for_representation_of_behavior_result = self.mlp_for_category_representation_of_behavior(embedded_category_for_representation_of_behavior)
                embedded_time_for_representation_of_behavior_result = self.mlp_for_time_representation_of_behavior(embedded_time_for_representation_of_behavior)

                batch_size = batch_data['hist_cid'].shape[0]
                history_length = batch_data['hist_cid'].shape[1]

                item_attention_category0_tensor = self.mlp_list_for_item_attention[0](embedded_item_for_attention_of_behavior)
                item_attention_category1_tensor = self.mlp_list_for_item_attention[1](embedded_item_for_attention_of_behavior)
                item_attention_category2_tensor = self.mlp_list_for_item_attention[2](embedded_item_for_attention_of_behavior)
                item_attention_category3_tensor = self.mlp_list_for_item_attention[3](embedded_item_for_attention_of_behavior)
                item_attention_category4_tensor = self.mlp_list_for_item_attention[4](embedded_item_for_attention_of_behavior)
                item_attention_category5_tensor = self.mlp_list_for_item_attention[5](embedded_item_for_attention_of_behavior)
                item_attention_category6_tensor = self.mlp_list_for_item_attention[6](embedded_item_for_attention_of_behavior)
                item_attention_category7_tensor = self.mlp_list_for_item_attention[7](embedded_item_for_attention_of_behavior)
                item_attention_category8_tensor = self.mlp_list_for_item_attention[8](embedded_item_for_attention_of_behavior)
                item_attention_category9_tensor = self.mlp_list_for_item_attention[9](embedded_item_for_attention_of_behavior)
                item_attention_tensor_list = [item_attention_category0_tensor, item_attention_category1_tensor, item_attention_category2_tensor, item_attention_category3_tensor, item_attention_category4_tensor, item_attention_category5_tensor, item_attention_category6_tensor, item_attention_category7_tensor, item_attention_category8_tensor, item_attention_category9_tensor]

                item_representation_category0_tensor = self.mlp_list_for_item_representation[0](embedded_item_for_representation_of_behavior)
                item_representation_category1_tensor = self.mlp_list_for_item_representation[1](embedded_item_for_representation_of_behavior)
                item_representation_category2_tensor = self.mlp_list_for_item_representation[2](embedded_item_for_representation_of_behavior)
                item_representation_category3_tensor = self.mlp_list_for_item_representation[3](embedded_item_for_representation_of_behavior)
                item_representation_category4_tensor = self.mlp_list_for_item_representation[4](embedded_item_for_representation_of_behavior)
                item_representation_category5_tensor = self.mlp_list_for_item_representation[5](embedded_item_for_representation_of_behavior)
                item_representation_category6_tensor = self.mlp_list_for_item_representation[6](embedded_item_for_representation_of_behavior)
                item_representation_category7_tensor = self.mlp_list_for_item_representation[7](embedded_item_for_representation_of_behavior)
                item_representation_category8_tensor = self.mlp_list_for_item_representation[8](embedded_item_for_representation_of_behavior)
                item_representation_category9_tensor = self.mlp_list_for_item_representation[9](embedded_item_for_representation_of_behavior)
                item_representation_tensor_list = [item_representation_category0_tensor, item_representation_category1_tensor, item_representation_category2_tensor, item_representation_category3_tensor, item_representation_category4_tensor, item_representation_category5_tensor, item_representation_category6_tensor, item_representation_category7_tensor, item_representation_category8_tensor, item_representation_category9_tensor]

                embedded_item_for_attention_of_behavior_result = []
                embedded_item_for_representation_of_behavior_result = []
                for i in range(batch_size):
                    temp_batch_item_attention_tensor = []
                    temp_batch_item_representation_tensor = []
                    for j in range(history_length):
                        temp_category = batch_data['hist_cid'][i][j] % 3
                        temp_batch_item_attention_tensor.append(item_attention_tensor_list[temp_category][i][j])
                        temp_batch_item_representation_tensor.append(item_representation_tensor_list[temp_category][i][j])
                    temp_batch_item_attention_tensor = torch.stack(temp_batch_item_attention_tensor)
                    temp_batch_item_representation_tensor = torch.stack(temp_batch_item_representation_tensor)
                    embedded_item_for_attention_of_behavior_result.append(temp_batch_item_attention_tensor)
                    embedded_item_for_representation_of_behavior_result.append(temp_batch_item_representation_tensor)
                embedded_item_for_attention_of_behavior_result = torch.stack(embedded_item_for_attention_of_behavior_result)
                embedded_item_for_representation_of_behavior_result = torch.stack(embedded_item_for_representation_of_behavior_result)

                item_attention_category0_tensor_target = self.mlp_list_for_item_attention[0](embedded_item_for_attention_of_target)
                item_attention_category1_tensor_target = self.mlp_list_for_item_attention[1](embedded_item_for_attention_of_target)
                item_attention_category2_tensor_target = self.mlp_list_for_item_attention[2](embedded_item_for_attention_of_target)
                item_attention_category3_tensor_target = self.mlp_list_for_item_attention[3](embedded_item_for_attention_of_target)
                item_attention_category4_tensor_target = self.mlp_list_for_item_attention[4](embedded_item_for_attention_of_target)
                item_attention_category5_tensor_target = self.mlp_list_for_item_attention[5](embedded_item_for_attention_of_target)
                item_attention_category6_tensor_target = self.mlp_list_for_item_attention[6](embedded_item_for_attention_of_target)
                item_attention_category7_tensor_target = self.mlp_list_for_item_attention[7](embedded_item_for_attention_of_target)
                item_attention_category8_tensor_target = self.mlp_list_for_item_attention[8](embedded_item_for_attention_of_target)
                item_attention_category9_tensor_target = self.mlp_list_for_item_attention[9](embedded_item_for_attention_of_target)
                item_attention_target_tensor_list = [item_attention_category0_tensor_target, item_attention_category1_tensor_target,
                                              item_attention_category2_tensor_target, item_attention_category3_tensor_target,
                                              item_attention_category4_tensor_target, item_attention_category5_tensor_target,
                                              item_attention_category6_tensor_target, item_attention_category7_tensor_target,
                                              item_attention_category8_tensor_target, item_attention_category9_tensor_target]

                item_representation_category0_tensor_target = self.mlp_list_for_item_representation[0](embedded_item_for_representation_of_target)
                item_representation_category1_tensor_target = self.mlp_list_for_item_representation[1](embedded_item_for_representation_of_target)
                item_representation_category2_tensor_target = self.mlp_list_for_item_representation[2](embedded_item_for_representation_of_target)
                item_representation_category3_tensor_target = self.mlp_list_for_item_representation[3](embedded_item_for_representation_of_target)
                item_representation_category4_tensor_target = self.mlp_list_for_item_representation[4](embedded_item_for_representation_of_target)
                item_representation_category5_tensor_target = self.mlp_list_for_item_representation[5](embedded_item_for_representation_of_target)
                item_representation_category6_tensor_target = self.mlp_list_for_item_representation[6](embedded_item_for_representation_of_target)
                item_representation_category7_tensor_target = self.mlp_list_for_item_representation[7](embedded_item_for_representation_of_target)
                item_representation_category8_tensor_target = self.mlp_list_for_item_representation[8](embedded_item_for_representation_of_target)
                item_representation_category9_tensor_target = self.mlp_list_for_item_representation[9](embedded_item_for_representation_of_target)
                item_representation_target_tensor_list = [item_representation_category0_tensor_target,
                                                   item_representation_category1_tensor_target,
                                                   item_representation_category2_tensor_target,
                                                   item_representation_category3_tensor_target,
                                                   item_representation_category4_tensor_target,
                                                   item_representation_category5_tensor_target,
                                                   item_representation_category6_tensor_target,
                                                   item_representation_category7_tensor_target,
                                                   item_representation_category8_tensor_target,
                                                   item_representation_category9_tensor_target]

                embedded_item_for_attention_of_target_result = []
                embedded_item_for_representation_of_target_result = []
                for i in range(batch_size):
                    temp_category = batch_data['cid'][i] % 3
                    embedded_item_for_attention_of_target_result.append(item_attention_target_tensor_list[temp_category][i])
                    embedded_item_for_representation_of_target_result.append(item_representation_target_tensor_list[temp_category][i])
                embedded_item_for_attention_of_target_result = torch.stack(embedded_item_for_attention_of_target_result)
                embedded_item_for_representation_of_target_result = torch.stack(embedded_item_for_representation_of_target_result)

                result_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target_result, embedded_category_for_attention_of_target_result, embedded_time_for_attention_of_target_result), dim=1)
                result_for_representation_of_target = torch.cat((embedded_item_for_representation_of_target_result, embedded_category_for_representation_of_target_result, embedded_time_for_representation_of_target_result), dim=1)
                result_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior_result, embedded_category_for_attention_of_behavior_result, embedded_time_for_attention_of_behavior_result), dim=2)
                result_for_representation_of_behavior = torch.cat((embedded_item_for_representation_of_behavior_result, embedded_category_for_representation_of_behavior_result, embedded_time_for_representation_of_behavior_result), dim=2)
            else:
                if self.mlp_for_item_attention_of_target is not None:
                    embedded_item_for_attention_of_target_result = self.mlp_for_item_attention_of_target(embedded_item_for_attention_of_target)
                    # assert torch.equal(embedded_item_for_attention_of_target_result, embedded_item_for_attention_of_target)
                else:
                    embedded_item_for_attention_of_target_result = embedded_item_for_attention_of_target

                if self.mlp_for_category_attention_of_target is not None:
                    embedded_category_for_attention_of_target_result = self.mlp_for_category_attention_of_target(embedded_category_for_attention_of_target)
                else:
                    embedded_category_for_attention_of_target_result = embedded_category_for_attention_of_target

                if self.mlp_for_time_attention_of_target is not None:
                    embedded_time_for_attention_of_target_result = self.mlp_for_time_attention_of_target(embedded_time_for_attention_of_target)
                else:
                    embedded_time_for_attention_of_target_result = embedded_time_for_attention_of_target

                if self.mlp_for_item_representation_of_target is not None:
                    embedded_item_for_representation_of_target_result = self.mlp_for_item_representation_of_target(embedded_item_for_representation_of_target)
                else:
                    embedded_item_for_representation_of_target_result = embedded_item_for_representation_of_target

                if self.mlp_for_category_representation_of_target is not None:
                    embedded_category_for_representation_of_target_result = self.mlp_for_category_representation_of_target(embedded_category_for_representation_of_target)
                else:
                    embedded_category_for_representation_of_target_result = embedded_category_for_representation_of_target

                if self.mlp_for_time_representation_of_target is not None:
                    embedded_time_for_representation_of_target_result = self.mlp_for_time_representation_of_target(embedded_time_for_representation_of_target)
                else:
                    embedded_time_for_representation_of_target_result = embedded_time_for_representation_of_target

                if self.mlp_for_item_attention_of_behavior is not None:
                    embedded_item_for_attention_of_behavior_result = self.mlp_for_item_attention_of_behavior(embedded_item_for_attention_of_behavior)
                else:
                    embedded_item_for_attention_of_behavior_result = embedded_item_for_attention_of_behavior

                if self.mlp_for_category_attention_of_behavior is not None:
                    embedded_category_for_attention_of_behavior_result = self.mlp_for_category_attention_of_behavior(embedded_category_for_attention_of_behavior)
                else:
                    embedded_category_for_attention_of_behavior_result = embedded_category_for_attention_of_behavior

                if self.mlp_for_time_attention_of_behavior is not None:
                    embedded_time_for_attention_of_behavior_result = self.mlp_for_time_attention_of_behavior(embedded_time_for_attention_of_behavior)
                else:
                    embedded_time_for_attention_of_behavior_result = embedded_time_for_attention_of_behavior

                if self.mlp_for_item_representation_of_behavior is not None:
                    embedded_item_for_representation_of_behavior_result = self.mlp_for_item_representation_of_behavior(embedded_item_for_representation_of_behavior)
                else:
                    embedded_item_for_representation_of_behavior_result = embedded_item_for_representation_of_behavior

                if self.mlp_for_category_representation_of_behavior is not None:
                    embedded_category_for_representation_of_behavior_result = self.mlp_for_category_representation_of_behavior(embedded_category_for_representation_of_behavior)
                else:
                    embedded_category_for_representation_of_behavior_result = embedded_category_for_representation_of_behavior

                if self.mlp_for_time_representation_of_behavior is not None:
                    embedded_time_for_representation_of_behavior_result = self.mlp_for_time_representation_of_behavior(embedded_time_for_representation_of_behavior)
                else:
                    embedded_time_for_representation_of_behavior_result = embedded_time_for_representation_of_behavior

                if self.use_time:
                    if self.use_time_mode == "concat":
                        result_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target_result, embedded_category_for_attention_of_target_result, embedded_time_for_attention_of_target_result), dim=1)
                        result_for_representation_of_target = torch.cat((embedded_item_for_representation_of_target_result, embedded_category_for_representation_of_target_result, embedded_time_for_representation_of_target_result), dim=1)
                        result_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior_result, embedded_category_for_attention_of_behavior_result, embedded_time_for_attention_of_behavior_result), dim=2)
                        result_for_representation_of_behavior = torch.cat((embedded_item_for_representation_of_behavior_result, embedded_category_for_representation_of_behavior_result, embedded_time_for_representation_of_behavior_result), dim=2)
                    else:
                        result_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target_result, embedded_category_for_attention_of_target_result), dim=1) + embedded_time_for_attention_of_target_result
                        result_for_representation_of_target = torch.cat((embedded_item_for_representation_of_target_result, embedded_category_for_representation_of_target_result), dim=1) + embedded_time_for_representation_of_target_result
                        result_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior_result, embedded_category_for_attention_of_behavior_result), dim=2) + embedded_time_for_attention_of_behavior_result
                        result_for_representation_of_behavior = torch.cat((embedded_item_for_representation_of_behavior_result, embedded_category_for_representation_of_behavior_result), dim=2) + embedded_time_for_representation_of_behavior_result
                else:
                    result_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target_result, embedded_category_for_attention_of_target_result), dim=1)
                    result_for_representation_of_target = torch.cat((embedded_item_for_representation_of_target_result, embedded_category_for_representation_of_target_result), dim=1)
                    result_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior_result, embedded_category_for_attention_of_behavior_result), dim=2)
                    result_for_representation_of_behavior = torch.cat((embedded_item_for_representation_of_behavior_result, embedded_category_for_representation_of_behavior_result), dim=2)
        
        else:
            if self.use_time:
                if self.use_time_mode == "concat":
                    embedded_result_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target, embedded_category_for_attention_of_target, embedded_time_for_attention_of_target), dim=1)
                    embedded_result_for_representation_of_target = torch.cat((embedded_item_for_representation_of_target, embedded_category_for_representation_of_target, embedded_time_for_representation_of_target), dim=1)
                    embedded_result_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior, embedded_category_for_attention_of_behavior, embedded_time_for_attention_of_behavior), dim=2)
                    embedded_result_for_representation_of_behavior = torch.cat((embedded_item_for_representation_of_behavior, embedded_category_for_representation_of_behavior, embedded_time_for_representation_of_behavior), dim=2)
                elif self.use_time_mode == "NOVA" or self.use_time_mode == "DIF":
                    embedded_result_for_attention_of_target = embedded_item_for_attention_of_target + embedded_category_for_attention_of_target + embedded_time_for_attention_of_target
                    embedded_result_for_representation_of_target = embedded_item_for_representation_of_target + embedded_category_for_representation_of_target + embedded_time_for_representation_of_target
                    embedded_result_for_attention_of_behavior = embedded_item_for_attention_of_behavior + embedded_category_for_attention_of_behavior + embedded_time_for_attention_of_behavior
                    embedded_result_for_representation_of_behavior = embedded_item_for_representation_of_behavior + embedded_category_for_representation_of_behavior + embedded_time_for_representation_of_behavior
                else:
                    embedded_result_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target, embedded_category_for_attention_of_target), dim=1) + embedded_time_for_attention_of_target
                    embedded_result_for_representation_of_target = torch.cat((embedded_item_for_representation_of_target, embedded_category_for_representation_of_target), dim=1) + embedded_time_for_representation_of_target
                    embedded_result_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior, embedded_category_for_attention_of_behavior), dim=2) + embedded_time_for_attention_of_behavior
                    embedded_result_for_representation_of_behavior = torch.cat((embedded_item_for_representation_of_behavior, embedded_category_for_representation_of_behavior), dim=2) + embedded_time_for_representation_of_behavior
                    if self.use_decoupled_embedding:
                        embedded_result_for_attention_of_target_2 = torch.cat((embedded_item_for_attention_of_target_2, embedded_category_for_attention_of_target_2), dim=1) + embedded_time_for_attention_of_target_2
                        embedded_result_for_representation_of_target_2 = torch.cat((embedded_item_for_representation_of_target_2, embedded_category_for_representation_of_target_2), dim=1) + embedded_time_for_representation_of_target_2
                        embedded_result_for_attention_of_behavior_2 = torch.cat((embedded_item_for_attention_of_behavior_2, embedded_category_for_attention_of_behavior_2), dim=2) + embedded_time_for_attention_of_behavior_2
                        embedded_result_for_representation_of_behavior_2 = torch.cat((embedded_item_for_representation_of_behavior_2, embedded_category_for_representation_of_behavior_2), dim=2) + embedded_time_for_representation_of_behavior_2
            else:
                embedded_result_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target, embedded_category_for_attention_of_target), dim=1)
                embedded_result_for_representation_of_target = torch.cat((embedded_item_for_representation_of_target, embedded_category_for_representation_of_target), dim=1)
                embedded_result_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior, embedded_category_for_attention_of_behavior), dim=2)
                embedded_result_for_representation_of_behavior = torch.cat((embedded_item_for_representation_of_behavior, embedded_category_for_representation_of_behavior), dim=2)

            if self.mlp_for_attention_of_target is not None:
                result_for_attention_of_target = self.mlp_for_attention_of_target(embedded_result_for_attention_of_target)
            else:
                result_for_attention_of_target = embedded_result_for_attention_of_target
                if self.use_decoupled_embedding:
                    result_for_attention_of_target_2 = embedded_result_for_attention_of_target_2

            if self.mlp_for_representation_of_target is not None:
                result_for_representation_of_target = self.mlp_for_representation_of_target(embedded_result_for_representation_of_target)
            else:
                result_for_representation_of_target = embedded_result_for_representation_of_target
                if self.use_decoupled_embedding:
                    result_for_representation_of_target_2 = embedded_result_for_representation_of_target_2

            if self.mlp_for_attention_of_behavior is not None:
                result_for_attention_of_behavior = self.mlp_for_attention_of_behavior(embedded_result_for_attention_of_behavior)
            else:
                result_for_attention_of_behavior = embedded_result_for_attention_of_behavior
                if self.use_decoupled_embedding:
                    result_for_attention_of_behavior_2 = embedded_result_for_attention_of_behavior_2

            if self.mlp_for_representation_of_behavior is not None:
                result_for_representation_of_behavior = self.mlp_for_representation_of_behavior(embedded_result_for_representation_of_behavior)
            else:
                result_for_representation_of_behavior = embedded_result_for_representation_of_behavior
                if self.use_decoupled_embedding:
                    result_for_representation_of_behavior_2 = embedded_result_for_representation_of_behavior_2

        assert len(result_for_attention_of_target.shape) == 2
        assert len(result_for_representation_of_target.shape) == 2
        assert len(result_for_attention_of_behavior.shape) == 3
        assert len(result_for_representation_of_behavior.shape) == 3

        # predict
        if not self.only_odot:
            if self.use_time_mode == 'decoupled2' and self.use_decoupled_embedding:
                features = [torch.cat((result_for_representation_of_target, result_for_representation_of_target_2), dim=1)]
            elif self.use_time_mode == 'decoupled2' and not self.use_decoupled_embedding:
                features = [torch.cat((result_for_representation_of_target, result_for_representation_of_target), dim=1)]
            elif self.use_time_mode == 'decoupled' and self.use_decoupled_embedding:
                features = [result_for_representation_of_target + result_for_representation_of_target_2]
            elif self.use_time_mode == 'decoupled' and not self.use_decoupled_embedding:
                features = [result_for_representation_of_target + result_for_representation_of_target]
            elif self.use_time_mode == 'NOVA' or self.use_time_mode == 'DIF':
                features = [embedded_item_for_representation_of_target]
            elif self.use_time_mode == 'complete_decoupling':
                features = [torch.cat((embedded_time_for_representation_of_target,torch.cat((embedded_item_for_representation_of_target, embedded_category_for_representation_of_target), dim=1) ), dim=1)]
            else:
                features = [result_for_representation_of_target]
        else:
            features = []
        if self.short_seq_split:
            seq_split = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in self.short_seq_split.split(",")]
            for idx, (left_idx, right_idx) in enumerate(seq_split):
                short_seq_mask = mask[:, left_idx:right_idx]
                short_seq_embed = attention(result_for_attention_of_target, result_for_attention_of_behavior[:, left_idx:right_idx], result_for_representation_of_behavior[:, left_idx:right_idx], short_seq_mask, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, use_time_mode=self.use_time_mode, model_name=self.model_name, ETA_H_metirc=self.ETA_H_metrics)
                if not self.only_odot:
                    features.append(short_seq_embed)
                if self.use_cross_feature:
                    features.append(torch.mul(result_for_representation_of_target, short_seq_embed))

        if self.long_seq_split:
            seq_split = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in self.long_seq_split.split(",")]
            for idx, (left_idx, right_idx) in enumerate(seq_split):
                long_seq_mask = mask[:, left_idx:right_idx]
                long_seq_average_embed = torch.sum(torch.mul(result_for_representation_of_behavior[:, left_idx:right_idx], long_seq_mask.unsqueeze(2)), dim=1) / (torch.sum(long_seq_mask, dim=1, keepdim=True) + 1.0)

                if self.model_name not in ["SDIM", "TWIN_V2"]:
                    if self.use_time_mode == "decoupled" or self.use_time_mode == "decoupled2":
                        embedded_sideinfo_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target, embedded_category_for_attention_of_target), dim=1)
                        embedded_sideinfo_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior, embedded_category_for_attention_of_behavior), dim=2)
                        long_seq_embed_1, original_attn_scores_1, _ = attention(embedded_time_for_attention_of_target, embedded_time_for_attention_of_behavior[:, left_idx:right_idx], result_for_representation_of_behavior[:, left_idx:right_idx], long_seq_mask, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, return_scores=True, use_time_mode=self.use_time_mode, model_name=self.model_name, ETA_H_metirc=self.ETA_H_metrics, state="gsu", mode=mode, soft_search=self.soft_search)
                        if self.use_decoupled_embedding:
                            embedded_sideinfo_for_attention_of_target_2 = torch.cat((embedded_item_for_attention_of_target_2, embedded_category_for_attention_of_target_2), dim=1)
                            embedded_sideinfo_for_attention_of_behavior_2 = torch.cat((embedded_item_for_attention_of_behavior_2, embedded_category_for_attention_of_behavior_2), dim=2)
                            long_seq_embed_2, original_attn_scores_2, _ = attention(embedded_sideinfo_for_attention_of_target_2, embedded_sideinfo_for_attention_of_behavior_2[:, left_idx:right_idx], result_for_representation_of_behavior_2[:, left_idx:right_idx], long_seq_mask, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, return_scores=True, use_time_mode=self.use_time_mode, model_name=self.model_name, ETA_H_metirc=self.ETA_H_metrics, state="gsu", mode=mode, soft_search=self.soft_search)
                        else:
                            long_seq_embed_2, original_attn_scores_2, _ = attention(embedded_sideinfo_for_attention_of_target, embedded_sideinfo_for_attention_of_behavior[:, left_idx:right_idx], result_for_representation_of_behavior[:, left_idx:right_idx], long_seq_mask, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, return_scores=True, use_time_mode=self.use_time_mode, model_name=self.model_name, ETA_H_metirc=self.ETA_H_metrics, state="gsu", mode=mode, soft_search=self.soft_search)
                    elif self.use_time_mode == 'complete_decoupling':
                        embedded_sideinfo_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target, embedded_category_for_attention_of_target), dim=1)
                        embedded_sideinfo_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior, embedded_category_for_attention_of_behavior), dim=2)
                        embedded_sideinfo_for_representation_of_target = torch.cat((embedded_item_for_representation_of_target, embedded_category_for_representation_of_target), dim=1)
                        embedded_sideinfo_for_representation_of_behavior = torch.cat((embedded_item_for_representation_of_behavior, embedded_category_for_representation_of_behavior), dim=2)                        
                        
                        long_seq_embed_1, original_attn_scores_1, _ = attention(embedded_time_for_attention_of_target, embedded_time_for_attention_of_behavior[:, left_idx:right_idx], embedded_time_for_representation_of_behavior[:, left_idx:right_idx], long_seq_mask, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, return_scores=True, use_time_mode=self.use_time_mode, model_name=self.model_name, ETA_H_metirc=self.ETA_H_metrics, state="gsu", mode=mode, soft_search=self.soft_search)
                        long_seq_embed_2, original_attn_scores_2, _ = attention(embedded_sideinfo_for_attention_of_target, embedded_sideinfo_for_attention_of_behavior[:, left_idx:right_idx], embedded_sideinfo_for_representation_of_behavior[:, left_idx:right_idx], long_seq_mask, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, return_scores=True, use_time_mode=self.use_time_mode, model_name=self.model_name, ETA_H_metirc=self.ETA_H_metrics, state="gsu", mode=mode, soft_search=self.soft_search)
                    elif self.use_time_mode == "NOVA":
                        long_seq_embed, original_attn_scores, _ = attention(result_for_attention_of_target, result_for_attention_of_behavior[:, left_idx:right_idx], embedded_item_for_representation_of_behavior[:, left_idx:right_idx], long_seq_mask, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, return_scores=True, use_time_mode=self.use_time_mode, model_name=self.model_name, ETA_H_metirc=self.ETA_H_metrics, state="gsu", mode=mode, soft_search=self.soft_search)
                    elif self.use_time_mode == "DIF":
                        long_seq_embed_1, original_attn_scores_1, _ = attention(embedded_time_for_attention_of_target, embedded_time_for_attention_of_behavior[:, left_idx:right_idx], embedded_item_for_representation_of_behavior[:, left_idx:right_idx], long_seq_mask, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, return_scores=True, use_time_mode=self.use_time_mode, model_name=self.model_name, ETA_H_metirc=self.ETA_H_metrics, state="gsu", mode=mode, soft_search=self.soft_search)
                        long_seq_embed_2, original_attn_scores_2, _ = attention(embedded_category_for_attention_of_target, embedded_category_for_attention_of_behavior[:, left_idx:right_idx], embedded_item_for_representation_of_behavior[:, left_idx:right_idx], long_seq_mask, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, return_scores=True, use_time_mode=self.use_time_mode, model_name=self.model_name, ETA_H_metirc=self.ETA_H_metrics, state="gsu", mode=mode, soft_search=self.soft_search)
                        long_seq_embed_3, original_attn_scores_3, _ = attention(embedded_item_for_attention_of_target, embedded_item_for_attention_of_behavior[:, left_idx:right_idx], embedded_item_for_representation_of_behavior[:, left_idx:right_idx], long_seq_mask, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, return_scores=True, use_time_mode=self.use_time_mode, model_name=self.model_name, ETA_H_metirc=self.ETA_H_metrics, state="gsu", mode=mode, soft_search=self.soft_search)
                        long_seq_embed = long_seq_embed_1 + long_seq_embed_2 + long_seq_embed_3
                        original_attn_scores = original_attn_scores_1 + original_attn_scores_2 + original_attn_scores_3
                    else:
                        long_seq_embed, original_attn_scores, _ = attention(result_for_attention_of_target, result_for_attention_of_behavior[:, left_idx:right_idx], result_for_representation_of_behavior[:, left_idx:right_idx], long_seq_mask, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, return_scores=True, use_time_mode=self.use_time_mode, model_name=self.model_name, ETA_H_metirc=self.ETA_H_metrics, state="gsu", mode=mode, soft_search=self.soft_search)
                    if only_return_scores:
                        return original_attn_scores
                    # get aux_loss by auxiliary_mlp
                    if self.use_cross_feature:
                        if self.only_odot:
                            aux_input_embed = torch.mul(result_for_representation_of_target, long_seq_embed)
                        else:
                            if self.use_time_mode == "decoupled2":
                                if self.use_decoupled_embedding:
                                    aux_input_embed = torch.cat((result_for_representation_of_target, result_for_representation_of_target_2, long_seq_embed_1, long_seq_embed_2, torch.mul(result_for_representation_of_target, long_seq_embed_1), torch.mul(result_for_representation_of_target_2, long_seq_embed_2)), dim=1)
                                else:
                                    aux_input_embed = torch.cat((result_for_representation_of_target, result_for_representation_of_target, long_seq_embed_1, long_seq_embed_2, torch.mul(result_for_representation_of_target, long_seq_embed_1), torch.mul(result_for_representation_of_target, long_seq_embed_2)), dim=1)
                            elif self.use_time_mode == "decoupled":
                                if self.use_decoupled_embedding:
                                    aux_input_embed = torch.cat((result_for_representation_of_target + result_for_representation_of_target_2, long_seq_embed_1 + long_seq_embed_2, torch.mul(result_for_representation_of_target, long_seq_embed_1) + torch.mul(result_for_representation_of_target_2, long_seq_embed_2)), dim=1)
                                else:
                                    aux_input_embed = torch.cat((result_for_representation_of_target + result_for_representation_of_target, long_seq_embed_1 + long_seq_embed_2, torch.mul(result_for_representation_of_target, long_seq_embed_1) + torch.mul(result_for_representation_of_target, long_seq_embed_2)), dim=1)
                            elif self.use_time_mode == 'complete_decoupling':
                                aux_input_embed = torch.cat((embedded_time_for_representation_of_target, embedded_sideinfo_for_representation_of_target, long_seq_embed_1, long_seq_embed_2, torch.mul(embedded_time_for_representation_of_target, long_seq_embed_1), torch.mul(embedded_sideinfo_for_representation_of_target, long_seq_embed_2)), dim=1)
                            elif self.use_time_mode == "NOVA" or self.use_time_mode == "DIF":
                                aux_input_embed = torch.cat((embedded_item_for_representation_of_target, long_seq_embed, torch.mul(embedded_item_for_representation_of_target, long_seq_embed)), dim=1)
                            else:
                                aux_input_embed = torch.cat((result_for_representation_of_target, long_seq_embed, torch.mul(result_for_representation_of_target, long_seq_embed)), dim=1)
                    else:
                        aux_input_embed = torch.cat((result_for_representation_of_target, long_seq_embed), dim=1)
                    assert aux_input_embed.shape == (item_ids.shape[0], self.auxiliary_mlp_input_dimension)
                    aux_y_hat = self.auxiliary_mlp(aux_input_embed)
                    assert aux_y_hat.shape == (item_ids.shape[0], 2)
                    aux_predicted_probability = nn.functional.softmax(aux_y_hat, dim=1) + 1e-8
                    aux_loss = -torch.mean(torch.log(torch.sum(torch.mul(aux_predicted_probability, target), dim=1)))

                    if self.soft_search:
                        if self.model_name == "ETA":
                            _, top_k_indices = torch.topk(original_attn_scores, self.top_k, dim=1)
                            mask_for_soft_search = torch.zeros_like(long_seq_mask)
                            for k in range(top_k_indices.shape[0]):
                                mask_for_soft_search[k, top_k_indices[k]] = 1
                            assert torch.all(torch.sum(mask_for_soft_search, dim=1) == self.top_k)
                            _, scores_for_soft_search, _ = attention(result_for_attention_of_target, result_for_attention_of_behavior[:, left_idx:right_idx], result_for_representation_of_behavior[:, left_idx:right_idx], mask_for_soft_search, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, return_scores=True, use_time_mode=self.use_time_mode, model_name=self.model_name, ETA_H_metirc=self.ETA_H_metrics, mode=mode, state="esu")
                            attention_distribution_for_soft_search = nn.functional.softmax(scores_for_soft_search, dim=1)
                            soft_search_long_seq_embed = torch.matmul(attention_distribution_for_soft_search.unsqueeze(1), result_for_representation_of_behavior[:, left_idx:right_idx])
                            long_seq_embed = soft_search_long_seq_embed.squeeze(1)
                        else:
                            if self.use_time_mode == "decoupled" or self.use_time_mode == "decoupled2":
                                _, top_k_indices_1 = torch.topk(original_attn_scores_1, self.top_k, dim=1)
                                _, top_k_indices_2 = torch.topk(original_attn_scores_2, self.top_k, dim=1)
                                mask_for_soft_search_1 = torch.zeros_like(long_seq_mask)
                                mask_for_soft_search_2 = torch.zeros_like(long_seq_mask)
                                for k in range(top_k_indices_1.shape[0]):
                                    mask_for_soft_search_1[k, top_k_indices_1[k]] = 1
                                    mask_for_soft_search_2[k, top_k_indices_2[k]] = 1
                                assert torch.all(torch.sum(mask_for_soft_search_1, dim=1) == self.top_k)
                                scores_for_soft_search_1 = torch.where(mask_for_soft_search_1.type(torch.bool), original_attn_scores_1, torch.ones_like(original_attn_scores_1).to(original_attn_scores_1.device) * (- 2 ** 32 + 1))
                                scores_for_soft_search_2 = torch.where(mask_for_soft_search_2.type(torch.bool), original_attn_scores_2, torch.ones_like(original_attn_scores_2).to(original_attn_scores_2.device) * (- 2 ** 32 + 1))
                                attention_distribution_for_soft_search_1 = nn.functional.softmax(scores_for_soft_search_1, dim=1)
                                soft_search_long_seq_embed_1 = torch.matmul(attention_distribution_for_soft_search_1.unsqueeze(1), result_for_representation_of_behavior[:, left_idx:right_idx]).squeeze(1)
                                attention_distribution_for_soft_search_2 = nn.functional.softmax(scores_for_soft_search_2, dim=1)
                                if self.use_decoupled_embedding :
                                    soft_search_long_seq_embed_2 = torch.matmul(attention_distribution_for_soft_search_2.unsqueeze(1), result_for_representation_of_behavior_2[:, left_idx:right_idx]).squeeze(1)
                                else:
                                    soft_search_long_seq_embed_2 = torch.matmul(attention_distribution_for_soft_search_2.unsqueeze(1), result_for_representation_of_behavior[:, left_idx:right_idx]).squeeze(1)
                            elif self.use_time_mode == 'complete_decoupling':
                                _, top_k_indices_1 = torch.topk(original_attn_scores_1, self.top_k, dim=1)
                                _, top_k_indices_2 = torch.topk(original_attn_scores_2, self.top_k, dim=1)
                                mask_for_soft_search_1 = torch.zeros_like(long_seq_mask)
                                mask_for_soft_search_2 = torch.zeros_like(long_seq_mask)
                                for k in range(top_k_indices_1.shape[0]):
                                    mask_for_soft_search_1[k, top_k_indices_1[k]] = 1
                                    mask_for_soft_search_2[k, top_k_indices_2[k]] = 1
                                assert torch.all(torch.sum(mask_for_soft_search_1, dim=1) == self.top_k)
                                scores_for_soft_search_1 = torch.where(mask_for_soft_search_1.type(torch.bool), original_attn_scores_1, torch.ones_like(original_attn_scores_1).to(original_attn_scores_1.device) * (- 2 ** 32 + 1))
                                scores_for_soft_search_2 = torch.where(mask_for_soft_search_2.type(torch.bool), original_attn_scores_2, torch.ones_like(original_attn_scores_2).to(original_attn_scores_2.device) * (- 2 ** 32 + 1))
                                attention_distribution_for_soft_search_1 = nn.functional.softmax(scores_for_soft_search_1, dim=1)
                                soft_search_long_seq_embed_1 = torch.matmul(attention_distribution_for_soft_search_1.unsqueeze(1), embedded_time_for_representation_of_behavior[:, left_idx:right_idx]).squeeze(1)
                                attention_distribution_for_soft_search_2 = nn.functional.softmax(scores_for_soft_search_2, dim=1)
                                soft_search_long_seq_embed_2 = torch.matmul(attention_distribution_for_soft_search_2.unsqueeze(1), embedded_sideinfo_for_representation_of_behavior[:, left_idx:right_idx]).squeeze(1)

                            elif self.use_time_mode == "NOVA" or self.use_time_mode == "DIF":
                                _, top_k_indices = torch.topk(original_attn_scores, self.top_k, dim=1)
                                mask_for_soft_search = torch.zeros_like(long_seq_mask)
                                for k in range(top_k_indices.shape[0]):
                                    mask_for_soft_search[k, top_k_indices[k]] = 1
                                assert torch.all(torch.sum(mask_for_soft_search, dim=1) == self.top_k)
                                scores_for_soft_search = torch.where(mask_for_soft_search.type(torch.bool), original_attn_scores, torch.ones_like(original_attn_scores).to(original_attn_scores.device) * (- 2**32 + 1))
                                attention_distribution_for_soft_search = nn.functional.softmax(scores_for_soft_search, dim=1)
                                soft_search_long_seq_embed = torch.matmul(attention_distribution_for_soft_search.unsqueeze(1), embedded_item_for_representation_of_behavior[:, left_idx:right_idx])
                                long_seq_embed = soft_search_long_seq_embed.squeeze(1)
                            else:
                                _, top_k_indices = torch.topk(original_attn_scores, self.top_k, dim=1)
                                mask_for_soft_search = torch.zeros_like(long_seq_mask)
                                for k in range(top_k_indices.shape[0]):
                                    mask_for_soft_search[k, top_k_indices[k]] = 1
                                assert torch.all(torch.sum(mask_for_soft_search, dim=1) == self.top_k)
                                scores_for_soft_search = torch.where(mask_for_soft_search.type(torch.bool), original_attn_scores, torch.ones_like(original_attn_scores).to(original_attn_scores.device) * (- 2**32 + 1))
                                attention_distribution_for_soft_search = nn.functional.softmax(scores_for_soft_search, dim=1)
                                soft_search_long_seq_embed = torch.matmul(attention_distribution_for_soft_search.unsqueeze(1), result_for_representation_of_behavior[:, left_idx:right_idx])
                                long_seq_embed = soft_search_long_seq_embed.squeeze(1)
                else:  # SDIM, TWIN v2
                    long_seq_embed = attention(result_for_attention_of_target, result_for_attention_of_behavior[:, left_idx:right_idx], result_for_representation_of_behavior[:, left_idx:right_idx], long_seq_mask, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, return_scores=True, use_time_mode=self.use_time_mode, model_name=self.model_name, ETA_H_metirc=self.ETA_H_metrics, SDIM_H_metric=self.SDIM_H_metrics, mode=mode, history_category=history_category_ids, item_embedding_dim=self.item_embedding_dim, top_k=self.top_k)

                if not self.only_odot:
                    if self.use_time_mode == "decoupled":
                        features.append(soft_search_long_seq_embed_1 + soft_search_long_seq_embed_2)
                    elif self.use_time_mode == "decoupled2" or self.use_time_mode == 'complete_decoupling':
                        features.append(soft_search_long_seq_embed_1)
                        features.append(soft_search_long_seq_embed_2)
                    else:
                        features.append(long_seq_embed)
                    if self.use_long_seq_average:
                        features.append(long_seq_average_embed)
                if self.use_cross_feature:
                    if self.use_time_mode == "decoupled":
                        if self.use_decoupled_embedding :
                            features.append(torch.mul(result_for_representation_of_target, soft_search_long_seq_embed_1) + torch.mul(result_for_representation_of_target_2, soft_search_long_seq_embed_2))
                        else:
                            features.append(torch.mul(result_for_representation_of_target, soft_search_long_seq_embed_1) + torch.mul(result_for_representation_of_target, soft_search_long_seq_embed_2))
                    elif self.use_time_mode == "decoupled2":
                        if self.use_decoupled_embedding :
                            features.append(torch.mul(result_for_representation_of_target, soft_search_long_seq_embed_1))
                            features.append(torch.mul(result_for_representation_of_target_2, soft_search_long_seq_embed_2))
                        else:
                            features.append(torch.mul(result_for_representation_of_target, soft_search_long_seq_embed_1))
                            features.append(torch.mul(result_for_representation_of_target, soft_search_long_seq_embed_2))
                    elif self.use_time_mode == 'complete_decoupling':
                        features.append(torch.mul(embedded_time_for_representation_of_target, soft_search_long_seq_embed_1))
                        features.append(torch.mul(embedded_sideinfo_for_representation_of_target, soft_search_long_seq_embed_2))
                    elif self.use_time_mode == "NOVA" or self.use_time_mode == "DIF":
                        features.append(torch.mul(embedded_item_for_representation_of_target, long_seq_embed))
                    else:
                        features.append(torch.mul(result_for_representation_of_target, long_seq_embed))
                    if self.use_long_seq_average:
                        features.append(torch.mul(result_for_representation_of_target, long_seq_average_embed))

        # logging.info(features)
        features = torch.cat(features, dim=1)
        if only_return_representation:
            return features
        assert features.shape == (item_ids.shape[0], self.main_mlp_input_dimension)
        main_predicted_y = self.main_mlp(features)
        main_predicted_probility = nn.functional.softmax(main_predicted_y, dim=1) + 1e-8

        ctr_loss = -torch.mean(torch.log(torch.sum(torch.mul(main_predicted_probility, target), dim=1)))

        if self.use_aux_loss:
            return main_predicted_probility, ctr_loss+aux_loss, aux_loss
        else:
            return main_predicted_probility, ctr_loss
