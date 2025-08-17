## Long-Sequential-Modeling

#### 数据集处理
见[README](./preprocess/README.md)
#### 对各组件进行离线实验（以alipay数据集为例）

##### baseline
###### TIN
`python train_pytorch.py --use_time_mode add --use_temporal_encoding_mode None --top_k 40 --category_embedding_dim 32 --time_embedding_dim 64`
###### DIF
`python train_pytorch.py --use_time_mode DIF --use_temporal_encoding_mode None --top_k 40 --category_embedding_dim 64 --time_embedding_dim 64`
###### NOVA
`python train_pytorch.py --use_time_mode NOVA --use_temporal_encoding_mode None --top_k 40 --category_embedding_dim 64 --time_embedding_dim 64`
##### Decoupled Side Info TIN (DSI-TIN) for Ranking
###### TIN+DSI_sum
`python train_pytorch.py --use_time_mode decoupled --use_temporal_encoding_mode None --top_k 20 --category_embedding_dim 32 --time_embedding_dim 64`
###### TIN+DSI_concat
`python train_pytorch.py --use_time_mode decoupled2 --use_temporal_encoding_mode None --top_k 20 --category_embedding_dim 32 --time_embedding_dim 64`

实验结果如下
![](./figure/Decoupled_Side_Info_TIN_offline_evaluation.png)
##### Target decoupled Temporal Encoding for Ranking
###### TIN+TdPE
`python train_pytorch.py --use_time_mode add --use_temporal_encoding_mode target --top_k 40 --category_embedding_dim 32 --time_embedding_dim 64`
###### TIN+dPE
`python train_pytorch.py --use_time_mode add --use_temporal_encoding_mode all --top_k 40 --category_embedding_dim 32 --time_embedding_dim 64`

实验结果如下
![](./figure/Decoupled_Temporal_Encoding_offline_evaluation.png)
