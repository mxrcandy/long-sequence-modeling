from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json

embedding_path = 'log/taobao/TIN/learned_embedding/attention_for_target_category.npy'
target_cate_num = 50
category_embeddings = np.load(embedding_path)

# 初始化 KMeans 聚类器
kmeans = KMeans(n_clusters=target_cate_num, random_state=42, n_init=10,)

# 拟合聚类器
kmeans.fit(category_embeddings)

# 获取聚类结果
cluster_labels = kmeans.labels_

# 创建一个字典，将每个类别映射到其对应的聚类中心
category_to_cluster_mapping = {i: int(cluster_labels[i])+1 for i in range(len(cluster_labels))}

# 将字典保存为 JSON 文件
with open('data/taobao/category_to_cluster_mapping.json', 'w') as f:
    json.dump(category_to_cluster_mapping, f, indent=4)


tsne = TSNE(n_components=2, random_state=42)
category_embeddings_2d = tsne.fit_transform(category_embeddings)

# 绘制聚类结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(category_embeddings_2d[:, 0], category_embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', s=50)
plt.colorbar(scatter, label='Cluster')
plt.title('K-Means Clustering Visualization')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.savefig('data/taobao/kmeans_clustering_visualization.png', dpi=300, bbox_inches='tight')