import torch
import torch.nn as nn
import numpy as np

class UserIDOnlyModel(nn.Module):
    def __init__(self, item_n, cate_n, user_n, embedding_dim, batch_size, **kwargs):
        super().__init__()

        self.user_embedding = nn.Embedding(user_n, embedding_dim)
        self.item_embedding = nn.Embedding(item_n, embedding_dim)
        self.category_embedding = nn.Embedding(cate_n, embedding_dim)

        self.main_mlp = nn.Sequential(
            nn.BatchNorm1d(5 * embedding_dim),
            nn.Linear(5*embedding_dim, 200),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.Linear(200, 80),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.Linear(80, 2)
        )

    def forward(self, batch_data, mode="train"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        user_ids = torch.tensor(batch_data['uid'], dtype=torch.int32).to(device)
        item_ids = torch.tensor(batch_data['iid'], dtype=torch.int32).to(device)
        category_ids = torch.tensor(batch_data['cid'], dtype=torch.int32).to(device)
        target = torch.tensor(batch_data['target'], dtype=torch.int32).to(device)

        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        category_embedding = self.category_embedding(category_ids)

        features = [user_embedding, item_embedding, category_embedding, torch.mul(user_embedding, item_embedding),
                    torch.mul(user_embedding, category_embedding)]
        features = torch.cat(features, dim=1)

        main_predicted_y = self.main_mlp(features)
        main_predicted_probility = nn.functional.softmax(main_predicted_y, dim=1) + 1e-8
        ctr_loss = -torch.mean(torch.log(torch.sum(torch.mul(main_predicted_probility, target), dim=1)))
        return main_predicted_probility, ctr_loss

    def save(self, path):
        torch.save(self.state_dict(), path)
        print('model saved at %s' % path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print('model restored from %s' % path)