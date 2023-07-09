from abc import ABC
from kge.model.embedder.lookup_embedder import LookupEmbedder
from pydoc import locate
import torch
import numpy as np
import random


class BPRMFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 optimizer,
                 init,
                 embed_regularizer,
                 learning_rate,
                 embed_k,
                 l_w,
                 random_seed,
                 name="BPRMF",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.embed_regularizer = embed_regularizer

        # EMBEDDING INITIALIZER
        initializer_type = getattr(torch.nn.init, init.split('|')[0])
        initializer_params = {ktv.split('=')[0]: locate(ktv.split('=')[1])(ktv.split('=')[2])
                              for ktv in init.split('|')[1:]}

        self.Gu = LookupEmbedder(
            space='euclidean',
            regularize=embed_regularizer,
            regularize_weight=self.l_w,
            num_elements=self.num_users,
            latent_dim=self.embed_k,
            init=initializer_type,
            init_params=initializer_params,
            device=self.device
        )

        self.Gi = LookupEmbedder(
            space='euclidean',
            regularize=embed_regularizer,
            regularize_weight=self.l_w,
            num_elements=self.num_items,
            latent_dim=self.embed_k,
            init=initializer_type,
            init_params=initializer_params,
            device=self.device
        )

        # OPTIMIZER
        optimizer_type = optimizer.split('|')[0]
        optimizer_params = {ktv.split('=')[0]: locate(ktv.split('=')[1])(ktv.split('=')[2])
                            for ktv in optimizer.split('|')[1:]}
        optimizer_params['lr'] = self.learning_rate

        self.optimizer = getattr(torch.optim, optimizer_type)(self.parameters(), **optimizer_params)

    def forward(self, users, items, **kwargs):
        gamma_u = torch.squeeze(self.Gu.embed_all()[users]).to(self.device)
        gamma_i = torch.squeeze(self.Gi.embed_all()[items]).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, -1)

        return xui, gamma_u, gamma_i

    def predict(self, users, **kwargs):
        return torch.matmul(self.Gu.embeddings_all()[users].to(self.device),
                            torch.transpose(self.Gi.embeddings_all().to(self.device), 0, 1))

    def train_step(self, batch):
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos = self.forward(users=user[:, 0], items=pos[:, 0])
        xu_neg, _, gamma_i_neg = self.forward(users=user[:, 0], items=neg[:, 0])
        loss = torch.mean(torch.nn.functional.softplus(xu_neg - xu_pos))

        if self.embed_regularizer == 'none':
            reg_loss = self.l_w * (1 / 2) * (gamma_u.norm(2).pow(2) +
                                             gamma_i_pos.norm(2).pow(2) +
                                             gamma_i_neg.norm(2).pow(2)) / float(batch[0].shape[0])
        else:
            reg_loss = self.Gu.penalty(indexes=user if self.Gu.weighted else None) + \
                       self.Gi.penalty(indexes=pos if self.Gi.weighted else None) + \
                       self.Gi.penalty(indexes=neg if self.Gi.weighted else None)

        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
