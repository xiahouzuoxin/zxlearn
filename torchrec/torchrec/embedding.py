import torch
import torch.nn as nn
import torch.nn.init as init

class DynamicEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, 
                 scale_grad_by_freq=False, sparse=False, _weight=None):
        super(DynamicEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, 
                                               norm_type, scale_grad_by_freq, sparse, _weight)

    def _expand_embeddings(self, new_num_embeddings):
        if new_num_embeddings <= self.num_embeddings:
            return
        else:
            # only init the expanded embeddings and keep the original embeddings weights
            new_embeddings = torch.empty(new_num_embeddings - self.num_embeddings, self.embedding_dim, 
                                        dtype=self.weight.dtype, device=self.weight.device)
            init.normal_(new_embeddings, mean=0, std=0.01)
            self.weight = nn.Parameter(torch.cat([self.weight, new_embeddings], dim=0))
            self.num_embeddings = new_num_embeddings

    def forward(self, input):
        if input.numel() == 0:
            raise ValueError("Indices tensor is empty")
        if input.min().item() < 0:
            raise ValueError("Indices contain negative values")
        max_index = input.max().item()
        self._expand_embeddings(max_index + 1)
        return super(DynamicEmbedding, self).forward(input)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_embeddings = state_dict[prefix + 'weight'].size(0)
        self._expand_embeddings(num_embeddings)
        if num_embeddings < self.num_embeddings:
            # load part of the weights from the state_dict as embedding size of state_dict is smaller than current embedding size
            state_dict[prefix + 'weight'] = torch.cat([state_dict[prefix + 'weight'], self.weight[num_embeddings:]], dim=0)
        super(DynamicEmbedding, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
