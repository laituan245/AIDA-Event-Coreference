import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils

from models.base import BaseModel, ScoreModule
from models.encoder import TransformerEncoder
from models.helpers import *

# BasicCorefModel (assuming ground truth event mentions are provided)
class BasicCorefModel(BaseModel):
    def __init__(self, configs):
        BaseModel.__init__(self, configs)

        self.encoder = TransformerEncoder(configs)
        self.pair_scorer = ScoreModule(self.get_pair_embs_size(),
                                      [configs['ffnn_size']] * configs['ffnn_depth'],
                                      configs['dropout_rate'])

        # Embeddings for additional features
        # Event type embedding (If use_event_type_features enabled)
        if configs['use_event_type_features']:
            self.type_embeddings = nn.Embedding(2, configs['feature_size'])

        # Initialize embeddings
        for name, param in self.named_parameters():
            if (not 'transformer' in name.lower()) and 'embedding' in name.lower():
                print('Re-initialize embedding {}'.format(name))
                param.data.uniform_(-0.1, 0.1)

        # Move model to device
        self.to(self.device)

    def forward(self, inst, is_training):
        self.train() if is_training else self.eval()

        input_ids = torch.tensor(inst.token_windows).to(self.device)
        input_masks = torch.tensor(inst.input_masks).to(self.device)
        mask_windows = torch.tensor(inst.mask_windows).to(self.device)
        num_windows, window_size = input_ids.size()

        # Apply the Transfomer encoder to get tokens features
        tokens_features = self.encoder(input_ids, input_masks, mask_windows,
                                       num_windows, window_size, is_training).squeeze()
        num_tokens = tokens_features.size()[0]

        # Compute word_features (averaging)
        word_features = []
        word_starts_indexes = inst.word_starts_indexes
        word_ends_indexes = word_starts_indexes[1:] + [num_tokens]
        word_features = get_span_emb(tokens_features, word_starts_indexes, word_ends_indexes)
        assert(word_features.size()[0] == inst.num_words)

        # Compute event_mention_features (averaging)
        if self.configs['use_groundtruth']:
            event_mentions = inst.event_mentions
        else:
            event_mentions = inst.pred_event_mentions
        event_mention_starts = [e['trigger']['start'] for e in event_mentions]
        event_mention_ends = [e['trigger']['end'] for e in event_mentions]
        event_mention_features = get_span_emb(word_features, event_mention_starts, event_mention_ends)
        assert(event_mention_features.size()[0] == len(event_mentions))

        # Compute event types features (if enabled)
        same_types_features = None
        if self.configs['use_event_type_features']:
            n = len(event_mentions)
            e_types = [e['event_type'] for e in event_mentions]
            same_types = torch.zeros((n, n)).to(self.device).long()
            for i in range(n):
                for j in range(n):
                    same_types[i, j] = int(e_types[i] == e_types[j])
            same_types_features = self.type_embeddings(same_types)

        # Compute pair features and score the pairs
        pair_features = self.get_pair_embs(event_mention_features, event_mentions)
        if not same_types_features is None:
            pair_features = torch.cat([pair_features, same_types_features], dim=-1)
        pair_scores = self.pair_scorer(pair_features)

        # Compute antecedent_scores
        k = len(event_mentions)
        span_range = torch.arange(0, k).to(self.device)
        antecedent_offsets = span_range.view(-1, 1) - span_range.view(1, -1)
        antecedents_mask = antecedent_offsets >= 1 # [k, k]
        antecedent_scores = pair_scores + torch.log(antecedents_mask.float())

        # Compute antecedent_labels
        candidate_cluster_ids = self.get_cluster_ids(event_mentions, inst.coreferential_pairs)
        same_cluster_indicator = candidate_cluster_ids.unsqueeze(0) == candidate_cluster_ids.unsqueeze(1)
        same_cluster_indicator = same_cluster_indicator & antecedents_mask

        non_dummy_indicator = (candidate_cluster_ids > -1).unsqueeze(1)
        pairwise_labels = same_cluster_indicator & non_dummy_indicator
        dummy_labels = ~pairwise_labels.any(1, keepdim=True)
        antecedent_labels = torch.cat([dummy_labels, pairwise_labels], 1)

        # Compute loss
        dummy_zeros = torch.zeros([k, 1]).to(self.device)
        antecedent_scores = torch.cat([dummy_zeros, antecedent_scores], dim=1)
        gold_scores = antecedent_scores + torch.log(antecedent_labels.float())
        log_norm = logsumexp(antecedent_scores, dim = 1)
        loss = torch.sum(log_norm - logsumexp(gold_scores, dim=1))

        # loss and preds
        top_antecedents = torch.arange(0, k).to(self.device)
        top_antecedents = top_antecedents.unsqueeze(0).repeat(k, 1)
        preds = [torch.tensor(event_mention_starts),
                 torch.tensor(event_mention_ends),
                 top_antecedents,
                 antecedent_scores]

        return loss, preds

    def get_cluster_ids(self, event_mentions, coreferential_pairs):
        cluster_ids = [-1] * len(event_mentions)
        nb_nonsingleton_clusters = 0
        for i in range(len(event_mentions)):
            mention_i = event_mentions[i]
            loc_i = (mention_i['trigger']['start'], mention_i['trigger']['end'])
            for j in range(i-1, -1, -1):
                mention_j = event_mentions[j]
                loc_j = (mention_j['trigger']['start'], mention_j['trigger']['end'])
                if ((loc_i, loc_j)) in coreferential_pairs:
                    if cluster_ids[j] > -1:
                        cluster_ids[i] = cluster_ids[j]
                    else:
                        cluster_ids[i] = cluster_ids[j] = nb_nonsingleton_clusters
                        nb_nonsingleton_clusters += 1
        return torch.tensor(cluster_ids).to(self.device)

    def get_pair_embs(self, event_features, event_mentions):
        n, d = event_features.size()
        features_list = []

        # Compute diff_embs and prod_embs
        src_embs = event_features.view(1, n, d).repeat([n, 1, 1])
        target_embs = event_features.view(n, 1, d).repeat([1, n, 1])
        prod_embds = src_embs * target_embs

        # Update features_list
        features_list.append(src_embs)
        features_list.append(target_embs)
        features_list.append(prod_embds)

        # Concatenation
        pair_embs = torch.cat(features_list, 2)

        return pair_embs

    def get_distance_features(self, locations, embeddings, nb_buckets):
        if type(locations) == list:
            locations = torch.tensor(locations).to(self.device)
        offsets = locations.view(-1, 1) - locations.view(1,-1)
        distance_buckets = utils.bucket_distance(offsets, nb_buckets)
        distance_features = embeddings(distance_buckets)
        return distance_features

    def get_span_emb_size(self):
        span_emb_size = self.encoder.transformer_hidden_size
        return span_emb_size

    def get_pair_embs_size(self):
        pair_embs_size = 3 * self.get_span_emb_size() # src_vector, target_vector, product_vector
        if self.configs['use_event_type_features']:
            pair_embs_size += self.configs['feature_size']
        return pair_embs_size
