# igat_predictor.py (Final Hybrid Edition)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class SubgraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, dropout=0.2): super().__init__(); self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True); self.norm1 = nn.LayerNorm(hidden_channels * num_heads); self.conv2 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout, concat=True); self.norm2 = nn.LayerNorm(hidden_channels * num_heads); self.conv3 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, concat=True, dropout=dropout); self.norm3 = nn.LayerNorm(out_channels); self.res_connection = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
    def forward(self, x, edge_index, batch_index): initial_x_pooled = global_mean_pool(self.res_connection(x), batch_index); x = F.gelu(self.norm1(self.conv1(x, edge_index))); x = F.gelu(self.norm2(self.conv2(x, edge_index))); x = self.norm3(self.conv3(x, edge_index)); x_pooled = global_mean_pool(x, batch_index); return F.gelu(x_pooled + initial_x_pooled)

class TimeIntervalEncoder(nn.Module):
    def __init__(self, out_dim): super().__init__(); self.net = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, out_dim))
    def forward(self, normalized_intervals): return self.net(normalized_intervals)

class IGATPredictor(nn.Module):
    def __init__(self, num_entities, num_relations, feature_dim, hidden_dim, embedding_dim, num_heads=8, num_layers=8, dropout=0.1):
        super().__init__()
        self.node_features = nn.Embedding(num_entities, feature_dim)
        self.relation_embeds = nn.Embedding(num_relations, embedding_dim)
        
        gnn_in_channels = feature_dim + embedding_dim + 1
        self.gnn_encoder = SubgraphEncoder(gnn_in_channels, hidden_dim, embedding_dim, num_heads, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.time_encoder = TimeIntervalEncoder(embedding_dim)

        scoring_input_dim = embedding_dim * 4
        self.scoring_head = nn.Sequential(nn.Linear(scoring_input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        
        regression_input_dim = embedding_dim * 3
        self.regression_head = nn.Sequential(nn.Linear(regression_input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))

    def _embed_subgraph_features(self, subgraph_batch):
        node_ids, avg_rel_ids, time_features = subgraph_batch.x[:, 0].long(), subgraph_batch.x[:, 1].long(), subgraph_batch.x[:, 2].unsqueeze(1)
        node_embeds = self.node_features(node_ids); mask = avg_rel_ids == -1; safe_rel_ids = avg_rel_ids.clone(); safe_rel_ids[mask] = 0; rel_embeds = self.relation_embeds(safe_rel_ids); rel_embeds[mask] = 0.0
        return torch.cat([node_embeds, rel_embeds, time_features], dim=1)

    def forward(self, s_history_batch, o_history_batch, s_hist_lengths, o_hist_lengths, query_triplet, intervals_norm):
        s_x_embedded = self._embed_subgraph_features(s_history_batch)
        o_x_embedded = self._embed_subgraph_features(o_history_batch)
        s_hist_embeds = self.gnn_encoder(s_x_embedded, s_history_batch.edge_index, s_history_batch.batch)
        o_hist_embeds = self.gnn_encoder(o_x_embedded, o_history_batch.edge_index, o_history_batch.batch)
        s_hist_seq = nn.utils.rnn.pad_sequence(torch.split(s_hist_embeds, s_hist_lengths.tolist()), batch_first=True)
        o_hist_seq = nn.utils.rnn.pad_sequence(torch.split(o_hist_embeds, o_hist_lengths.tolist()), batch_first=True)
        s_padding_mask = (torch.arange(s_hist_seq.size(1), device=s_hist_seq.device)[None, :] >= s_hist_lengths[:, None])
        o_padding_mask = (torch.arange(o_hist_seq.size(1), device=o_hist_seq.device)[None, :] >= o_hist_lengths[:, None])
        s_encoded_seq = self.transformer_encoder(s_hist_seq, src_key_padding_mask=s_padding_mask)
        o_encoded_seq = self.transformer_encoder(o_hist_seq, src_key_padding_mask=o_padding_mask)
        s_encoded_seq.masked_fill_(s_padding_mask.unsqueeze(-1), 0.0); o_encoded_seq.masked_fill_(o_padding_mask.unsqueeze(-1), 0.0)
        s_context = torch.sum(s_encoded_seq, dim=1) / s_hist_lengths.view(-1, 1).clamp(min=1)
        o_context = torch.sum(o_encoded_seq, dim=1) / o_hist_lengths.view(-1, 1).clamp(min=1)
        r_q_emb = self.relation_embeds(query_triplet[:, 1])

        batch_size, num_candidates, _ = intervals_norm.shape
        s_context_exp = s_context.unsqueeze(1).expand(-1, num_candidates, -1)
        o_context_exp = o_context.unsqueeze(1).expand(-1, num_candidates, -1)
        r_q_emb_exp = r_q_emb.unsqueeze(1).expand(-1, num_candidates, -1)
        t_embeds = self.time_encoder(intervals_norm)
        scoring_input = torch.cat([s_context_exp, r_q_emb_exp, o_context_exp, t_embeds], dim=-1)
        scores = self.scoring_head(scoring_input.view(batch_size * num_candidates, -1)).view(batch_size, num_candidates)
        
        combined_repr = torch.cat([s_context, r_q_emb, o_context], dim=1)
        regression_output = self.regression_head(combined_repr)
        pred_center_logit, pred_log_span = regression_output[:, 0], regression_output[:, 1]
        pred_center = torch.sigmoid(pred_center_logit)
        pred_span = torch.exp(pred_log_span)
        predicted_interval_norm = torch.stack([pred_center - pred_span / 2, pred_center + pred_span / 2], dim=1)

        return scores, predicted_interval_norm