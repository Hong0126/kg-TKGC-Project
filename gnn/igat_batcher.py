# igat_batcher.py
import torch
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph

class InductiveTKGStitcher:
    def __init__(self, full_graph_df, entity_map, relation_map, time_range):
        self.k_hops, self.max_history = 1, 10
        self.t_min, self.t_max = time_range
        self.t_span = float(self.t_max - self.t_min) if self.t_max > self.t_min else 1.0
        
        self.facts_np = full_graph_df[['s_id', 'r_id', 'o_id', 'start_time']].values.astype(np.int64)
        self.edge_index = torch.from_numpy(self.facts_np[:, [0, 2]].T).long()
        self.edge_relations = torch.from_numpy(self.facts_np[:, 1]).long()
        self.num_nodes = len(entity_map)
        
        self.s_history_lookup, self.o_history_lookup = defaultdict(list), defaultdict(list)
        for i, (s, _, o, _) in enumerate(self.facts_np):
            self.s_history_lookup[s].append(i)
            self.o_history_lookup[o].append(i)

    def _normalize_time(self, t):
        return (t - self.t_min) / self.t_span

    def _create_subgraph_for_event(self, center_node, timestamp):
        center_node_tensor = torch.tensor([center_node], dtype=torch.long)
        subset, sub_edge_index, _, edge_mask = k_hop_subgraph(
            center_node_tensor, self.k_hops, self.edge_index, True, self.num_nodes
        )
        
        node_ids = subset.view(-1, 1).float()
        sub_edge_rel_ids = self.edge_relations[edge_mask]
        avg_rel_id = sub_edge_rel_ids.float().mean().item() if sub_edge_rel_ids.shape[0] > 0 else -1.0
        rel_ids_feat = torch.full((len(subset), 1), avg_rel_id)
        time_feature = self._normalize_time(torch.tensor(timestamp, dtype=torch.float)).view(1, 1).expand(len(subset), -1)
        
        x = torch.cat([node_ids, rel_ids_feat, time_feature], dim=1)
        return Data(x=x, edge_index=sub_edge_index)

    def _get_history_subgraphs(self, entity_id, lookup_dict, query_time):
        indices = lookup_dict.get(entity_id, [])
        if not indices: return []
        
        candidate_history = self.facts_np[indices]
        historical_facts = candidate_history[candidate_history[:, 3] < query_time]
        if historical_facts.shape[0] == 0: return []
        
        historical_facts = historical_facts[np.argsort(historical_facts[:, 3])]
        return [self._create_subgraph_for_event(s if s != entity_id else o, t) for s, r, o, t in historical_facts[-self.max_history:]]

class TKGCollator:
    """为每个真实样本生成负样本，用于基于排名的训练。"""
    def __init__(self, stitcher: InductiveTKGStitcher, num_neg_samples: int = 32):
        self.stitcher = stitcher
        self.num_neg_samples = num_neg_samples

    def __call__(self, batch_items):
        s_all_subgraphs, o_all_subgraphs = [], []
        queries_b, s_hist_lengths, o_hist_lengths = [], [], []
        all_intervals_b, labels_b = [], []

        for item in batch_items:
            s, r, o, t_start = item["s"], item["r"], item["o"], item["start_time"]
            s_subgraphs = self.stitcher._get_history_subgraphs(s, self.stitcher.s_history_lookup, t_start)
            o_subgraphs = self.stitcher._get_history_subgraphs(o, self.stitcher.o_history_lookup, t_start)
            if not s_subgraphs or not o_subgraphs: continue
            
            s_all_subgraphs.extend(s_subgraphs)
            o_all_subgraphs.extend(o_subgraphs)
            s_hist_lengths.append(len(s_subgraphs))
            o_hist_lengths.append(len(o_subgraphs))
            queries_b.append([s, r, o])
            
            # For training, create positive and negative samples
            if self.num_neg_samples > 0:
                true_interval = torch.tensor([item["start_time"], item["end_time"]], dtype=torch.float)
                current_intervals = [true_interval]
                current_labels = [1.0]
                
                for _ in range(self.num_neg_samples):
                    neg_start = np.random.randint(self.stitcher.t_min, self.stitcher.t_max + 1)
                    # Ensure duration > 0, and not too long
                    max_duration = self.stitcher.t_max - neg_start + 1
                    duration = np.random.randint(0, max(1, max_duration // 4)) # Sample shorter durations
                    neg_end = neg_start + duration
                    neg_interval = torch.tensor([neg_start, neg_end], dtype=torch.float)
                    current_intervals.append(neg_interval)
                    current_labels.append(0.0)
                
                all_intervals_b.append(torch.stack(current_intervals))
                labels_b.append(torch.tensor(current_labels))
            else: # For evaluation, only need the true interval
                all_intervals_b.append(torch.tensor([[item["start_time"], item["end_time"]]], dtype=torch.float))


        if not queries_b: return None
        
        batch_dict = {
            "s_hist": Batch.from_data_list(s_all_subgraphs),
            "o_hist": Batch.from_data_list(o_all_subgraphs),
            "queries": torch.tensor(queries_b, dtype=torch.long),
            "s_lens": torch.tensor(s_hist_lengths, dtype=torch.long),
            "o_lens": torch.tensor(o_hist_lengths, dtype=torch.long),
            "intervals": torch.stack(all_intervals_b),
        }
        if self.num_neg_samples > 0:
            batch_dict["labels"] = torch.stack(labels_b)

        return batch_dict