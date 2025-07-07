import torch
import torch.nn as nn
import math


# a pointer network layer for policy output
class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k, mask=None):

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)

        k_flat = k.reshape(-1, n_dim)
        q_flat = q.reshape(-1, n_dim)

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(k_flat, self.w_key).view(shape_k)

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            U = U.masked_fill(mask == 1, -1e8)
        attention = torch.log_softmax(U, dim=-1)  # n_batch*n_query*n_key

        return attention

# standard multi head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)
        n_value = v.size(1)

        k_flat = k.contiguous().view(-1, n_dim)
        v_flat = v.contiguous().view(-1, n_dim)
        q_flat = q.contiguous().view(-1, n_dim)
        shape_v = (self.n_heads, n_batch, n_value, -1)
        shape_k = (self.n_heads, n_batch, n_key, -1)
        shape_q = (self.n_heads, n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(k_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(v_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if attn_mask is not None:
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(U)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1)
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(U)  # copy for n_heads times

        if attn_mask is not None and key_padding_mask is not None:
            mask = (attn_mask + key_padding_mask)
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            U = U.masked_fill(mask > 0, -1e8)

        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        # out = heads.permute(1, 2, 0, 3).reshape(n_batch, n_query, n_dim)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(-1, n_query, self.embedding_dim)

        return out, attention  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        h0 = src
        h = self.normalization1(src)
        h, _ = self.multiHeadAttention(q=h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h, w = self.multiHeadAttention(q=tgt, k=memory, v=memory, key_padding_mask=key_padding_mask,
                                       attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2, w


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            src = layer(src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            tgt, w = layer(tgt, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return tgt, w


class PolicyNet(nn.Module):
    def __init__(self, node_dim, frontiers_clusters_dim, embedding_dim):
        super(PolicyNet, self).__init__()

        # Graph Encoder
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=4, n_layer=6)

        # Teamwork decoder
        self.past_trajectory_embedding = nn.Linear(embedding_dim + 1, embedding_dim)
        self.trajectories_score = nn.Linear(embedding_dim, 1)
        self.teamwork_decoder = Decoder(embedding_dim=embedding_dim+1, n_head=4, n_layer=1)

        # Frontiers Encoder
        self.frontiers_embedding = nn.Linear(frontiers_clusters_dim, embedding_dim)
        self.frontier_encoder = Encoder(embedding_dim=embedding_dim, n_head=3, n_layer=3)
        self.frontiers_decoder = Decoder(embedding_dim=embedding_dim, n_head=3, n_layer=1)

        self.teamwork_fusion = nn.Linear(embedding_dim * 2 + 1, embedding_dim)

        # Local frontiers distribution encoder
        # self.frontiers_embedding = nn.Linear(num_angles_bin, embedding_dim)
        # self.frontiers_embedding =  nn.Conv1d(num_angles_bin, embedding_dim, kernel_size=3, padding=1)
        # self.node_frontiers_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        # decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.current_embedding = nn.Linear(embedding_dim * 3, embedding_dim)

        # heading layer
        # self.best_headings_embedding = nn.Linear(num_angles_bin, embedding_dim)
        # self.visited_headings_embedding = nn.Linear(num_angles_bin, embedding_dim)
        # self.neighboring_node_embedding = nn.Linear(embedding_dim * 3, embedding_dim)

        # pointer
        self.pointer = SingleHeadAttention(embedding_dim)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(src=node_feature,
                                            key_padding_mask=node_padding_mask,
                                            attn_mask=edge_mask)
        return enhanced_node_feature
    def decode_teamwork(self, enhanced_node_feature, all_robot_past_locations, current_robot_past_locations):
        B, N, T = all_robot_past_locations.shape
        _, num_nodes, embedding_dim = enhanced_node_feature.shape

        # Prepare common embedding expansion [B, 1, 1, num_nodes, embedding_dim]
        # We'll re-use this without expensive .expand(...)
        enhanced_node_features_exp = enhanced_node_feature.unsqueeze(1).unsqueeze(1)

        def extract_trajectory_features(indices, target_N):
            """Gathers node embeddings based on indices: [B, N, T] -> [B, N, T, embedding_dim]"""
            safe_idx = indices.clamp(min=0)  # clamp -1 to 0
            expanded_idx = safe_idx.unsqueeze(-1).expand(-1, -1, -1, embedding_dim)  # [B, N, T, D]
            gather_idx = expanded_idx.unsqueeze(3)  # [B, N, T, 1, D]
            features_exp = enhanced_node_features_exp.expand(B, target_N, T, num_nodes, embedding_dim)
            gathered = torch.gather(features_exp, dim=3, index=gather_idx).squeeze(3)
            mask = (indices == -1).unsqueeze(-1)

            # --- Compute recency values ---
            # Count the number of valid setpoints per trajectory; valid when index != -1.
            valid_count = (indices != -1).sum(dim=-1, keepdim=True).float()  # [B, N, 1]
            # To avoid division by zero, clamp the count to at least 0.
            valid_count = valid_count.clamp(min=0)

            # Create a recency vector for all timesteps.
            # torch.arange(T) creates [0, 1, ..., T-1]; add 1 so that positions become 1, 2, ..., T.
            # Then normalize by the number of valid setpoints.
            recency = (torch.arange(T, device=indices.device).float().view(1, 1, T) + 1) / valid_count
            # For invalid indices (where indices == -1), we set the recency value to 0.
            recency = torch.where(indices == -1, torch.zeros_like(recency), recency)
            # Add a new last dimension so that recency has shape [B, N, T, 1].
            recency = recency.unsqueeze(-1)

            # Concatenate the recency value to the gathered embedding.
            gathered = torch.cat([gathered, recency], dim=-1) 

            return torch.where(mask, torch.zeros_like(gathered), gathered), mask.squeeze(-1)

        # All robots
        trajectory_features, mask = extract_trajectory_features(all_robot_past_locations, N)

        # Current robot
        current_traj_feats, current_mask = extract_trajectory_features(current_robot_past_locations, 1)

        # Attention for all robots
        h = torch.tanh(self.past_trajectory_embedding(trajectory_features))  # [B, N, T, H]
        scores = self.trajectories_score(h).squeeze(-1)                      # [B, N, T]
        scores = scores.masked_fill(mask == 1, float('-inf'))               # invalid = -inf
        alpha = torch.softmax(scores, dim=-1)                                # [B, N, T]
        pooled_trajectory_features = torch.sum(trajectory_features * alpha.unsqueeze(-1), dim=2)  # [B, N, D]

        # Attention for current robot
        h_curr = torch.tanh(self.past_trajectory_embedding(current_traj_feats))  # [B, 1, T, H]
        scores_curr = self.trajectories_score(h_curr).squeeze(-1)                # [B, 1, T]
        scores_curr = scores_curr.masked_fill(current_mask == 1, float('-inf'))
        alpha_curr = torch.softmax(scores_curr, dim=-1)                          # [B, 1, T]
        current_pooled_trajectory_features = torch.sum(current_traj_feats * alpha_curr.unsqueeze(-1), dim=2)  # [B, 1, D]

        # Teamwork decoding
        enhanced_trajectory_features, _ = self.teamwork_decoder(
            current_pooled_trajectory_features, pooled_trajectory_features
        )
        return enhanced_trajectory_features

    def encode_frontiers_clusters(self, frontiers_clusters_inputs, frontiers_padding_mask):
        frontiers_feature = self.frontiers_embedding(frontiers_clusters_inputs)
        enhanced_frontiers_feature = self.frontier_encoder(src=frontiers_feature,
                                                           key_padding_mask=frontiers_padding_mask)
        return enhanced_frontiers_feature
    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1,
                                            current_index.repeat(1, 1, embedding_dim))
        
        enhanced_current_node_feature, _ = self.decoder(current_node_feature,
                                                        enhanced_node_feature,
                                                        node_padding_mask)
        return current_node_feature, enhanced_current_node_feature
    
    def decode_frontiers_clusters(self, current_robot_embeddings, enhanced_trajectory_features, enhanced_frontiers_feature, frontiers_padding_mask):
        teamwork_embeddings = self.teamwork_fusion(torch.cat((current_robot_embeddings, enhanced_trajectory_features), dim=-1))
        enhanced_frontiers_feature, _ = self.frontiers_decoder(teamwork_embeddings,
                                                                enhanced_frontiers_feature,
                                                                frontiers_padding_mask)
        return enhanced_frontiers_feature
    def output_policy(self, current_node_feature, enhanced_current_node_feature,
                      enhanced_node_feature, current_edge, edge_padding_mask, decoded_robot_frontiers_feature):
        
        ### headings_visited is [1,360,6], neighbor_best_headings is [1,25,3,36]
        embedding_dim = enhanced_node_feature.size()[2]
        # batch_size = enhanced_node_feature.size()[0]
        current_state_feature = self.current_embedding(torch.cat((enhanced_current_node_feature,
                                                                  current_node_feature,
                                                                  decoded_robot_frontiers_feature), dim=-1))

        neighboring_feature = torch.gather(enhanced_node_feature, 1,
                                           current_edge.repeat(1, 1, embedding_dim))        # [1,25,128]

        # enhanced_neighbor_best_headings = self.best_headings_embedding(neighbor_best_headings)  # [1,25, 3,128]
        # all_headings_visited = self.visited_headings_embedding(headings_visited)               # [1,360,128]
        # all_neighbor_headings_visited = torch.gather(all_headings_visited, 1,
        #                                    current_edge.repeat(1, 1, embedding_dim))            # [1,25,128]

        # neighboring_nodes_feature = neighboring_feature.unsqueeze(2).repeat(1, 1, num_best_headings, 1)                 # [1,25,3,128]
        # neighbor_headings_visited = all_neighbor_headings_visited.unsqueeze(2).repeat(1, 1, num_best_headings, 1)       # [1,25,3,128]

        # enhanced_neighbor_features = self.neighboring_node_embedding(torch.cat((neighboring_nodes_feature, neighbor_headings_visited,
        #                                                                         enhanced_neighbor_best_headings), dim=-1)).reshape(batch_size, -1, embedding_dim)       # [1,75,128]
     
        # current_mask = edge_padding_mask.unsqueeze(-1).repeat(1, 1, 1, num_best_headings).reshape(batch_size, 1, -1)
        logp = self.pointer(current_state_feature, neighboring_feature, edge_padding_mask)
        logp = logp.squeeze(1)
        return logp

    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask, frontiers_clusters_inputs, frontiers_padding_mask,
                all_robot_past_locations, current_robot_past_locations):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        enhanced_trajectory_features = self.decode_teamwork(enhanced_node_feature, all_robot_past_locations, current_robot_past_locations)
        enhanced_frontiers_features = self.encode_frontiers_clusters(frontiers_clusters_inputs, frontiers_padding_mask)
        # border_feature = self.border_embedding(border_node_coords)
        current_node_feature, enhanced_current_node_feature = self.decode_state(enhanced_node_feature, current_index, node_padding_mask)
        decoded_robot_frontiers_feature = self.decode_frontiers_clusters(current_node_feature, enhanced_trajectory_features, enhanced_frontiers_features, frontiers_padding_mask)
        logp = self.output_policy(current_node_feature, enhanced_current_node_feature,
                                  enhanced_node_feature, current_edge, edge_padding_mask, decoded_robot_frontiers_feature)
        return logp

class QNet(nn.Module):
    def __init__(self, node_dim, frontiers_clusters_dim, embedding_dim, train_algo):
        super(QNet, self).__init__()

        # Graph encoder
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=4, n_layer=6)

        # Teamwork decoder
        self.past_trajectory_embedding = nn.Linear(embedding_dim + 1, embedding_dim)
        self.trajectories_score = nn.Linear(embedding_dim, 1)
        self.teamwork_decoder = Decoder(embedding_dim=embedding_dim + 1, n_head=4, n_layer=1)
        
        # Frontiers embeddings
        self.frontiers_embedding = nn.Linear(frontiers_clusters_dim, embedding_dim)
        self.frontier_encoder = Encoder(embedding_dim=embedding_dim, n_head=3, n_layer=3)
        self.frontiers_decoder = Decoder(embedding_dim=embedding_dim, n_head=3, n_layer=1)
        # self.border_embedding = nn.Linear(num_border_points * 2, embedding_dim)
        # Local frontiers distribution encoder
        # self.frontiers_embedding = nn.Linear(num_angles_bin, embedding_dim)
        # self.frontiers_embedding = nn.Conv1d(num_angles_bin, embedding_dim, kernel_size=3, padding=1)
        # self.node_frontiers_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        self.teamwork_fusion = nn.Linear(embedding_dim * 2 + 1, embedding_dim)

        # Decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        # Heading layer
        # self.best_headings_embedding = nn.Linear(num_angles_bin, embedding_dim)
        # self.visited_headings_embedding = nn.Linear(num_angles_bin, embedding_dim)
        # self.neighboring_node_embedding = nn.Linear(embedding_dim * 3, embedding_dim)

        # Ground truth critic
        if train_algo in (2 ,3):
            self.initial_embedding = nn.Linear(node_dim + 1, embedding_dim)
        else:
            self.initial_embedding = nn.Linear(node_dim, embedding_dim)

        # MAAC
        if train_algo in (1, 3):
            self.agent_decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
            self.all_agent_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

            self.q_values_layer = nn.Linear(embedding_dim * 5, 1)
        else:
            self.q_values_layer = nn.Linear(embedding_dim * 3, 1)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(src=node_feature,
                                            key_padding_mask=node_padding_mask,
                                            attn_mask=edge_mask)
        
        # frontier_distribution = frontier_distribution.permute(0, 2, 1)
        # frontiers_feature = self.frontiers_embedding(frontier_distribution)
        # frontiers_feature = frontiers_feature.permute(0, 2, 1)

        # enhanced_node_feature = self.node_frontiers_embedding(torch.cat((enhanced_node_feature, frontiers_feature), dim=-1))

        return enhanced_node_feature
    
    def decode_teamwork(self, enhanced_node_feature, all_robot_past_locations, current_robot_past_locations):
        B, N, T = all_robot_past_locations.shape
        _, num_nodes, embedding_dim = enhanced_node_feature.shape

        # Prepare common embedding expansion [B, 1, 1, num_nodes, embedding_dim]
        # We'll re-use this without expensive .expand(...)
        enhanced_node_features_exp = enhanced_node_feature.unsqueeze(1).unsqueeze(1)

        def extract_trajectory_features(indices, target_N):
            """Gathers node embeddings based on indices: [B, N, T] -> [B, N, T, embedding_dim]"""
            safe_idx = indices.clamp(min=0)  # clamp -1 to 0
            expanded_idx = safe_idx.unsqueeze(-1).expand(-1, -1, -1, embedding_dim)  # [B, N, T, D]
            gather_idx = expanded_idx.unsqueeze(3)  # [B, N, T, 1, D]
            features_exp = enhanced_node_features_exp.expand(B, target_N, T, num_nodes, embedding_dim)
            gathered = torch.gather(features_exp, dim=3, index=gather_idx).squeeze(3)
            mask = (indices == -1).unsqueeze(-1)

            # --- Compute recency values ---
            # Count the number of valid setpoints per trajectory; valid when index != -1.
            valid_count = (indices != -1).sum(dim=-1, keepdim=True).float()  # [B, N, 1]
            # To avoid division by zero, clamp the count to at least 0.
            valid_count = valid_count.clamp(min=0)

            # Create a recency vector for all timesteps.
            # torch.arange(T) creates [0, 1, ..., T-1]; add 1 so that positions become 1, 2, ..., T.
            # Then normalize by the number of valid setpoints.
            recency = (torch.arange(T, device=indices.device).float().view(1, 1, T) + 1) / valid_count
            # For invalid indices (where indices == -1), we set the recency value to 0.
            recency = torch.where(indices == -1, torch.zeros_like(recency), recency)
            # Add a new last dimension so that recency has shape [B, N, T, 1].
            recency = recency.unsqueeze(-1)

            # Concatenate the recency value to the gathered embedding.
            gathered = torch.cat([gathered, recency], dim=-1) 

            return torch.where(mask, torch.zeros_like(gathered), gathered), mask.squeeze(-1)

        # All robots
        trajectory_features, mask = extract_trajectory_features(all_robot_past_locations, N)

        # Current robot
        current_traj_feats, current_mask = extract_trajectory_features(current_robot_past_locations, 1)

        # Attention for all robots
        h = torch.tanh(self.past_trajectory_embedding(trajectory_features))  # [B, N, T, H]
        scores = self.trajectories_score(h).squeeze(-1)                      # [B, N, T]
        scores = scores.masked_fill(mask == 1, float('-inf'))               # invalid = -inf
        alpha = torch.softmax(scores, dim=-1)                                # [B, N, T]
        pooled_trajectory_features = torch.sum(trajectory_features * alpha.unsqueeze(-1), dim=2)  # [B, N, D]

        # Attention for current robot
        h_curr = torch.tanh(self.past_trajectory_embedding(current_traj_feats))  # [B, 1, T, H]
        scores_curr = self.trajectories_score(h_curr).squeeze(-1)                # [B, 1, T]
        scores_curr = scores_curr.masked_fill(current_mask == 1, float('-inf'))
        alpha_curr = torch.softmax(scores_curr, dim=-1)                          # [B, 1, T]
        current_pooled_trajectory_features = torch.sum(current_traj_feats * alpha_curr.unsqueeze(-1), dim=2)  # [B, 1, D]

        # Teamwork decoding
        enhanced_trajectory_features, _ = self.teamwork_decoder(
            current_pooled_trajectory_features, pooled_trajectory_features
        )
        return enhanced_trajectory_features
    
    def encode_frontiers_clusters(self, frontiers_clusters_inputs, frontiers_padding_mask):
        frontiers_feature = self.frontiers_embedding(frontiers_clusters_inputs)
        enhanced_frontiers_feature = self.frontier_encoder(src=frontiers_feature,
                                                           key_padding_mask=frontiers_padding_mask)
        return enhanced_frontiers_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1,
                                                  current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.decoder(current_node_feature,
                                                                    enhanced_node_feature,
                                                                    node_padding_mask)

        return current_node_feature, enhanced_current_node_feature
    
    def decode_frontiers_clusters(self, current_robot_embeddings, enhanced_trajectory_features, enhanced_frontiers_feature, frontiers_padding_mask):
        teamwork_embeddings = self.teamwork_fusion(torch.cat((current_robot_embeddings, enhanced_trajectory_features), dim=-1))
        enhanced_frontiers_feature, _ = self.frontiers_decoder(teamwork_embeddings,
                                                                enhanced_frontiers_feature,
                                                                frontiers_padding_mask)
        return enhanced_frontiers_feature

    def output_q(self, current_node_feature, enhanced_current_node_feature, enhanced_node_feature,
                 current_edge, edge_padding_mask, current_index, all_agent_indices, all_agent_next_indices, decoded_robot_frontiers_feature):
        ### headings_visited is [256,360,6], neighbor_best_headings is [256,25,3,36]
        embedding_dim = enhanced_node_feature.size()[2]
        # batch_size = enhanced_node_feature.size()[0]
        k_size = current_edge.size()[1]
        current_state_feature = current_node_feature
        enhanced_current_state_feature = enhanced_current_node_feature

        neighboring_feature = torch.gather(enhanced_node_feature, 1,
                                           current_edge.repeat(1, 1, embedding_dim))
        
        # enhanced_neighbor_best_headings = self.best_headings_embedding(neighbor_best_headings)
        # all_headings_visited = self.visited_headings_embedding(headings_visited)
        # all_neighbor_headings_visited = torch.gather(all_headings_visited, 1,
        #                                    current_edge.repeat(1, 1, embedding_dim))   

        # neighboring_nodes_feature = neighboring_feature.unsqueeze(2).repeat(1, 1, num_best_headings, 1)
        # neighbor_headings_visited = all_neighbor_headings_visited.unsqueeze(2).repeat(1, 1, num_best_headings, 1)

        # enhanced_neighbor_features = self.neighboring_node_embedding(torch.cat((neighboring_nodes_feature, neighbor_headings_visited,
        #                                                                         enhanced_neighbor_best_headings), dim=-1)).reshape(batch_size, -1, embedding_dim)
        
        if all_agent_indices != None:
            all_agent_node_feature = torch.gather(enhanced_node_feature, 1,
                                                all_agent_indices.repeat(1, 1, embedding_dim))
            all_agent_selected_neighboring_feature = torch.gather(enhanced_node_feature, 1,
                                                                all_agent_next_indices.repeat(1, 1, embedding_dim))

            all_agent_action_features = torch.cat((all_agent_node_feature, all_agent_selected_neighboring_feature), dim=-1)
            all_agent_action_features = self.all_agent_embedding(all_agent_action_features)

            agent_mask = all_agent_indices == current_index
            global_state_action_feature, _ = self.agent_decoder(current_state_feature, all_agent_action_features, agent_mask)
            action_features = torch.cat((current_state_feature.repeat(1, k_size, 1),
                                     enhanced_current_state_feature.repeat(1, k_size, 1),
                                     decoded_robot_frontiers_feature.repeat(1, k_size, 1),
                                     global_state_action_feature.repeat(1, k_size, 1),
                                     neighboring_feature), dim=-1)
            q_values = self.q_values_layer(action_features)
        else:
            action_features = torch.cat((current_state_feature.repeat(1, neighboring_feature.size()[1], 1), neighboring_feature, decoded_robot_frontiers_feature.repeat(1, neighboring_feature.size()[1], 1)), dim=-1)
            q_values = self.q_values_layer(action_features)
        return q_values

    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask, frontiers_clusters_inputs, frontiers_padding_mask,
                all_robot_past_locations, current_robot_past_locations, all_agent_indices=None, all_agent_next_indices=None):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        enhanced_trajectory_features = self.decode_teamwork(enhanced_node_feature, all_robot_past_locations, current_robot_past_locations)
        enhanced_frontiers_features = self.encode_frontiers_clusters(frontiers_clusters_inputs, frontiers_padding_mask)
        current_node_feature, enhanced_current_node_feature = self.decode_state(enhanced_node_feature, current_index, node_padding_mask)
        decoded_robot_frontiers_feature = self.decode_frontiers_clusters(current_node_feature, enhanced_trajectory_features, enhanced_frontiers_features, frontiers_padding_mask)
        q_values = self.output_q(current_node_feature, enhanced_current_node_feature,
                                 enhanced_node_feature, current_edge, edge_padding_mask, current_index, all_agent_indices, all_agent_next_indices,
                                 decoded_robot_frontiers_feature)
        return q_values