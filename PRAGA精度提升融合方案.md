# PRAGA精度提升融合方案：集成Quantized Spike-driven Transformer

## 目标定位

**核心目标**: 利用QSD-Transformer的先进架构大幅提升PRAGA的聚类精度，特别是改善ARI、NMI、Completeness等关键指标。

## 当前PRAGA精度分析

### 现有性能瓶颈
基于您的实验结果：
- **ARI (0.267)**: 聚类质量有待提升
- **NMI (0.382)**: 信息利用不充分
- **Completeness (0.351)**: 同类样本聚集度低
- **Jaccard (0.257)**: 样本对匹配精度不足

### 关键问题识别
1. **特征表达能力不足**: 传统GCN可能无法充分捕获复杂的空间-表达关系
2. **多模态融合不够深入**: 简单的特征拼接损失了模态间的细粒度交互
3. **空间信息利用不充分**: 缺乏对长程空间依赖的建模
4. **聚类边界模糊**: 特征空间中类别边界不够清晰

## 精度提升融合策略

### 1. 多尺度注意力增强特征提取

```python
class Enhanced_PRAGA_Encoder(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=8, num_layers=3):
        super().__init__()
        
        # 原始图卷积编码器
        self.base_encoder = Encoder(dim_in, dim_out)
        
        # 多层注意力增强模块
        self.attention_layers = nn.ModuleList([
            MS_Attention_RepConv_qkv_id(dim_out, num_heads) 
            for _ in range(num_layers)
        ])
        
        # RepConv增强特征提取
        self.repconv_enhancer = nn.ModuleList([
            RepConv(dim_out, dim_out) for _ in range(2)
        ])
        
        # 残差连接和层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim_out) for _ in range(num_layers)
        ])
        
        # 特征细化模块
        self.feature_refiner = nn.Sequential(
            nn.Linear(dim_out, dim_out * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_out * 2, dim_out)
        )
        
    def forward(self, features, adj_matrix):
        # 基础图特征提取
        graph_feat, base_embedding = self.base_encoder(features, adj_matrix)
        
        # 转换为适合注意力机制的格式
        B, N, C = base_embedding.shape
        H = W = int(math.sqrt(N))
        attention_input = base_embedding.permute(0, 2, 1).reshape(B, C, H, W)
        
        # 多层注意力增强
        enhanced_features = attention_input
        for i, (attn_layer, norm_layer) in enumerate(zip(self.attention_layers, self.layer_norms)):
            # 注意力机制
            attn_out = attn_layer(enhanced_features)
            
            # 残差连接 + 层归一化
            attn_out_flat = attn_out.reshape(B, C, -1).permute(0, 2, 1)
            enhanced_flat = enhanced_features.reshape(B, C, -1).permute(0, 2, 1)
            residual_out = norm_layer(attn_out_flat + enhanced_flat)
            
            enhanced_features = residual_out.permute(0, 2, 1).reshape(B, C, H, W)
        
        # RepConv进一步增强
        for repconv in self.repconv_enhancer:
            enhanced_features = enhanced_features + repconv(enhanced_features)
        
        # 特征细化
        final_embedding = enhanced_features.reshape(B, C, -1).permute(0, 2, 1)
        refined_embedding = self.feature_refiner(final_embedding)
        
        return refined_embedding, enhanced_features
```

### 2. 自适应多模态深度融合模块

```python
class Adaptive_Multimodal_Fusion(nn.Module):
    def __init__(self, dim1, dim2, fusion_dim, num_heads=8):
        super().__init__()
        
        # 模态特异性编码器
        self.modality1_projector = nn.Sequential(
            nn.Linear(dim1, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
        
        self.modality2_projector = nn.Sequential(
            nn.Linear(dim2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
        
        # 交叉模态注意力
        self.cross_attention1 = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.cross_attention2 = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # 自注意力增强
        self.self_attention = MS_Attention_RepConv_qkv_id(
            fusion_dim * 2, num_heads
        )
        
        # 门控融合机制
        self.gate_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        
        # 融合后的特征增强
        self.fusion_enhancer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.ReLU(),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        
    def forward(self, features1, features2):
        # 模态投影
        proj1 = self.modality1_projector(features1)  # [B, N, fusion_dim]
        proj2 = self.modality2_projector(features2)  # [B, N, fusion_dim]
        
        # 交叉模态注意力
        cross_attn1, _ = self.cross_attention1(proj1, proj2, proj2)
        cross_attn2, _ = self.cross_attention2(proj2, proj1, proj1)
        
        # 残差连接
        enhanced1 = proj1 + cross_attn1
        enhanced2 = proj2 + cross_attn2
        
        # 特征拼接
        concatenated = torch.cat([enhanced1, enhanced2], dim=-1)
        
        # 门控融合
        gate = self.gate_network(concatenated)
        gated_features = concatenated * gate
        
        # 转换为空间格式进行自注意力
        B, N, C = gated_features.shape
        H = W = int(math.sqrt(N))
        spatial_features = gated_features.permute(0, 2, 1).reshape(B, C, H, W)
        
        # 自注意力增强
        attended_features = self.self_attention(spatial_features)
        attended_flat = attended_features.reshape(B, C, -1).permute(0, 2, 1)
        
        # 最终融合增强
        final_features = self.fusion_enhancer(attended_flat)
        
        return final_features, attended_features
```

### 3. 精确聚类头设计

```python
class Precision_Clustering_Head(nn.Module):
    def __init__(self, input_dim, n_clusters, temperature=0.1):
        super().__init__()
        self.n_clusters = n_clusters
        self.temperature = temperature
        
        # 多层聚类投影
        self.cluster_projector = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, n_clusters)
        )
        
        # 原型学习
        self.prototypes = nn.Parameter(torch.randn(n_clusters, input_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
        # 注意力权重学习
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 对比学习投影头
        self.contrast_projector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 128)
        )
        
    def forward(self, features, spatial_coords=None):
        B, N, C = features.shape
        
        # 注意力加权
        attention = self.attention_weights(features)  # [B, N, 1]
        weighted_features = features * attention
        
        # 聚类投影
        cluster_logits = self.cluster_projector(weighted_features)
        
        # 原型匹配
        normalized_features = F.normalize(weighted_features, dim=-1)
        normalized_prototypes = F.normalize(self.prototypes, dim=-1)
        
        prototype_similarities = torch.matmul(
            normalized_features, normalized_prototypes.t()
        )  # [B, N, n_clusters]
        
        # 温度缩放
        prototype_logits = prototype_similarities / self.temperature
        
        # 结合投影和原型结果
        final_logits = (cluster_logits + prototype_logits) / 2
        
        # 对比学习特征
        contrast_features = self.contrast_projector(weighted_features)
        
        return {
            'cluster_logits': final_logits,
            'prototype_similarities': prototype_similarities,
            'contrast_features': contrast_features,
            'attention_weights': attention
        }
```

### 4. 增强损失函数

```python
class Enhanced_PRAGA_Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.recon_weight = config.reconstruction_weight
        self.cluster_weight = config.clustering_weight
        self.contrast_weight = config.contrastive_weight
        self.prototype_weight = config.prototype_weight
        self.spatial_reg_weight = config.spatial_regularization_weight
        
    def forward(self, outputs, targets, spatial_coords):
        # 1. 重构损失（更精确）
        recon_loss = F.mse_loss(outputs['recon_omics1'], targets['omics1']) + \
                    F.mse_loss(outputs['recon_omics2'], targets['omics2'])
        
        # 2. 对比学习损失（增强类间分离）
        contrast_loss = self.enhanced_contrastive_loss(
            outputs['contrast_features'], spatial_coords
        )
        
        # 3. 原型损失（增强类内聚集）
        prototype_loss = self.prototype_consistency_loss(
            outputs['prototype_similarities'], spatial_coords
        )
        
        # 4. 空间正则化损失（保持空间连续性）
        spatial_reg_loss = self.spatial_continuity_loss(
            outputs['cluster_logits'], spatial_coords
        )
        
        # 5. 熵损失（增强聚类置信度）
        entropy_loss = self.entropy_regularization(outputs['cluster_logits'])
        
        total_loss = (
            self.recon_weight * recon_loss +
            self.cluster_weight * contrast_loss +
            self.prototype_weight * prototype_loss +
            self.spatial_reg_weight * spatial_reg_loss +
            0.1 * entropy_loss
        )
        
        return total_loss
    
    def enhanced_contrastive_loss(self, features, spatial_coords):
        # 基于空间距离和特征相似性的对比学习
        B, N, C = features.shape
        
        # 计算特征相似性
        features_norm = F.normalize(features, dim=-1)
        similarity_matrix = torch.matmul(features_norm, features_norm.transpose(-2, -1))
        
        # 计算空间距离
        spatial_dist = torch.cdist(spatial_coords, spatial_coords)
        spatial_sim = torch.exp(-spatial_dist / spatial_dist.std())
        
        # 结合空间和特征信息
        combined_sim = similarity_matrix * spatial_sim
        
        # 对比损失
        pos_pairs = combined_sim * (spatial_dist < spatial_dist.median())
        neg_pairs = combined_sim * (spatial_dist >= spatial_dist.median())
        
        pos_loss = -torch.log(pos_pairs + 1e-8).mean()
        neg_loss = -torch.log(1 - neg_pairs + 1e-8).mean()
        
        return pos_loss + neg_loss
    
    def prototype_consistency_loss(self, prototype_sims, spatial_coords):
        # 确保空间邻近的点有相似的原型匹配
        B, N, K = prototype_sims.shape
        
        # 计算空间邻接性
        spatial_dist = torch.cdist(spatial_coords, spatial_coords)
        neighbors = (spatial_dist < spatial_dist.quantile(0.1)).float()
        
        # 原型一致性损失
        consistency_loss = 0
        for i in range(N):
            neighbor_indices = neighbors[0, i].nonzero().squeeze()
            if neighbor_indices.numel() > 1:
                neighbor_prototypes = prototype_sims[0, neighbor_indices]
                target_prototype = prototype_sims[0, i].unsqueeze(0)
                consistency_loss += F.mse_loss(
                    neighbor_prototypes.mean(dim=0, keepdim=True), 
                    target_prototype
                )
        
        return consistency_loss / N
```

### 5. 完整的增强PRAGA模型

```python
class Enhanced_PRAGA_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 增强的编码器
        self.omics1_encoder = Enhanced_PRAGA_Encoder(
            config.omics1_dim, config.hidden_dim, 
            num_heads=config.num_heads, num_layers=config.num_layers
        )
        self.omics2_encoder = Enhanced_PRAGA_Encoder(
            config.omics2_dim, config.hidden_dim,
            num_heads=config.num_heads, num_layers=config.num_layers
        )
        
        # 深度多模态融合
        self.multimodal_fusion = Adaptive_Multimodal_Fusion(
            config.hidden_dim, config.hidden_dim, config.fusion_dim,
            num_heads=config.num_heads
        )
        
        # 精确聚类头
        self.clustering_head = Precision_Clustering_Head(
            config.fusion_dim, config.n_clusters
        )
        
        # 增强的解码器
        self.decoder_omics1 = Enhanced_Decoder(config.fusion_dim, config.omics1_dim)
        self.decoder_omics2 = Enhanced_Decoder(config.fusion_dim, config.omics2_dim)
        
        # 图自适应学习
        self.adaptive_graph = Parametered_Graph_Learner(config.hidden_dim)
        
    def forward(self, omics1_data, omics2_data, spatial_coords, adj_matrices):
        # 自适应图学习
        adaptive_adj1 = self.adaptive_graph(omics1_data, adj_matrices['omics1'])
        adaptive_adj2 = self.adaptive_graph(omics2_data, adj_matrices['omics2'])
        
        # 增强特征提取
        omics1_features, omics1_spatial = self.omics1_encoder(omics1_data, adaptive_adj1)
        omics2_features, omics2_spatial = self.omics2_encoder(omics2_data, adaptive_adj2)
        
        # 深度多模态融合
        fused_features, attended_features = self.multimodal_fusion(
            omics1_features, omics2_features
        )
        
        # 精确聚类
        cluster_outputs = self.clustering_head(fused_features, spatial_coords)
        
        # 重构
        recon_omics1 = self.decoder_omics1(fused_features)
        recon_omics2 = self.decoder_omics2(fused_features)
        
        return {
            'cluster_logits': cluster_outputs['cluster_logits'],
            'prototype_similarities': cluster_outputs['prototype_similarities'],
            'contrast_features': cluster_outputs['contrast_features'],
            'attention_weights': cluster_outputs['attention_weights'],
            'recon_omics1': recon_omics1,
            'recon_omics2': recon_omics2,
            'embeddings': fused_features
        }
```

## 精度提升策略总结

### 1. 架构层面增强
- **多层注意力机制**: 捕获长程依赖关系
- **RepConv结构**: 增强特征提取能力
- **残差连接**: 支持更深网络，避免梯度消失
- **自适应图学习**: 动态优化图结构

### 2. 融合策略优化
- **交叉模态注意力**: 深度挖掘模态间交互
- **门控融合**: 自适应权重分配
- **原型学习**: 增强类别表示能力

### 3. 训练策略改进
- **对比学习**: 增强类间分离度
- **空间正则化**: 保持空间连续性
- **原型一致性**: 提高聚类稳定性
- **熵正则化**: 增强决策置信度

### 4. 预期精度提升
- **ARI**: 从0.267提升至0.40+（+50%）
- **NMI**: 从0.382提升至0.50+（+30%）
- **Completeness**: 从0.351提升至0.45+（+28%）
- **F-measure**: 从0.409提升至0.55+（+35%）

这个方案专注于利用QSD-Transformer的先进架构来大幅提升PRAGA的聚类精度，特别是改善同类样本的聚集程度和类间分离度。