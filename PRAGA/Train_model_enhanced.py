import torch
import time
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from .model_enhanced import Encoder_overall_enhanced, Encoder_overall
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from .preprocess import adjacent_matrix_preprocessing
from .optimal_clustering_HLN import R5

class Parametered_Graph(nn.Module):
    def __init__(self, adj, device):
        super(Parametered_Graph, self).__init__()
        self.device = device
        adj_matrix = adj + torch.eye(adj.size(0)).to(device)
        
        self.adj_matrix = nn.Parameter(adj_matrix, requires_grad=True)
        
    def forward(self):
        return self.adj_matrix

class EnhancedTrain:
    """Enhanced training class with SNN-Transformer integration"""
    def __init__(self, 
        data,
        datatype,
        device,
        random_seed=2024,
        dim_input=3000,
        dim_output=64,
        num_heads=8,  # New parameter for attention heads
        use_enhanced=True,  # Toggle for enhanced model
        Arg=None
        ):

        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_heads = num_heads
        self.use_enhanced = use_enhanced
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to_dense().to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to_dense().to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to_dense().to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to_dense().to(self.device)

        self.paramed_adj_omics1 = Parametered_Graph(self.adj_feature_omics1, self.device).to(self.device)
        self.paramed_adj_omics2 = Parametered_Graph(self.adj_feature_omics2, self.device).to(self.device)

        self.adj_feature_omics1_copy = copy.deepcopy(self.adj_feature_omics1)
        self.adj_feature_omics2_copy = copy.deepcopy(self.adj_feature_omics2)

        self.EMA_coeffi = 0.9
        self.K = 5
        self.T = 4
        
        # Initialize appropriate model
        if self.use_enhanced:
            print("Using Enhanced PRAGA model with SNN-Transformer integration")
            self.model = Encoder_overall_enhanced(
                self.adata_omics1.X.shape[1], 
                self.dim_output,
                self.adata_omics2.X.shape[1], 
                self.dim_output,
                num_heads=self.num_heads
            ).to(self.device)
        else:
            print("Using Original PRAGA model")
            self.model = Encoder_overall(
                self.adata_omics1.X.shape[1], 
                self.dim_output,
                self.adata_omics2.X.shape[1], 
                self.dim_output
            ).to(self.device)

        # Processing features
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.X.A if hasattr(self.adata_omics1.X, 'A') else self.adata_omics1.X).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.X.A if hasattr(self.adata_omics2.X, 'A') else self.adata_omics2.X).to(self.device)

        print(f"Features omics1 shape: {self.features_omics1.shape}")
        print(f"Features omics2 shape: {self.features_omics2.shape}")
        print(f"Output dimension: {self.dim_output}")
        print(f"Number of attention heads: {self.num_heads}")

    def augment_param_adj(self, adj_feature, adj_feature_copy, K, T):
        """Augment parameterized adjacency matrix"""
        gamma1 = R5(adj_feature, K, T, device=self.device)
        adj_feature = gamma1 * adj_feature + (1-gamma1) * adj_feature_copy
        return adj_feature

    def train_model(self, epochs=1000, lr=1e-4, RNA_weight=1, ADT_weight=1, 
                   patience=20, min_delta=1e-4, save_path=None):
        """Enhanced training with improved learning schedule and quantization awareness"""
        
        # Enhanced optimizer with different learning rates for different components
        param_groups = [
            {'params': self.model.encoder_omics1.parameters(), 'lr': lr},
            {'params': self.model.encoder_omics2.parameters(), 'lr': lr},
            {'params': self.model.decoder_omics1.parameters(), 'lr': lr},
            {'params': self.model.decoder_omics2.parameters(), 'lr': lr},
            {'params': self.model.MLP.parameters(), 'lr': lr},
        ]
        
        # Add attention parameters if using enhanced model
        if self.use_enhanced:
            param_groups.extend([
                {'params': self.model.encoder_omics1.attention.parameters(), 'lr': lr * 0.5},  # Lower LR for attention
                {'params': self.model.encoder_omics2.attention.parameters(), 'lr': lr * 0.5},
            ])
        
        param_groups.extend([
            {'params': self.paramed_adj_omics1.parameters(), 'lr': lr * 10},  # Higher LR for adj matrices
            {'params': self.paramed_adj_omics2.parameters(), 'lr': lr * 10},
        ])
        
        optimizer = torch.optim.Adam(param_groups)
        
        # Enhanced learning rate scheduler
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
        
        # Enhanced loss tracking
        losses = {
            'total': [],
            'recon_omics1': [],
            'recon_omics2': [],
            'reg': []
        }
        
        best_loss = float('inf')
        patience_counter = 0
        
        print("Starting enhanced training...")
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # Augment adjacency matrices
            adj_feature_omics1 = self.augment_param_adj(
                self.paramed_adj_omics1(), 
                self.adj_feature_omics1_copy, 
                self.K, self.T
            )
            adj_feature_omics2 = self.augment_param_adj(
                self.paramed_adj_omics2(), 
                self.adj_feature_omics2_copy, 
                self.K, self.T
            )

            # Forward pass
            results = self.model(
                self.features_omics1, self.features_omics2,
                self.adj_spatial_omics1, adj_feature_omics1,
                self.adj_spatial_omics2, adj_feature_omics2
            )

            # Enhanced loss computation
            recon_loss_omics1 = F.mse_loss(results['emb_recon_omics1'], self.features_omics1)
            recon_loss_omics2 = F.mse_loss(results['emb_recon_omics2'], self.features_omics2)
            
            # Regularization loss
            reg_loss = 0.01 * (
                torch.norm(self.paramed_adj_omics1.adj_matrix - self.adj_feature_omics1_copy) +
                torch.norm(self.paramed_adj_omics2.adj_matrix - self.adj_feature_omics2_copy)
            )
            
            # If using enhanced model, add attention regularization
            if self.use_enhanced:
                # L2 regularization on attention weights
                attn_reg = 0.001 * (
                    sum(p.pow(2.0).sum() for p in self.model.encoder_omics1.attention.parameters()) +
                    sum(p.pow(2.0).sum() for p in self.model.encoder_omics2.attention.parameters())
                )
                reg_loss += attn_reg
            
            # Total loss
            total_loss = (RNA_weight * recon_loss_omics1 + 
                         ADT_weight * recon_loss_omics2 + 
                         reg_loss)

            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track losses
            losses['total'].append(total_loss.item())
            losses['recon_omics1'].append(recon_loss_omics1.item())
            losses['recon_omics2'].append(recon_loss_omics2.item())
            losses['reg'].append(reg_loss.item())
            
            # Early stopping check
            if total_loss.item() < best_loss - min_delta:
                best_loss = total_loss.item()
                patience_counter = 0
                # Save best model
                if save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': best_loss,
                        'enhanced': self.use_enhanced
                    }, save_path)
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 100 == 0 or patience_counter == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d} | Loss: {total_loss.item():.6f} | "
                      f"Recon1: {recon_loss_omics1.item():.6f} | "
                      f"Recon2: {recon_loss_omics2.item():.6f} | "
                      f"Reg: {reg_loss.item():.6f} | LR: {current_lr:.2e}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience: {patience})")
                break
        
        print(f"Training completed. Best loss: {best_loss:.6f}")
        return losses

    def get_embeddings(self):
        """Get enhanced embeddings from the trained model"""
        self.model.eval()
        with torch.no_grad():
            adj_feature_omics1 = self.augment_param_adj(
                self.paramed_adj_omics1(), 
                self.adj_feature_omics1_copy, 
                self.K, self.T
            )
            adj_feature_omics2 = self.augment_param_adj(
                self.paramed_adj_omics2(), 
                self.adj_feature_omics2_copy, 
                self.K, self.T
            )

            results = self.model(
                self.features_omics1, self.features_omics2,
                self.adj_spatial_omics1, adj_feature_omics1,
                self.adj_spatial_omics2, adj_feature_omics2
            )
            
            return {
                'combined': results['emb_latent_combined'].cpu().numpy(),
                'omics1': results['emb_latent_omics1'].cpu().numpy(),
                'omics2': results['emb_latent_omics2'].cpu().numpy(),
            }

    def cluster_embeddings(self, n_clusters, embeddings=None, init_k=None):
        """Enhanced clustering with automatic component selection"""
        if embeddings is None:
            embeddings = self.get_embeddings()['combined']
        
        if init_k is None:
            init_k = n_clusters
        
        # Try different numbers of components and select the best one
        best_score = -np.inf
        best_labels = None
        best_n_components = init_k
        
        for n_comp in range(max(1, init_k-2), min(len(embeddings), init_k+3)):
            try:
                gm = GaussianMixture(
                    n_components=n_comp, 
                    covariance_type='full',
                    random_state=self.random_seed,
                    max_iter=200,
                    tol=1e-4
                )
                labels = gm.fit_predict(embeddings)
                score = gm.score(embeddings)
                
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_n_components = n_comp
                    
            except Exception as e:
                continue
        
        print(f"Best clustering: {best_n_components} components (score: {best_score:.4f})")
        return best_labels

# Backward compatibility: original Train class
class Train(EnhancedTrain):
    """Original Train class for backward compatibility"""
    def __init__(self, *args, **kwargs):
        kwargs['use_enhanced'] = False
        super().__init__(*args, **kwargs)