import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

# Import quantization components from SNN-Transformer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '113 Quantized Spike-driven Transformer（ICLR 2025）'))

from quan_w import Conv2dLSQ
from _quan_base_plus import grad_scale, round_pass

class Multispike(nn.Module):
    """Multi-spike activation function from SNN-Transformer"""
    def __init__(self, lens=4):
        super().__init__()
        self.lens = lens
        self.spike = multispike
        
    def forward(self, inputs):
        return self.spike.apply(4 * inputs, self.lens) / 4

class multispike(torch.autograd.Function):
    """Multi-spike activation function implementation"""
    @staticmethod
    def forward(ctx, input, lens):
        ctx.save_for_backward(input)
        ctx.lens = lens
        relu4 = lambda x: torch.clamp(x, 0, 4)
        return torch.floor(relu4(input) + 0.5)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None

class BNAndPadLayer(nn.Module):
    """Batch Normalization and Padding layer from SNN-Transformer"""
    def __init__(self, pad_pixels, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

class RepConv(nn.Module):
    """Rep-parametrizable convolution from SNN-Transformer"""
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()
        conv1x1 = Conv2dLSQ(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            Conv2dLSQ(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            Conv2dLSQ(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)

class SpikeAttention(nn.Module):
    """Multi-spike attention mechanism adapted for PRAGA features"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25
        
        # Multi-spike activation
        self.head_lif = Multispike()
        
        # Linear projections for q, k, v (adapted for 1D features)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)  
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # Multi-spike activations for q, k, v
        self.q_lif = Multispike()
        self.k_lif = Multispike()
        self.v_lif = Multispike()
        
        # Output projection
        self.proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: Input features [N, dim] where N is number of spots
        Returns:
            Enhanced features with self-attention
        """
        N, C = x.shape
        
        # Apply multi-spike activation
        x = self.head_lif(x)
        
        # Compute q, k, v
        q = self.q_proj(x)  # [N, dim]
        k = self.k_proj(x)  # [N, dim]
        v = self.v_proj(x)  # [N, dim]
        
        # Apply multi-spike activations
        q = self.q_lif(q)
        k = self.k_lif(k)
        v = self.v_lif(v)
        
        # Reshape for multi-head attention
        q = q.reshape(N, self.num_heads, C // self.num_heads).transpose(0, 1)  # [num_heads, N, head_dim]
        k = k.reshape(N, self.num_heads, C // self.num_heads).transpose(0, 1)  # [num_heads, N, head_dim]
        v = v.reshape(N, self.num_heads, C // self.num_heads).transpose(0, 1)  # [num_heads, N, head_dim]
        
        # Compute attention: k^T @ v then q @ (k^T @ v)
        kv = k.transpose(-2, -1) @ v  # [num_heads, head_dim, head_dim]
        attn_out = (q @ kv) * self.scale  # [num_heads, N, head_dim]
        
        # Reshape back
        attn_out = attn_out.transpose(0, 1).reshape(N, C)  # [N, dim]
        
        # Output projection
        out = self.proj(attn_out)
        
        return out

class QuantizedEncoder(Module):
    """Enhanced encoder with multi-spike attention and quantization"""
    def __init__(self, in_feat, out_feat, num_heads=8, dropout=0.0, act=None):
        super(QuantizedEncoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        
        # Quantized weight parameter
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        # Multi-spike attention
        self.attention = SpikeAttention(out_feat, num_heads)
        
        # Multi-spike activation
        self.spike_act = Multispike()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        # Standard graph convolution
        feat_embedding = torch.mm(feat, self.weight)
        x = torch.spmm(adj, feat_embedding)
        
        # Apply multi-spike activation
        x = self.spike_act(x)
        
        # Apply self-attention enhancement
        x_enhanced = self.attention(x)
        
        # Residual connection
        x = x + x_enhanced
        
        return feat_embedding, x

class QuantizedDecoder(Module):
    """Enhanced decoder with multi-spike activation"""
    def __init__(self, in_feat, out_feat, dropout=0.0, act=None):
        super(QuantizedDecoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        # Multi-spike activation
        self.spike_act = Multispike()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        x = self.spike_act(x)
        return x

class EnhancedMLP(nn.Module):
    """Enhanced MLP with multi-spike activations"""
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(EnhancedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.spike_act = Multispike()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.spike_act(out)
        out = self.fc2(out)
        return out

class Encoder_overall_enhanced(Module):
    """Enhanced overall encoder integrating SNN-Transformer technologies"""
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, 
                 num_heads=8, dropout=0.0, act=F.relu):
        super(Encoder_overall_enhanced, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act

        # Use quantized 1x1 convolutions
        self.conv1X1_omics1 = Conv2dLSQ(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1X1_omics2 = Conv2dLSQ(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Enhanced MLP with multi-spike activations
        self.MLP = EnhancedMLP(self.dim_out_feat_omics1 * 2, self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        
        # Enhanced encoders and decoders
        self.encoder_omics1 = QuantizedEncoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1, num_heads)
        self.decoder_omics1 = QuantizedDecoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = QuantizedEncoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2, num_heads)
        self.decoder_omics2 = QuantizedDecoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)
        
        # Multi-spike activation for final outputs
        self.final_activation = Multispike()
        
    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1, 
                adj_spatial_omics2, adj_feature_omics2):

        _adj_spatial_omics1 = adj_spatial_omics1.unsqueeze(0)
        _adj_feature_omics1 = adj_feature_omics1.unsqueeze(0)
        _adj_spatial_omics2 = adj_spatial_omics2.unsqueeze(0)
        _adj_feature_omics2 = adj_feature_omics2.unsqueeze(0)

        # Concatenate and process adjacency matrices
        cat_adj_omics1 = torch.cat((_adj_spatial_omics1, _adj_feature_omics1), dim=0)
        cat_adj_omics2 = torch.cat((_adj_spatial_omics2, _adj_feature_omics2), dim=0)

        adj_feature_omics1 = self.conv1X1_omics1(cat_adj_omics1).squeeze(0)
        adj_feature_omics2 = self.conv1X1_omics2(cat_adj_omics2).squeeze(0)

        # Enhanced encoding with attention
        feat_embedding1, emb_latent_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        feat_embedding2, emb_latent_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)

        # Enhanced MLP fusion
        cat_emb_latent = torch.cat((emb_latent_omics1, emb_latent_omics2), dim=1)
        emb_latent_combined = self.MLP(cat_emb_latent)

        # Enhanced decoding
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)

        # Apply final multi-spike activation
        emb_latent_combined = self.final_activation(emb_latent_combined)

        results = {
            'emb_latent_omics1': emb_latent_omics1,
            'emb_latent_omics2': emb_latent_omics2,
            'emb_latent_combined': emb_latent_combined,
            'emb_recon_omics1': emb_recon_omics1,
            'emb_recon_omics2': emb_recon_omics2,
        }
        
        return results

# Keep original classes for compatibility
class Encoder(Module): 
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        feat_embeding = torch.mm(feat, self.weight)
        x = torch.spmm(adj, feat_embeding)
        return feat_embeding, x
    
class Decoder(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x                  

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

class Encoder_overall(Module):
    """Original encoder for compatibility"""
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act

        self.conv1X1_omics1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1X1_omics2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.MLP = MLP(self.dim_out_feat_omics1 * 2, self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        
        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)
        
    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2, adj_feature_omics2):
        _adj_spatial_omics1 = adj_spatial_omics1.unsqueeze(0)
        _adj_feature_omics1 = adj_feature_omics1.unsqueeze(0)
        _adj_spatial_omics2 = adj_spatial_omics2.unsqueeze(0)
        _adj_feature_omics2 = adj_feature_omics2.unsqueeze(0)

        cat_adj_omics1 = torch.cat((_adj_spatial_omics1, _adj_feature_omics1), dim=0)
        cat_adj_omics2 = torch.cat((_adj_spatial_omics2, _adj_feature_omics2), dim=0)

        adj_feature_omics1 = self.conv1X1_omics1(cat_adj_omics1).squeeze(0)
        adj_feature_omics2 = self.conv1X1_omics2(cat_adj_omics2).squeeze(0)

        feat_embeding1, emb_latent_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        feat_embeding2, emb_latent_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)

        cat_emb_latent = torch.cat((emb_latent_omics1, emb_latent_omics2), dim=1)
        emb_latent_combined = self.MLP(cat_emb_latent)

        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)

        results = {'emb_latent_omics1':emb_latent_omics1,
                   'emb_latent_omics2':emb_latent_omics2,
                   'emb_latent_combined':emb_latent_combined,
                   'emb_recon_omics1':emb_recon_omics1,
                   'emb_recon_omics2':emb_recon_omics2,
                   }
        
        return results