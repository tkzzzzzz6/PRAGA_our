import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import torch
import pandas as pd
import scanpy as sc
import numpy as np
import argparse
import time
from PRAGA.preprocess import fix_seed
from PRAGA.preprocess import clr_normalize_each_cell, pca
from PRAGA.preprocess import construct_neighbor_graph, lsi
from PRAGA.Train_model_enhanced import EnhancedTrain, Train
from PRAGA.Train_model_3M import Train_3M
from PRAGA.utils import clustering
from PRAGA import preprocess, preprocess_3M

def main(args):
    """Enhanced main function with SNN-Transformer integration"""
    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Enhanced model: {args.use_enhanced}")
    print(f"Attention heads: {args.num_heads}")

    # read data
    if args.data_type in ['10x', 'SPOTS', 'Stereo-CITE-seq']:
        adata_omics1 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_RNA.h5ad'))
        adata_omics2 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_ADT.h5ad'))
    elif args.data_type == 'Spatial-epigenome-transcriptome':
        adata_omics1 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_RNA.h5ad'))
        adata_omics2 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_peaks_normalized.h5ad'))
    elif args.data_type == 'Simulation':
        adata_omics1 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_RNA.h5ad'))
        adata_omics2 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_ADT.h5ad'))
        adata_omics3 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_ATAC.h5ad'))

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()
    if args.data_type == 'Simulation':
        adata_omics3.var_names_make_unique()

    # Fix random seed
    random_seed = 2024
    fix_seed(random_seed)

    # Preprocess
    if args.data_type == '10x':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, n_top_genes=3000, flavor='seurat_v3')
        adata_omics1 = adata_omics1[:, adata_omics1.var.highly_variable]
        clr_normalize_each_cell(adata_omics1)
        adata_omics1 = pca(adata_omics1, 50)
        
        # ADT
        clr_normalize_each_cell(adata_omics2)
        adata_omics2 = pca(adata_omics2, 20)
        
    elif args.data_type == 'SPOTS':  
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, n_top_genes=3000, flavor='seurat_v3')
        adata_omics1 = adata_omics1[:, adata_omics1.var.highly_variable]
        clr_normalize_each_cell(adata_omics1)
        adata_omics1 = pca(adata_omics1, 50)
        
        # ADT
        clr_normalize_each_cell(adata_omics2)
        adata_omics2 = pca(adata_omics2, 20)
        
    elif args.data_type == 'Stereo-CITE-seq':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, n_top_genes=3000, flavor='seurat_v3')
        adata_omics1 = adata_omics1[:, adata_omics1.var.highly_variable]
        clr_normalize_each_cell(adata_omics1)
        adata_omics1 = pca(adata_omics1, 50)
        
        # ADT
        clr_normalize_each_cell(adata_omics2)
        adata_omics2 = pca(adata_omics2, 20)
        
    elif args.data_type == 'Spatial-epigenome-transcriptome':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=20)
        sc.pp.highly_variable_genes(adata_omics1, n_top_genes=3000, flavor='seurat_v3')
        adata_omics1 = adata_omics1[:, adata_omics1.var.highly_variable]
        clr_normalize_each_cell(adata_omics1)
        adata_omics1 = pca(adata_omics1, 50)
        
        # ATAC
        adata_omics2 = lsi(adata_omics2, 50)
        
    elif args.data_type == 'Simulation':
        if args.n_clusters <= 20:
            # 2 modalities
            # RNA
            sc.pp.filter_genes(adata_omics1, min_cells=3)
            sc.pp.highly_variable_genes(adata_omics1, n_top_genes=3000)
            adata_omics1 = adata_omics1[:, adata_omics1.var.highly_variable]
            clr_normalize_each_cell(adata_omics1)
            adata_omics1 = pca(adata_omics1, 50)
            
            # ADT
            clr_normalize_each_cell(adata_omics2)
            adata_omics2 = pca(adata_omics2, 20)
            
            print('2 modalities')
            
            data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}
            
            # Use Enhanced Train
            if args.use_enhanced:
                train_model = EnhancedTrain(
                    data, 
                    args.data_type, 
                    device, 
                    random_seed, 
                    dim_output=args.dim_output,
                    num_heads=args.num_heads,
                    use_enhanced=True
                )
            else:
                train_model = Train(
                    data, 
                    args.data_type, 
                    device, 
                    random_seed, 
                    dim_output=args.dim_output
                )
        else:
            # 3 modalities
            # RNA
            sc.pp.filter_genes(adata_omics1, min_cells=3)
            sc.pp.highly_variable_genes(adata_omics1, n_top_genes=3000)
            adata_omics1 = adata_omics1[:, adata_omics1.var.highly_variable]
            clr_normalize_each_cell(adata_omics1)
            adata_omics1 = pca(adata_omics1, 50)
            
            # ADT
            clr_normalize_each_cell(adata_omics2)
            adata_omics2 = pca(adata_omics2, 20)
            
            # ATAC
            adata_omics3 = lsi(adata_omics3, 50)
            
            print('3 modalities')
            
            data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2, 'adata_omics3': adata_omics3}
            
            train_model = Train_3M(data, args.data_type, device, random_seed, dim_output=args.dim_output)

    # For non-Simulation data types
    if args.data_type != 'Simulation':
        data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}
        
        # Use Enhanced Train
        if args.use_enhanced:
            train_model = EnhancedTrain(
                data, 
                args.data_type, 
                device, 
                random_seed, 
                dim_output=args.dim_output,
                num_heads=args.num_heads,
                use_enhanced=True
            )
        else:
            train_model = Train(
                data, 
                args.data_type, 
                device, 
                random_seed, 
                dim_output=args.dim_output
            )

    # Training
    print("Starting training...")
    start_time = time.time()
    
    if hasattr(train_model, 'train_model'):
        # Enhanced training
        save_path = f"model_checkpoint_enhanced_{args.data_type}.pth" if args.use_enhanced else None
        losses = train_model.train_model(
            epochs=args.epochs,
            lr=args.learning_rate,
            RNA_weight=args.RNA_weight,
            ADT_weight=args.ADT_weight,
            patience=args.patience,
            save_path=save_path
        )
    else:
        # Original training for 3M case
        train_model.train_model(args.RNA_weight, args.ADT_weight, args.ATAC_weight)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Get embeddings and cluster
    print("Generating embeddings and clustering...")
    
    if hasattr(train_model, 'get_embeddings'):
        # Enhanced model
        embeddings = train_model.get_embeddings()
        combined_emb = embeddings['combined']
        
        # Enhanced clustering
        cluster_labels = train_model.cluster_embeddings(args.n_clusters, combined_emb, args.init_k)
    else:
        # Original model
        combined_emb = train_model.emb_combined.detach().cpu()
        cluster_labels = clustering(combined_emb, args.n_clusters, 'GMM', args.init_k)

    print(f"Clustering completed. Found {len(np.unique(cluster_labels))} clusters")

    # Save results
    print("Saving results...")
    
    # Clustering results
    pd.DataFrame(cluster_labels, columns=['cluster']).to_csv(args.txt_out_path, index=False, header=False)
    print(f"Cluster labels saved to: {args.txt_out_path}")
    
    # Visualization (if embeddings have enough dimensions)
    if combined_emb.shape[1] >= 2:
        try:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            # PCA visualization
            if combined_emb.shape[1] > 2:
                pca = PCA(n_components=2)
                emb_2d = pca.fit_transform(combined_emb)
            else:
                emb_2d = combined_emb
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=cluster_labels, cmap='tab20', s=10)
            plt.colorbar(scatter)
            plt.title(f'Enhanced PRAGA Clustering Results ({args.data_type})')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.tight_layout()
            plt.savefig(args.vis_out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Visualization saved to: {args.vis_out_path}")
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    # Save enhanced model info
    if args.use_enhanced and hasattr(train_model, 'train_model'):
        info_path = args.txt_out_path.replace('.txt', '_enhanced_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Enhanced PRAGA Results\n")
            f.write(f"Data type: {args.data_type}\n")
            f.write(f"Attention heads: {args.num_heads}\n")
            f.write(f"Output dimension: {args.dim_output}\n")
            f.write(f"Training time: {training_time:.2f}s\n")
            f.write(f"Number of clusters: {len(np.unique(cluster_labels))}\n")
            f.write(f"Final losses: {losses['total'][-1]:.6f}\n" if 'losses' in locals() else "")
        print(f"Enhanced model info saved to: {info_path}")

    print("Analysis completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced PRAGA with SNN-Transformer Integration')
    
    # Original parameters
    parser.add_argument('--file_fold', type=str, default='./Data/HLN/', help='Path to data directory')
    parser.add_argument('--data_type', type=str, default='10x', 
                       choices=['10x', 'SPOTS', 'Stereo-CITE-seq', 'Spatial-epigenome-transcriptome', 'Simulation'],
                       help='Type of data')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for inference')
    parser.add_argument('--init_k', type=int, default=10, help='Estimated number of clusters for training')
    parser.add_argument('--KNN_k', type=int, default=20, help='Number of nearest neighbors')
    parser.add_argument('--RNA_weight', type=float, default=1, help='Reconstruction weight of RNA modality')
    parser.add_argument('--ADT_weight', type=float, default=1, help='Reconstruction weight of ADT modality')
    parser.add_argument('--ATAC_weight', type=float, default=1, help='Reconstruction weight of ATAC modality')
    parser.add_argument('--vis_out_path', type=str, default='results/enhanced_result.png', 
                       help='Path to save visualizations')
    parser.add_argument('--txt_out_path', type=str, default='results/enhanced_result.txt', 
                       help='Path to save cluster labels')
    
    # Enhanced parameters
    parser.add_argument('--use_enhanced', action='store_true', default=True,
                       help='Use enhanced model with SNN-Transformer integration')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dim_output', type=int, default=64, help='Output dimension')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.vis_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.txt_out_path), exist_ok=True)
    
    main(args)