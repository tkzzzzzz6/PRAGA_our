#!/bin/bash

# Enhanced PRAGA with SNN-Transformer Integration
# This script runs the enhanced version with quantized spike-driven transformer components

echo "==================================================="
echo "Enhanced PRAGA with SNN-Transformer Integration"
echo "==================================================="

# Create results directory
mkdir -p results

# Set default parameters
DATA_DIR="./Data/"
OUTPUT_DIR="results/"
EPOCHS=1000
LR=1e-4
NUM_HEADS=8
DIM_OUTPUT=64
PATIENCE=50

echo "Configuration:"
echo "- Data directory: $DATA_DIR"
echo "- Output directory: $OUTPUT_DIR"
echo "- Epochs: $EPOCHS"
echo "- Learning rate: $LR"
echo "- Attention heads: $NUM_HEADS"
echo "- Output dimension: $DIM_OUTPUT"
echo "- Early stopping patience: $PATIENCE"
echo ""

# Function to run enhanced PRAGA
run_enhanced_praga() {
    local dataset=$1
    local data_type=$2
    local n_clusters=$3
    local init_k=$4
    local rna_weight=$5
    local adt_weight=$6
    local atac_weight=${7:-1}
    
    echo "Running Enhanced PRAGA on $dataset dataset..."
    echo "Dataset type: $data_type"
    echo "Clusters: $n_clusters (init: $init_k)"
    echo "Weights - RNA: $rna_weight, ADT: $adt_weight, ATAC: $atac_weight"
    echo ""
    
    python main_enhanced.py \
        --file_fold "${DATA_DIR}${dataset}/" \
        --data_type "$data_type" \
        --n_clusters $n_clusters \
        --init_k $init_k \
        --KNN_k 20 \
        --RNA_weight $rna_weight \
        --ADT_weight $adt_weight \
        --ATAC_weight $atac_weight \
        --vis_out_path "${OUTPUT_DIR}${dataset}_enhanced.png" \
        --txt_out_path "${OUTPUT_DIR}${dataset}_enhanced.txt" \
        --use_enhanced \
        --num_heads $NUM_HEADS \
        --dim_output $DIM_OUTPUT \
        --epochs $EPOCHS \
        --learning_rate $LR \
        --patience $PATIENCE
    
    echo "Enhanced PRAGA completed for $dataset"
    echo "Results saved to:"
    echo "  - Visualization: ${OUTPUT_DIR}${dataset}_enhanced.png"
    echo "  - Cluster labels: ${OUTPUT_DIR}${dataset}_enhanced.txt"
    echo "  - Model info: ${OUTPUT_DIR}${dataset}_enhanced_info.txt"
    echo ""
}

# Function to run comparison with original PRAGA
run_comparison() {
    local dataset=$1
    local data_type=$2
    local n_clusters=$3
    local init_k=$4
    local rna_weight=$5
    local adt_weight=$6
    
    echo "Running comparison: Original vs Enhanced PRAGA on $dataset..."
    
    # Run original
    echo "Running Original PRAGA..."
    python main_enhanced.py \
        --file_fold "${DATA_DIR}${dataset}/" \
        --data_type "$data_type" \
        --n_clusters $n_clusters \
        --init_k $init_k \
        --KNN_k 20 \
        --RNA_weight $rna_weight \
        --ADT_weight $adt_weight \
        --vis_out_path "${OUTPUT_DIR}${dataset}_original.png" \
        --txt_out_path "${OUTPUT_DIR}${dataset}_original.txt" \
        --epochs $EPOCHS \
        --learning_rate $LR \
        --patience $PATIENCE
    
    # Run enhanced
    echo "Running Enhanced PRAGA..."
    run_enhanced_praga "$dataset" "$data_type" $n_clusters $init_k $rna_weight $adt_weight
    
    echo "Comparison completed for $dataset"
    echo ""
}

# Check if datasets exist
check_dataset() {
    local dataset=$1
    if [ ! -d "${DATA_DIR}${dataset}" ]; then
        echo "Warning: Dataset ${dataset} not found in ${DATA_DIR}"
        echo "Please ensure the dataset is downloaded and extracted."
        return 1
    fi
    return 0
}

# Main execution
case "${1:-all}" in
    "HLN")
        echo "Running Human Lymph Node (HLN) dataset..."
        if check_dataset "HLN"; then
            run_enhanced_praga "HLN" "10x" 6 6 1 1
        fi
        ;;
    
    "Mouse_Brain"|"MB")
        echo "Running Mouse Brain dataset..."
        if check_dataset "Mouse_Brain"; then
            run_enhanced_praga "Mouse_Brain" "Spatial-epigenome-transcriptome" 20 20 1 1
        fi
        ;;
    
    "Simulation")
        echo "Running Simulation dataset..."
        if check_dataset "Simulation"; then
            run_enhanced_praga "Simulation" "Simulation" 7 7 1 1
        fi
        ;;
    
    "compare")
        echo "Running comparison mode..."
        echo "This will run both original and enhanced versions for comparison."
        
        if check_dataset "HLN"; then
            echo "Comparing on HLN dataset..."
            run_comparison "HLN" "10x" 6 6 1 1
        fi
        
        if check_dataset "Mouse_Brain"; then
            echo "Comparing on Mouse Brain dataset..."
            run_comparison "Mouse_Brain" "Spatial-epigenome-transcriptome" 20 20 1 1
        fi
        ;;
    
    "all")
        echo "Running all available datasets with Enhanced PRAGA..."
        
        if check_dataset "HLN"; then
            run_enhanced_praga "HLN" "10x" 6 6 1 1
        fi
        
        if check_dataset "Mouse_Brain"; then
            run_enhanced_praga "Mouse_Brain" "Spatial-epigenome-transcriptome" 20 20 1 1
        fi
        
        if check_dataset "Simulation"; then
            run_enhanced_praga "Simulation" "Simulation" 7 7 1 1
        fi
        ;;
    
    "custom")
        echo "Custom run mode - modify the script to set your parameters"
        echo "Example:"
        echo "run_enhanced_praga \"YourDataset\" \"10x\" 10 10 1 1"
        ;;
    
    "help"|"-h"|"--help")
        echo "Enhanced PRAGA Usage:"
        echo "$0 [option]"
        echo ""
        echo "Options:"
        echo "  HLN              Run on Human Lymph Node dataset"
        echo "  Mouse_Brain|MB   Run on Mouse Brain dataset" 
        echo "  Simulation       Run on Simulation dataset"
        echo "  compare          Run comparison between original and enhanced"
        echo "  all              Run on all available datasets (default)"
        echo "  custom           Template for custom runs"
        echo "  help             Show this help message"
        echo ""
        echo "Configuration (edit script to modify):"
        echo "  EPOCHS=$EPOCHS"
        echo "  LR=$LR"
        echo "  NUM_HEADS=$NUM_HEADS"
        echo "  DIM_OUTPUT=$DIM_OUTPUT"
        echo "  PATIENCE=$PATIENCE"
        ;;
    
    *)
        echo "Unknown option: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

echo "==================================================="
echo "Enhanced PRAGA execution completed!"
echo "Check the results/ directory for outputs."
echo "==================================================="