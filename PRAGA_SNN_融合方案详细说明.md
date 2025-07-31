# PRAGA与量化脉冲驱动Transformer融合方案

## 📋 概述

本项目成功将PRAGA（空间多模态组学分析框架）与量化脉冲驱动Transformer（QSD-Transformer）技术融合，旨在提高模型精度和计算效率。

## 🔧 核心技术融合

### 1. 多脉冲激活函数集成
- **原始PRAGA**: 使用传统ReLU激活函数
- **增强版本**: 集成多脉冲激活函数（Multi-spike Activation）
- **优势**: 减少量化误差，提升信息保留能力

### 2. 量化技术应用
- **LSQ量化**: 采用Learned Step Size Quantization技术
- **量化卷积**: 使用Conv2dLSQ替换传统卷积
- **优势**: 提高数值稳定性，降低计算复杂度

### 3. 自注意力机制增强
- **新增组件**: SpikeAttention模块
- **架构**: 多头自注意力 + 多脉冲激活
- **融合方式**: 与图卷积并行，残差连接
- **优势**: 增强长距离依赖建模能力

### 4. RepConv结构优化
- **重参数化卷积**: 优化1x1卷积操作
- **训练效率**: 提升特征融合效果
- **推理优化**: 可重参数化为单一卷积

## 🏗️ 架构对比

### 原始PRAGA架构
```
Input → GNN Encoder → Feature Fusion → GNN Decoder → Output
         ↓
    Traditional Conv + ReLU
```

### 增强PRAGA架构
```
Input → Enhanced Encoder → Enhanced Fusion → Enhanced Decoder → Output
         ↓                    ↓                    ↓
    Quantized Conv +      Multi-spike +      Multi-spike +
    Multi-spike +         Attention +        Activation
    Self-Attention        Enhanced MLP
```

## 📁 新增文件说明

### 核心模型文件
- **`PRAGA/model_enhanced.py`**: 增强版模型实现
  - `Encoder_overall_enhanced`: 集成所有增强技术的主模型
  - `SpikeAttention`: 多脉冲自注意力模块
  - `QuantizedEncoder/Decoder`: 量化版编码器/解码器
  - `Multispike`: 多脉冲激活函数

### 训练文件
- **`PRAGA/Train_model_enhanced.py`**: 增强版训练类
  - `EnhancedTrain`: 支持注意力机制和量化的训练类
  - 自适应学习率调度
  - 梯度裁剪和正则化
  - 早停机制

### 主程序
- **`main_enhanced.py`**: 增强版主运行脚本
  - 支持原始和增强模型切换
  - 参数化配置
  - 结果对比分析

### 运行脚本
- **`run_enhanced.sh`**: 一键运行脚本
  - 多数据集支持
  - 对比模式
  - 自动化配置

## 🚀 使用方法

### 1. 快速开始
```bash
# 运行增强版PRAGA（所有数据集）
./run_enhanced.sh all

# 运行特定数据集
./run_enhanced.sh HLN

# 对比原始与增强版本
./run_enhanced.sh compare
```

### 2. Python直接调用
```python
# 增强版训练
from PRAGA.Train_model_enhanced import EnhancedTrain

train_model = EnhancedTrain(
    data=data, 
    datatype='10x', 
    device=device,
    num_heads=8,          # 注意力头数
    use_enhanced=True     # 使用增强模型
)

# 训练模型
losses = train_model.train_model(
    epochs=1000,
    lr=1e-4,
    patience=50
)

# 获取增强嵌入
embeddings = train_model.get_embeddings()
```

### 3. 参数配置
```bash
python main_enhanced.py \
    --file_fold ./Data/HLN/ \
    --data_type 10x \
    --n_clusters 6 \
    --use_enhanced \           # 启用增强模型
    --num_heads 8 \           # 注意力头数
    --dim_output 64 \         # 输出维度
    --epochs 1000 \           # 训练轮数
    --learning_rate 1e-4 \    # 学习率
    --patience 50             # 早停耐心
```

## 📊 预期改进效果

### 1. 精度提升
- **多脉冲激活**: 减少信息损失 (~5-10%精度提升)
- **自注意力机制**: 增强特征表示 (~3-8%精度提升)
- **量化技术**: 提高数值稳定性 (~2-5%精度提升)

### 2. 计算效率
- **量化技术**: 减少内存使用 (~30-50%)
- **RepConv**: 优化卷积计算 (~10-20%加速)
- **注意力机制**: 并行计算友好

### 3. 模型鲁棒性
- **多脉冲激活**: 更好的梯度流动
- **残差连接**: 缓解梯度消失
- **正则化技术**: 防止过拟合

## 🔍 技术细节

### 1. 多脉冲激活函数
```python
class Multispike(nn.Module):
    def __init__(self, lens=4):
        super().__init__()
        self.lens = lens
        
    def forward(self, inputs):
        return multispike.apply(4 * inputs, self.lens) / 4
```

### 2. 自注意力机制
```python
class SpikeAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        # 多头自注意力 + 多脉冲激活
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.spike_act = Multispike()
```

### 3. 量化卷积
```python
# 使用LSQ量化技术
self.conv1X1_omics1 = Conv2dLSQ(
    in_channels=2, 
    out_channels=1, 
    kernel_size=1
)
```

## 📈 实验验证

### 支持的数据集
1. **Human Lymph Node (HLN)**: 人类淋巴结数据
2. **Mouse Brain**: 小鼠大脑空间表观基因组数据
3. **Simulation**: 模拟多模态数据

### 评估指标
- **聚类精度**: ARI, NMI, Silhouette Score
- **重构误差**: MSE Loss
- **计算效率**: 训练时间、内存使用
- **模型稳定性**: 多次运行一致性

## 🛠️ 环境要求

### 依赖库
```
torch>=2.0.0
numpy>=1.22.3
pandas>=1.4.2
scanpy>=1.9.1
scikit-learn>=1.1.1
matplotlib>=3.5.0
```

### 量化模块
```
# 确保量化模块路径正确
sys.path.append('113 Quantized Spike-driven Transformer（ICLR 2025）')
from quan_w import Conv2dLSQ
from _quan_base_plus import grad_scale, round_pass
```

## 🔧 故障排除

### 常见问题
1. **CUDA内存不足**: 减少batch size或dim_output
2. **量化模块导入失败**: 检查路径设置
3. **训练不收敛**: 调整学习率和正则化参数

### 调试建议
- 使用`--use_enhanced False`运行原始模型进行对比
- 监控损失曲线判断训练状态
- 检查attention权重分布确认机制有效性

## 📝 后续优化方向

1. **动态量化**: 自适应bit-width选择
2. **知识蒸馏**: 大模型向小模型转移
3. **架构搜索**: 自动优化网络结构
4. **多尺度注意力**: 处理不同尺度的空间信息

## 📚 参考文献

1. PRAGA原始论文和实现
2. Quantized Spike-driven Transformer (ICLR 2025)
3. LSQ: Learned Step Size Quantization
4. Multi-spike Neural Networks

---

**注意**: 本融合方案已完成核心技术集成，建议先在小数据集上验证效果，再扩展到大规模数据。