# 邻域感知的自步聚类框架 (NSPGC)

本项目是一个新颖的自步聚类算法的 MATLAB 实现。该算法的核心创新在于将邻域信息融入自步学习过程，同时包含"同簇吸引"和"异簇推离"的机制。项目包含多个优化版本，从基础实现到张量优化版本，提供了完整的实验框架和性能分析工具。

## 📊 数学模型

### 核心目标函数

本算法旨在通过交替优化样本权重向量 **v** 和簇分配矩阵 **Y** 来求解以下优化问题：

$$
\min_{\mathbf{v}, \mathbf{Y}} \mathcal{L}(\mathbf{v}, \mathbf{Y}) = \sum_{i=1}^{n} v_i \left\| \mathbf{y}_i - \sum_{j=1}^{n} s_{ij} v_j \mathbf{y}_j \right\|^2_F \
+ \lambda \sum_{i=1}^{n} \sum_{j=1}^{n} s_{ij} \mathbb{I}(\mathbf{y}_i = \mathbf{y}_j) (v_i - v_j)^2 \
- \alpha \sum_{i=1}^{n} \sum_{j=1}^{n} s_{ij} \mathbb{I}(\mathbf{y}_i \neq \mathbf{y}_j) (v_i - v_j)^2 \
- \tau \sum_{i=1}^{n} v_i \
\text{s.t.} \quad v_i \in [0, 1], \quad \mathbf{y}_i \in \{0,1\}^c, \quad \|\mathbf{y}_i\|_1 = 1
$$

**核心组成部分:**
-   **重构损失 (第一项)**: 标准的自步学习重构误差，其中 `v_i` 控制每个样本的影响力。
-   **同簇正则化 (第二项)**: “同簇吸引”项，鼓励同一簇内的邻近样本拥有相似的权重 (`v_i` ≈ `v_j`)。
-   **异簇正则化 (第三项)**: 创新的“异簇推离”项，鼓励不同簇之间的邻近样本拥有相异的权重。
-   **自步正则化 (第四项)**: 标准的自步学习项，其中 `tau` 作为一个动态阈值，实现从易到难的样本选择过程。

## 📂 项目结构

```
zjh_code_final_tensor_old copy/
├── SOLUTION/                    # 核心算法实现目录 (精简版)
│   ├── main_heterogeneous_tensor.m         # 张量优化版主算法 (唯一版本)
│   ├── update_Y_heterogeneous_tensor.m     # Y更新函数 (张量版)
│   ├── update_v_heterogeneous_tensor.m     # v更新函数 (张量版)
│   ├── obj_grad_v_heterogeneous_tensor.m   # 梯度计算函数 (张量版)
│   ├── compute_heterogeneous_effect_tensor.m # 异簇效果计算
│   ├── build_similarity_matrix_optimized.m # 相似度矩阵构建 (优化版)
│   ├── generate_convergence_plot.m         # 收敛性分析图表
│   ├── test_tensor_performance.m           # 张量版性能测试
│   ├── test_optimizations.m                # 优化效果测试
│   └── progressive_hyperparameter_search.m # 渐进式参数搜索
├── exp/                         # 实验脚本目录
│   ├── run_hyperparameter_search.m    # 主要超参数搜索脚本
│   ├── zjh_reproduce_jaffe_result.m   # Jaffe数据集复现脚本
│   ├── comprehensive_sensitivity_analysis.m # 敏感性分析
│   ├── compact_sensitivity_visualization.m  # 可视化脚本
│   └── hyperparameter_results/        # 搜索结果保存
├── data/                        # 数据集目录 (存放 .mat 文件)
├── datasets/                    # 预处理数据集
├── lib/                         # 评估函数库
├── reproduce_results/           # 复现结果保存目录
├── grid_search_ldccsd.m         # 网格搜索脚本
├── bayes_optimization_ldccsd.m  # 贝叶斯优化脚本
└── hyperparameter_search*.m     # 多种超参数搜索策略
```

## 📁 代码框架详解

### 核心算法文件 (`/SOLUTION`) - 精简版

#### 主算法入口

-   **`main_heterogeneous_tensor.m`** ⭐ (唯一版本)
    -   **角色**: 张量优化版主算法，提供最高性能
    -   **优化**: 使用张量运算优化Y更新(2-4倍加速)和梯度计算(3-6倍加速)
    -   **特点**: 保持算法逻辑完全不变，仅提升计算效率
    -   **调用链**: 统一使用张量优化版本的所有子函数

#### 核心更新函数 (张量优化版)

-   **`update_Y_heterogeneous_tensor.m`**: 
    -   张量优化版Y更新函数
    -   包含完整的异簇推离逻辑
    -   向量化邻域计算，复杂度从O(n²c)降至O(nk_NN·c)

-   **`update_v_heterogeneous_tensor.m`**: 
    -   张量优化版v更新函数
    -   使用fmincon求解器进行约束优化
    -   集成自步学习机制

#### 辅助计算函数

-   **`obj_grad_v_heterogeneous_tensor.m`**: 张量版梯度计算
-   **`compute_heterogeneous_effect_tensor.m`**: 异簇效果计算
-   **`build_similarity_matrix_optimized.m`**: 优化版相似度矩阵构建



## 🔬 实验脚本与工具

### 主要实验脚本 (`/exp`)

#### `run_hyperparameter_search.m` ⭐ (主要运行脚本)

-   **角色**: 优化版单数据集超参数搜索与性能评估的核心脚本
-   **特点**:
    -   使用张量优化算法 (`main_heterogeneous_tensor`) 提升性能
    -   支持开关控制搜索模式/手动参数模式
    -   自动输出Top5最佳参数组合
    -   支持多种子复现验证
-   **优化内容**:
    -   Y-update: 向量化邻域计算，复杂度从O(n²c)降至O(nk_NN·c)
    -   梯度计算: 完全向量化稀疏矩阵操作
    -   相似度矩阵: 优化距离计算和k-NN搜索




### 超参数优化工具

#### 多种搜索策略

-   **`grid_search_ldccsd.m`**: 传统网格搜索
-   **`bayes_optimization_ldccsd.m`**: 贝叶斯优化搜索  
-   **`hyperparameter_search.m`**: 基础超参数搜索
-   **`hyperparameter_search_config.m`**: 搜索配置管理

#### 分析工具

-   `analyze_hyperparameter_results.m`: 结果分析工具
-   `progressive_hyperparameter_search.m`: 渐进式搜索策略

## 🚀 快速开始

### 代码完整性验证

首先验证清理后的代码完整性：
```matlab
cd SOLUTION/
verify_cleanup()  % 验证所有核心函数完整且可运行
```

### 推荐使用流程

1. **运行主要实验脚本**:
   ```matlab
   cd exp/
   run_hyperparameter_search()  % 自动搜索最优参数
   ```
    

### 算法版本说明

- **唯一版本**: `main_heterogeneous_tensor.m` 
  - 经过多轮优化的最终版本
  - 集成了所有性能优化和逻辑修复
  - 适用于所有规模的数据集
  - 保证了算法的正确性和高效性

## 📊 性能优化特点

### 张量优化版本优势

1. **计算加速**:
   - Y更新函数: 2-4倍性能提升
   - 梯度计算: 3-6倍性能提升
   - 相似度矩阵构建: 显著优化

2. **算法完整性**:
   - 保持原始算法逻辑完全不变
   - 仅在计算层面进行优化
   - 确保结果的一致性和可重现性

3. **多种搜索策略**:
   - 网格搜索: 全面但计算量大
   - 贝叶斯优化: 智能搜索，适合复杂参数空间
   - 渐进式搜索: 逐步细化参数范围

## 📝 项目特点与优势

### 算法创新

1. **邻域感知机制**: 将邻域信息融入自步学习，提升聚类质量
2. **异簇推离策略**: 创新的异簇样本权重分化机制
3. **多层次优化**: 从基础实现到张量优化的完整优化路径

### 实验框架完善

1. **多种评估工具**: 敏感性分析、可视化、性能对比
2. **参数搜索策略**: 网格搜索、贝叶斯优化等多种方法
3. **结果复现支持**: 完整的复现脚本和参数配置

### 工程实践优化

1. **性能分层**: 基础版、优化版、张量版满足不同需求
2. **模块化设计**: 各功能模块独立，便于维护和扩展
3. **完善文档**: 详细的代码注释和使用说明
