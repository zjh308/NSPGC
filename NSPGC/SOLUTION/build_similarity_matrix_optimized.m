function S = build_similarity_matrix_optimized(X, k_base, dist_type)
% 优化版相似度矩阵构建：减少预处理复杂度
% 主要优化：
% 1. 使用更高效的距离计算
% 2. 优化k-NN搜索
% 3. 减少内存分配
% 4. 保持数学逻辑完全不变

n = size(X, 1);
d = size(X, 2);

% 参数默认值处理
if nargin < 2, k_base = 10; end
if nargin < 3
    dist_type = 'cosine';
    if d <= 50, dist_type = 'euclidean'; end
end

if n == 0, S = []; return; end

% === 优化1：更高效的距离计算 ===
if strcmp(dist_type, 'cosine')
    % 优化：预计算归一化，避免重复计算
    X_norm = X ./ (sqrt(sum(X.^2, 2)) + eps);
    cosine_sim = X_norm * X_norm';
    D = 1 - cosine_sim;
    D(D < 0) = 0;
else
    % 对于欧几里得距离，使用更高效的实现
    if n < 2000  % 小数据集直接计算
        D = pdist2(X, X);
    else  % 大数据集分块计算
        D = zeros(n, n);
        block_size = 500;
        for i = 1:block_size:n
            end_i = min(i + block_size - 1, n);
            D(i:end_i, :) = pdist2(X(i:end_i, :), X);
        end
    end
end

% === 优化2：向量化的自适应k值计算 ===
% 预分配数组
local_density = zeros(n, 1);
k_adaptive = zeros(n, 1);

% 批量排序和密度计算
for i = 1:n
    [d_sorted, ~] = sort(D(i, :));
    end_idx = min(k_base + 1, n);
    local_density(i) = mean(d_sorted(2:end_idx));
end

% 向量化计算自适应k值
k_min = max(3, floor(k_base / 2));
k_max = min(2 * k_base, n - 1);
max_density = max(local_density);
if max_density > eps
    density_norm = 1 - (local_density / max_density);
    k_adaptive = k_min + round((k_max - k_min) * density_norm);
else
    k_adaptive = k_base * ones(n, 1);
end

% === 优化3：高效稀疏矩阵构建 ===
% 预估稀疏矩阵大小
max_edges = n * max(k_adaptive);
row_idx = zeros(max_edges, 1);
col_idx = zeros(max_edges, 1);
values = zeros(max_edges, 1);
edge_count = 0;

% 批量构建稀疏矩阵
for i = 1:n
    [d_sorted, idx] = sort(D(i, :));
    k_i = k_adaptive(i);
    end_idx = min(k_i + 1, n);
    neighbors = idx(2:end_idx);
    dists = d_sorted(2:end_idx);
    
    if ~isempty(neighbors)
        sigma = mean(dists) + eps;
        weights = exp(-dists.^2 / (2 * sigma^2));
        weights(isnan(weights) | isinf(weights)) = 0;
        
        % 批量添加到稀疏矩阵数组
        num_neighbors = length(neighbors);
        idx_range = edge_count + (1:num_neighbors);
        row_idx(idx_range) = i;
        col_idx(idx_range) = neighbors;
        values(idx_range) = weights;
        edge_count = edge_count + num_neighbors;
    end
end

% 构建稀疏矩阵
row_idx = row_idx(1:edge_count);
col_idx = col_idx(1:edge_count);
values = values(1:edge_count);
S = sparse(row_idx, col_idx, values, n, n);

% === 优化4：高效对称化与归一化 ===
% 对称化（保持稀疏性）
S = (S + S') / 2;

% 优化的行归一化
row_sums = full(sum(S, 2));  % 只在必要时转换为dense
valid_rows = row_sums > eps;

% 使用稀疏矩阵的逐行归一化
for i = find(valid_rows)'
    S(i, :) = S(i, :) / row_sums(i);
end

% 处理孤立点
isolated_rows = ~valid_rows;
if any(isolated_rows)
    warning('发现 %d 个孤立样本，分配均匀相似度权重', sum(isolated_rows));
    % 为孤立点分配稀疏的均匀权重
    isolated_indices = find(isolated_rows);
    for i = isolated_indices'
        S(i, :) = 0;
        S(i, i) = 1;  % 自连接
    end
end

% 确保输出为稀疏矩阵
S = sparse(S);

end
