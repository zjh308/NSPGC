function b = compute_bvec_heterogeneous_optimized(v, Y, S, lambda, alpha, X)
% 优化版计算包含异簇推离机制的b向量，用于tau的自适应调整
% 优化内容：
% - 向量化重构项计算，避免双重循环
% - 优化正则项损失计算
% - 改进局部密度计算
% - 向量化数值稳定性检查
%
% 输入:
%   v: n×1 样本权重
%   Y: n×c 聚类标签矩阵
%   S: n×n 相似度矩阵
%   lambda: 同簇正则化参数
%   alpha: 异簇推离参数
%   X: n×d 特征矩阵（可选，用于计算局部密度）
% 输出:
%   b: n×1 b向量

[n, c] = size(Y);
eps_stable = 1e-12;

% 向量化数值稳定性检查
nan_inf_mask = isnan(v) | isinf(v);
if any(nan_inf_mask)
    warning('v contains %d NaN or Inf values in compute_bvec_heterogeneous_optimized', sum(nan_inf_mask));
    v = max(0, min(1, v));  % 修复有问题的值
end

% === 优化版局部密度计算 ===
if nargin >= 6 && ~isempty(X)
    try
        % 向量化度计算
        degree = sum(S, 2);
        degree_range = max(degree) - min(degree);
        
        if degree_range > eps
            local_density = (degree - min(degree)) / (degree_range + eps);
        else
            local_density = 0.5 * ones(n, 1);
        end
    catch ME
        warning('Local density calculation failed: %s', ME.message);
        local_density = 0.5 * ones(n, 1);
    end
else
    local_density = 0.5 * ones(n, 1);
end

% === 优化版向量化重构项计算 ===
% 原始: recon_terms(i,:) = sum_j S(i,j) * v(j) * Y(j,:)
% 向量化: recon_terms = S * diag(v) * Y
v_diag = spdiags(v, 0, n, n);  % 创建稀疏对角矩阵
recon_terms = S * v_diag * Y;  % 向量化计算所有重构项

% === 优化版重构损失计算 ===
% 向量化计算所有样本的重构损失
diff_matrix = Y - recon_terms;  % n×c
recon_losses = sum(diff_matrix.^2, 2);  % n×1，每行的平方和

% 处理NaN/Inf值
nan_inf_recon = isnan(recon_losses) | isinf(recon_losses);
recon_losses(nan_inf_recon) = 1.0;

% === 优化版正则项损失计算 ===
% 计算样本间的标签距离矩阵
Y_dist = pdist2(Y, Y, 'euclidean');  % n×n
same_cluster = Y_dist < 1e-6;  % 同簇标记矩阵

% 计算权重差异矩阵
v_diff_sq = (v - v').^2;  % n×n，v(i) - v(j)的平方

% 向量化正则项计算
% 同簇项：lambda * S(i,j) * (v(i) - v(j))^2
same_cluster_reg = lambda * (S .* same_cluster .* v_diff_sq);

% 异簇项：-alpha * S(i,j) * (v(i) - v(j))^2  
diff_cluster_reg = -alpha * (S .* (~same_cluster) .* v_diff_sq);

% 每个样本的正则项损失（按行求和）
reg_losses = sum(same_cluster_reg + diff_cluster_reg, 2);

% === 优化版合并总难度 ===
% 向量化密度权重计算
density_weights = 1 + (1 - local_density);  % n×1

% 向量化总难度计算
b = recon_losses .* density_weights + reg_losses + eps_stable;

% 向量化数值稳定性检查
nan_inf_b = isnan(b) | isinf(b);
b(nan_inf_b) = 1.0;

% === 优化版归一化 ===
try
    b_range = max(b) - min(b);
    if b_range > eps_stable && max(b) > eps_stable
        % 向量化缩放到[0,10]区间
        b = (b - min(b)) / b_range * 10;
    else
        b = ones(n, 1);
    end
catch
    warning('b vector normalization failed, using fallback');
    b = ones(n, 1);
end

% 最终向量化数值稳定性检查
final_nan_inf = isnan(b) | isinf(b);
if any(final_nan_inf)
    warning('b vector contains %d NaN or Inf values after processing, using fallback', sum(final_nan_inf));
    b(final_nan_inf) = 1.0;
end

% 向量化数值范围控制
b_min = min(b);
b_max = max(b);

if b_max > 100 && b_min < 0.01
    % 向量化缩放到[0.1, 10]范围
    b = 0.1 + (b - b_min) * 9.9 / (b_max - b_min);
    warning('b vector scaled to [0.1, 10] range for numerical stability');
elseif b_max > 1000
    % 向量化极端情况缩放
    b = b * (10 / b_max);
    warning('b vector scaled down to improve numerical stability');
end

end
