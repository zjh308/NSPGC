function [f, g] = obj_grad_v_heterogeneous_tensor(v, Y, S, lambda, alpha, tau)
% 修复版张量目标函数和梯度计算
% 修复关键问题：使用与优化版本相同的数学逻辑，确保准确率

n = size(Y,1);
c = size(Y,2);

% 数值稳定性检查
eps_stable = 1e-12;
if any(isnan(v)) || any(isinf(v))
    warning('v contains NaN or Inf values in obj_grad_v_heterogeneous_tensor');
    v = max(min(v, 1.0), 0.0);
end

% === 使用与优化版本相同的向量化计算 ===
V_diag = spdiags(v, 0, n, n);  % 稀疏对角矩阵
recon_terms = V_diag * S * Y;  % 完全稀疏操作，避免v.*Y的dense结果

% 初始化
f = 0;
g = zeros(n,1);

% ========== 第一项：重构损失（完全向量化） ==========
diff = Y - recon_terms;  % n×c 误差矩阵
recon_losses = sum(diff .^ 2, 2);  % n×1 重构损失
f = f + sum(v .* recon_losses);    % 目标函数累加
g = g + recon_losses;             % 梯度基础部分

% ========== 预计算：稀疏矩阵非零元素 + 簇标签 ==========
[i_idx, j_idx, s_vals] = find(S > eps_stable);  % 一次性获取所有非零(i,j)对
num_nonzero = length(i_idx);

% 预计算簇标签，避免重复的max操作
[~, cluster_labels] = max(Y, [], 2);  % n×1 簇标签（one-hot转索引）

% ========== 重构损失的邻居梯度贡献（修复版） ==========
if num_nonzero > 0
    % 1. 批量计算所有内积（利用one-hot特性）
    col_idx = cluster_labels(j_idx);                  % 所有j对应的簇标签
    linear_idx = sub2ind(size(diff), i_idx, col_idx); % 转换为diff的线性索引
    inner_prod_vec = diff(linear_idx);                % 批量获取内积值
    
    % 2. 计算所有贡献项并累加
    contribution = -2 * v(i_idx) .* s_vals .* inner_prod_vec;
    delta_g_recon = accumarray(j_idx, contribution, [n, 1]);
    g = g + delta_g_recon;
end

% ========== 样本特异性惩罚项（与优化版本保持一致） ==========
try
    if n > 1 && length(recon_losses) == n
        loss_threshold = quantile(recon_losses, 0.3);
        core_flag = (recon_losses <= loss_threshold);
        boundary_flag = ~core_flag;
        penalty_coeff = 0.1;
        f = f + penalty_coeff * sum(v .* core_flag) - penalty_coeff * sum(v .* boundary_flag);
        g(core_flag) = g(core_flag) + penalty_coeff;
        g(boundary_flag) = g(boundary_flag) - penalty_coeff;
    end
catch ME
    warning('Sample-specific penalty calculation failed: %s', ME.message);
end

% ========== 第二项+第三项：正则化项（与优化版本完全一致） ==========
if num_nonzero > 0
    avg_recon_loss = mean(recon_losses) + eps_stable;
    lambda_scaled = lambda * avg_recon_loss;
    alpha_scaled = alpha * avg_recon_loss;
    
    % === 预分配向量，减少内存分配 ===
    v_i_vec = v(i_idx);
    v_j_vec = v(j_idx);
    v_diff_vec = v_i_vec - v_j_vec;
    v_diff_sq_vec = v_diff_vec .^ 2;
    same_cluster_vec = (cluster_labels(i_idx) == cluster_labels(j_idx));
    
    % === 批量计算正则项 ===
    % 同簇项
    same_indices = same_cluster_vec;
    if any(same_indices)
        reg_same = lambda_scaled * s_vals(same_indices) .* v_diff_sq_vec(same_indices);
        f = f + sum(reg_same);
        
        % 梯度贡献
        grad_same = 2 * lambda_scaled * s_vals(same_indices) .* v_diff_vec(same_indices);
        delta_g_same_i = accumarray(i_idx(same_indices), grad_same, [n, 1]);
        delta_g_same_j = accumarray(j_idx(same_indices), -grad_same, [n, 1]);
        g = g + delta_g_same_i + delta_g_same_j;
    end
    
    % 异簇项
    diff_indices = ~same_cluster_vec;
    if any(diff_indices)
        reg_diff = -alpha_scaled * s_vals(diff_indices) .* v_diff_sq_vec(diff_indices);
        f = f + sum(reg_diff);
        
        % 梯度贡献
        grad_diff = -2 * alpha_scaled * s_vals(diff_indices) .* v_diff_vec(diff_indices);
        delta_g_diff_i = accumarray(i_idx(diff_indices), grad_diff, [n, 1]);
        delta_g_diff_j = accumarray(j_idx(diff_indices), -grad_diff, [n, 1]);
        g = g + delta_g_diff_i + delta_g_diff_j;
    end
end

% ========== 第四项：自步学习项 ==========
f = f - tau * sum(v);
g = g - tau * ones(n, 1);

% 梯度裁剪
max_grad = 100;
g = min(max(g, -max_grad), max_grad);

% 数值稳定性检查
if isnan(f) || isinf(f)
    warning('Objective function is NaN or Inf, using fallback');
    f = 1e6;
end
if any(isnan(g)) || any(isinf(g))
    warning('Gradient contains NaN or Inf, using fallback');
    g = zeros(n, 1);
end

end
