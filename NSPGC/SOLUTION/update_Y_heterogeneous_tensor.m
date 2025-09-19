function Y = update_Y_heterogeneous_tensor(Y, v, S, lambda, alpha)
% 张量优化版Y更新函数 - 包含完整异簇推离机制
% 与优化版本保持相同的参数接口：(Y, v, S, lambda, alpha)
% 主要优化：使用张量化计算实现高效的异簇推离逻辑

if nargin < 4, lambda = 1e-3; end
if nargin < 5, alpha = 0.01; end

[n, c] = size(Y);
eps_stable = 1e-12;

% 数值稳定性检查
if any(isnan(v)) || any(isinf(v))
    warning('v contains NaN or Inf values in update_Y_heterogeneous_tensor');
    v = max(min(v, 1.0), 0.0);
end

% === 张量化预计算 ===
% 1. 预计算重构项：S * (v .* Y)
recon_terms = S * (v .* Y);

% 2. 计算重构损失的平均值以用于缩放正则项
recon_losses_all = sum((Y - recon_terms).^2, 2);
avg_recon_loss = mean(recon_losses_all(recon_losses_all > 0));
if isnan(avg_recon_loss) || avg_recon_loss < 1e-6, avg_recon_loss = 1; end

% 3. 预计算权重差异矩阵 (n×n)
v_diff_matrix = (v - v').^2;  % v_diff_matrix(i,j) = (v(i) - v(j))^2

% 4. 预计算加权权重差异矩阵
weighted_v_diff = S .* v_diff_matrix;  % S(i,j) * (v(i) - v(j))^2

% === 张量化损失计算 ===
% 计算每个样本到各簇的损失矩阵 U (n×c)
U = zeros(n, c);

for k = 1:c
    % 为簇k创建候选标签矩阵
    y_candidate = zeros(n, c);
    y_candidate(:, k) = 1;
    
    % 1. 张量化重构损失计算
    recon_diff = y_candidate - recon_terms;  % n×c
    recon_losses = sum(recon_diff.^2, 2);    % n×1
    
    % 2. 张量化同簇正则化项计算
    % 找到当前在簇k中的样本
    in_cluster_k = Y(:, k) == 1;  % n×1 逻辑向量
    
    % 同簇正则化：lambda * Σ_j S(i,j) * I(y_j = k) * (v_i - v_j)^2
    homo_reg_matrix = weighted_v_diff .* in_cluster_k';  % n×n .* 1×n = n×n
    homo_reg = lambda * sum(homo_reg_matrix, 2);  % n×1
    
    % 3. 张量化异簇正则化项计算
    % 找到当前不在簇k中的样本
    not_in_cluster_k = Y(:, k) == 0;  % n×1 逻辑向量
    
    % 异簇推离：-alpha * Σ_j S(i,j) * I(y_j ≠ k) * (v_i - v_j)^2
    % 在标签更新中作为正惩罚项处理
    hetero_reg_matrix = weighted_v_diff .* not_in_cluster_k';  % n×n .* 1×n = n×n
    hetero_reg = -alpha * sum(hetero_reg_matrix, 2);  % n×1 (负号表示推离)
    
    % 4. 总损失
    U(:, k) = v .* recon_losses + avg_recon_loss * (homo_reg + hetero_reg);
end

% 数值稳定性检查
U(isnan(U) | isinf(U)) = 1e6;

% === 高效ICM优化 ===
% 使用原始的高效增量策略
ff = sum(Y, 1);  % 1×c, 每个簇大小

% 簇平衡参数
lambda_bal_base = 5e-7;
min_cluster_size_base = 1;
max_cluster_size_base = 1000;

% 计算当前迭代进度
progress = (mean(v) - 0.1) / 0.9;
progress = max(0, min(1, progress));

% 渐进式约束强度
lambda_bal = lambda_bal_base * (progress^3);
min_cluster_size = min_cluster_size_base + round(4 * progress);
max_cluster_size = max_cluster_size_base - round(60 * progress);

% 计算簇平衡正则化项
target_size = n/c;

maxIter = 200;
for iterf = 1:maxIter
    converged = true;
    iter_changes = 0;

    for i = 1:n
        ui = U(i,:);
        id0 = find(Y(i,:)==1, 1);
        
        % 最小簇大小保护
        if ff(id0) <= min_cluster_size
            continue;
        end

        incre_F = zeros(1,c);
        incre_F(id0) = 0;

        % 试探移动到其它簇
        for k = 1:c
            if k == id0, continue; end
            
            % 最大簇大小限制
            if ff(k) >= max_cluster_size
                incre_F(k) = -inf;
                continue;
            end
            
            % 基础收益计算
            current_loss = U(i, id0);
            target_loss = U(i, k);
            base_gain = current_loss - target_loss;
            
            % 簇平衡正则化收益
            new_ff_id0 = ff(id0) - 1;
            new_ff_k = ff(k) + 1;
            
            old_balance = (ff(id0) - target_size)^2 + (ff(k) - target_size)^2;
            new_balance = (new_ff_id0 - target_size)^2 + (new_ff_k - target_size)^2;
            balance_gain = lambda_bal * (old_balance - new_balance);
            
            % 总收益
            incre_F(k) = base_gain + balance_gain;
        end

        % 选择最优簇
        [~, id] = max(incre_F);

        % 更新
        if id ~= id0
            converged = false;
            iter_changes = iter_changes + 1;

            % 更新 one-hot
            Y(i,id0) = 0; Y(i,id) = 1;

            % 更新簇大小
            ff(id0) = ff(id0) - 1;
            ff(id) = ff(id) + 1;
        end
    end

    % 终止条件
    if converged || iter_changes == 0
        break;
    end
end

end
