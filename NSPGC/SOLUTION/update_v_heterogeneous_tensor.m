function [v, tau] = update_v_heterogeneous_tensor(v, Y, S, lambda, alpha, tau, beta, it, it_max, X)
% 张量版v更新函数，使用张量版梯度计算
% 保持与原版本完全相同的逻辑，仅替换梯度计算为张量版本

n = numel(v);

% === 1) 优化版计算 b，并做稳健缩放成 b_eff ===
% 使用优化版b向量计算函数
if nargin >= 10 && ~isempty(X)
    b_raw = compute_bvec_heterogeneous_optimized(v, Y, S, lambda, alpha, X);   % n×1
else
    b_raw = compute_bvec_heterogeneous_optimized(v, Y, S, lambda, alpha);   % n×1
end

% 优化版数值稳定性检查 - 向量化操作
nan_inf_mask = isnan(b_raw) | isinf(b_raw);
if any(nan_inf_mask)
    warning('b_raw contains %d NaN or Inf values in update_v_heterogeneous_tensor, using fallback', sum(nan_inf_mask));
    b_raw(nan_inf_mask) = 1.0;  % 只替换有问题的值
end

% 优化版确保b向量有合理的分布
if all(b_raw < 1e-6)
    warning('b_raw too small in update_v_heterogeneous_tensor, using fallback');
    b_raw = ones(n,1);
end

% === 优化版历史难度对比机制 ===
persistent b_raw_history;
persistent n_samples;
persistent history_buffer_size;

% 初始化或检查样本数量变化
if isempty(n_samples) || n_samples ~= n
    n_samples = n;
    history_buffer_size = 5;  % 固定缓冲区大小
    b_raw_history = repmat(b_raw, 1, 1);  % 初始化为当前值
else
    % 优化版历史更新 - 避免重复维度检查
    if size(b_raw_history, 2) >= history_buffer_size
        % 滑动窗口更新
        b_raw_history = [b_raw_history(:, 2:end), b_raw];
    else
        b_raw_history = [b_raw_history, b_raw];
    end
end

% 向量化历史最小值计算
b_min_hist = min(b_raw_history(:));
b_shift = b_raw - b_min_hist;

% 优化版稳健尺度计算
b_scale = max(b_shift) * 0.8 + eps;
b_eff = b_shift / b_scale;

% 向量化限制最大难度
b_eff = min(b_eff, 1.5);

% === 2) 优化版设定当轮预算 ρ_t（修复：后期稳定化） ===
rho_start = 0.01;   
rho_end = 0.99;  
if it <= 20
    % 前20次迭代正常增长
    ramp = min(1, (it / max(1, 20))^0.3);
    rho_t = rho_start + (rho_end - rho_start) * ramp;
else
    % 20次后稳定在较高值，避免持续变化
    rho_t = 0.95;  % 固定在95%，给算法收敛机会
end

% === 3) 优化版自然增大的 τ 计算 ===
kappa = 0.2;

% 优化版安全的tau_tgt计算
try
    % 使用更稳定的分位数计算
    if length(b_eff) > 1
        tau_tgt = quantile(b_eff, rho_t) + kappa * std(b_eff);
    else
        tau_tgt = b_eff(1) + kappa * 0.1;  % 单样本情况
    end
    
    % 数值稳定性检查
    if ~isfinite(tau_tgt) || tau_tgt > 1e6
        warning('tau_tgt is unstable (%.2e) in update_v_heterogeneous_tensor, using fallback', tau_tgt);
        tau_tgt = 1.0;
    end
    
    % 确保单调增长
    tau_tgt = max(tau_tgt, tau * 1.05);
    
catch ME
    warning('Error in tau_tgt calculation in update_v_heterogeneous_tensor: %s, using fallback', ME.message);
    tau_tgt = tau * 1.2;
end

% 优化版平滑更新tau
tau_new = max((1 - beta) * tau + beta * tau_tgt, tau * 1.05);
tau = max(tau, tau_new);

% 优化版智能上界约束 - 向量化条件判断
if it <= 5
    tau_max = 100.0;
elseif it <= 15
    tau_max = 50.0;
else
    tau_max = 40.0;
end
tau = min(tau, tau_max);

% === 4) 优化版 fmincon 解带预算的子问题 ===
% 等式约束：sum(v) = rho_t * n
Aeq = ones(1, n); 
beq = rho_t * n;

% 边界约束
lb = zeros(n, 1); 
ub = ones(n, 1);

% === 高效动态配置 fmincon 优化器 ===
if it <= 5
    % 极早期：超快求解
    max_inner_iter = 8;
    opt_tol = 1e-3;
elseif it <= 15
    % 早期迭代：快速求解
    max_inner_iter = 12;
    opt_tol = 1e-4;
else
    % 后期迭代：精确求解
    max_inner_iter = 20;
    opt_tol = 1e-5;
end

% 高效fmincon选项配置
options_optimized = optimoptions('fmincon', ...
    'Algorithm', 'sqp', ...  % SQP通常比interior-point更快
    'SpecifyObjectiveGradient', true, ...
    'HessianApproximation', 'bfgs', ...  % BFGS比LBFGS在小规模问题上更快
    'Display', 'none', ...
    'MaxIterations', max_inner_iter, ...
    'OptimalityTolerance', opt_tol, ...
    'StepTolerance', 1e-6, ...  % 放宽步长容忍度
    'CheckGradients', false, ...
    'UseParallel', false);

% 使用张量版目标函数
fun = @(vv) obj_grad_v_heterogeneous_tensor(vv, Y, S, lambda, alpha, tau);

% 执行优化
try
    v = fmincon(fun, v, [], [], Aeq, beq, lb, ub, [], options_optimized);
catch ME
    warning('fmincon failed in update_v_heterogeneous_tensor: %s, using fallback', ME.message);
    % 简单fallback：保持当前v，只做归一化
    v = v / sum(v) * beq;
    v = max(0, min(1, v));  % 确保在边界内
end

% 优化版确保v不为全零
if all(v < 1e-6)
    warning('v update resulted in all zeros in update_v_heterogeneous_tensor, using uniform fallback');
    v = (beq / n) * ones(n, 1);  % 均匀分布满足预算约束
end

% 最终数值稳定性检查
v = max(lb, min(ub, v));  % 确保在边界内
if abs(sum(v) - beq) > 1e-6
    % 重新归一化以满足等式约束
    v = v / sum(v) * beq;
end

end
