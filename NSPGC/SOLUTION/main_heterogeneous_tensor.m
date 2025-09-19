function [labels_pred,Y,history] = main_heterogeneous_tensor(X, params)
% 张量优化版主算法：使用张量优化的子函数实现最高性能
% 主要优化：
% 1. 使用张量版Y更新函数 - 预期加速2-4倍
% 2. 使用张量版梯度计算函数 - 预期加速3-6倍
% 3. 使用优化版相似度矩阵构建
% 4. 保持算法逻辑完全不变，仅提升计算效率

% ---- 读取/补齐参数 ----
if ~isfield(params, 'lambda'), params.lambda = 0.05; end
if ~isfield(params, 'alpha'), params.alpha = 0.05; end
if ~isfield(params, 'tau'), params.tau = 0.01; end
if ~isfield(params, 'maxIter'), params.maxIter = 50; end
if ~isfield(params, 'verbose'), params.verbose = true; end
if ~isfield(params, 'nCluster'), params.nCluster = 3; end
if ~isfield(params, 'k'), params.k = 10; end
if ~isfield(params, 'innerU_MaxIters'), params.innerU_MaxIters = 20; end

% ---- 基本量 ----
n = size(X,1);
c = params.nCluster;
k = params.k;

% 添加张量版本标识
if params.verbose
    fprintf('  张量版初始化: 样本数=%d, 特征数=%d, 类别数=%d, k=%d\n', n, size(X,2), c, k);
    fprintf('  参数设置: lambda=%.4f, alpha=%.4f, tau=%.4f, maxIter=%d\n', ...
        params.lambda, params.alpha, params.tau, params.maxIter);
    fprintf('  使用优化版核心算法确保逻辑正确性\n');
end

% ---- 构建相似度矩阵（使用优化版） ----
if params.verbose
    fprintf('  构建优化版相似度矩阵...\n');
end
if isfield(params, 'dist_type')
    S = build_similarity_matrix_optimized(X, k, params.dist_type);
else
    S = build_similarity_matrix_optimized(X, k);
end

% ---- 初始化 Y 和 v ----
if params.verbose
    fprintf('  执行K-means初始化...\n');
end

% 改进的K-means初始化：多次运行选择最佳结果
best_inertia = inf;
best_lab = [];
for trial = 1:5
    [temp_lab, ~, temp_inertia] = kmeans(X, c, 'Replicates', 3, 'MaxIter', 100);
    if sum(temp_inertia) < best_inertia
        best_inertia = sum(temp_inertia);
        best_lab = temp_lab;
    end
end
lab = best_lab;
Y = full(sparse((1:n)', lab, 1, n, c));

% v初始化：与优化版本保持一致
rng(42 + n);
v = 0.3 + 0.4*rand(n,1);  % 与优化版本相同的初始化策略

% ---- 记录器 ----
history.obj = zeros(params.maxIter,1);
history.tau = zeros(params.maxIter,1);
history.v_stat = zeros(params.maxIter,3);
history.hetero_effect = zeros(params.maxIter,1);
history.v = zeros(n, 1);

tau = params.tau;

% ---- 外层迭代 ----
for it = 1:params.maxIter
    if params.verbose
        fprintf('Iter %2d: ', it);
    end
    
    % (1) 更新 Y（使用优化版函数确保逻辑正确性）
    Y_old = Y;
    tic;
    
    % 使用固定正则化参数，与优化版本保持一致
    effective_lambda = params.lambda;
    effective_alpha = params.alpha;
    
    Y = update_Y_heterogeneous_tensor(Y, v, S, effective_lambda, effective_alpha);
    y_update_time = toc;
    
    % 检查Y是否真的改变了
    Y_changed = any(Y(:) ~= Y_old(:));
    cluster_sizes = sum(Y,1);
    
    % 添加Y更新完成信息
    if params.verbose && mod(it, 10) == 0
        fprintf('Y更新完成(%.3fs) | ', y_update_time);
    end
    
    % (2) 更新 v（使用张量版梯度函数）
    % 清理persistent变量（在第一次迭代时）
    if it == 1
        clear update_v_heterogeneous_tensor; % 清理persistent变量
    end
    tic;
    [v, tau] = update_v_heterogeneous_tensor(v, Y, S, params.lambda, params.alpha, tau, 0.3, it, params.maxIter, X);
    v_update_time = toc;
    
    % 添加v更新完成信息
    if params.verbose && mod(it, 10) == 0
        fprintf('v更新完成(%.3fs) | ', v_update_time);
    end
    
    % (3) 计算异簇效果指标（优化：减少计算频率）
    if params.verbose || it <= 5 || mod(it, 10) == 0
        hetero_effect = compute_heterogeneous_effect_tensor(v, Y, S);
    else
        hetero_effect = history.hetero_effect(max(1, it-1));  % 复用上次值
    end
    
    % (4) 记录历史（优化：减少不必要的目标函数计算）
    % 只在需要时计算目标函数值
    if params.verbose || it <= 10 || mod(it, 5) == 0
        [obj_val, ~] = obj_grad_v_heterogeneous_tensor(v, Y, S, effective_lambda, effective_alpha, tau);
    else
        obj_val = history.obj(it-1);  % 复用上次值，减少计算
    end
    
    history.obj(it) = obj_val;
    history.tau(it) = tau;
    history.v_stat(it,:) = [min(v) median(v) max(v)];
    history.hetero_effect(it) = hetero_effect;
    
    % (5) 输出监控信息
    if params.verbose
        fprintf('tau=%.3f | obj=%.2e | hetero=%.3f | v=[%.3f,%.3f,%.3f] | sizes=[%s]\n', ...
            tau, obj_val, hetero_effect, min(v), median(v), max(v), ...
            sprintf('%d ', cluster_sizes));
    end
    
    % (6) 连续稳定性收敛检查
    if it > 5  % 需要足够历史数据
        % 检查连续5次目标函数变化是否都很小
        consecutive_stable = true;
        min_stable_count = 5;
        
        if it >= min_stable_count
            for check_it = (it-min_stable_count+1):it
                if check_it > 1
                    obj_change_check = abs(history.obj(check_it) - history.obj(check_it-1)) / (abs(history.obj(check_it-1)) + 1e-10);
                    if obj_change_check >= 1e-3  % 放宽阈值：从1e-5调整为1e-3
                        consecutive_stable = false;
                        break;
                    end
                end
            end
        else
            consecutive_stable = false;
        end
        
        % 检查Y矩阵连续稳定性（修复：需要连续多次稳定）
        Y_consecutive_stable = false;
        min_Y_stable_count = 5;  % 至少连续5次Y不变化
        if it >= min_Y_stable_count
            % 检查连续5次Y矩阵是否都没有变化
            Y_consecutive_stable = true;
            % 这里需要跟踪Y变化历史，暂时使用更严格的条件
            % 要求当前Y不变化且目标函数也连续稳定
            Y_consecutive_stable = ~Y_changed && consecutive_stable;
        end
        
        % === 收敛判定 ===
        converged = false;
        convergence_reason = '';
        
        % 主要收敛条件：连续多次目标函数稳定
        if consecutive_stable
            converged = true;
            convergence_reason = sprintf('连续%d次目标函数稳定 (变化<1e-3)', min_stable_count);
        end
        
        % 辅助收敛条件：Y矩阵完全稳定（修复：需要同时满足目标函数稳定）
        if Y_consecutive_stable && it > 15
            converged = true;
            if ~isempty(convergence_reason)
                convergence_reason = [convergence_reason, ' + Y完全稳定'];
            else
                convergence_reason = 'Y完全稳定且目标函数连续稳定';
            end
        end
        
        % 快速收敛条件：Y矩阵稳定且目标函数变化很小
        if it >= 5 && ~Y_changed
            % 检查连续3次目标函数变化小于1e-3
            if it >= 3
                recent_changes_small = true;
                for check_recent = (it-2):it
                    if check_recent > 1
                        recent_change = abs(history.obj(check_recent) - history.obj(check_recent-1)) / (abs(history.obj(check_recent-1)) + 1e-10);
                        if recent_change >= 1e-3
                            recent_changes_small = false;
                            break;
                        end
                    end
                end
                
                if recent_changes_small
                    converged = true;
                    convergence_reason = 'Y稳定且连续3次目标函数变化<1e-3';
                end
            end
        end
        
        if converged
            if params.verbose
                fprintf('  *** 算法收敛 *** %s (第%d次迭代)\n', convergence_reason, it);
            end
            break;
        end
        
        % === 收敛性监控信息 ===
        if params.verbose && mod(it, 5) == 0
            current_obj_change = abs(history.obj(it) - history.obj(it-1)) / (abs(history.obj(it-1)) + 1e-10);
            fprintf('  收敛监控: 当前Δobj=%.2e, Y变化=%s\n', ...
                current_obj_change, char(Y_changed*'是'+(~Y_changed)*'否'));
        end
    end
    
    % (7) 数值稳定性检查
    if any(isnan(v)) || any(isinf(v)) || any(isnan(Y(:))) || any(isinf(Y(:)))
        warning('检测到数值不稳定，终止迭代');
        break;
    end
end

% ---- 最终处理 ----
history.v = v;

% 从Y矩阵得到标签
[~, labels_pred] = max(Y, [], 2);

if params.verbose
    fprintf('  张量版算法完成，总迭代次数: %d\n', it);
    fprintf('  最终簇大小: [%s]\n', sprintf('%d ', sum(Y,1)));
    fprintf('  最终异簇效果: %.4f\n', hetero_effect);
    
    % 生成收敛性分析图表（用于ICASSP论文）
    if it > 1
        try
            generate_convergence_plot(history, 'Current Dataset', '');
            fprintf('  收敛性分析图表已生成\n');
        catch ME
            fprintf('  收敛性图表生成失败: %s\n', ME.message);
        end
    end
end
