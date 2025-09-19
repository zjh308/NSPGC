function [best_params, all_results, search_history] = bayes_optimization_ldccsd(X, Y_true, n_folds, max_trials, verbose)
% LDCCSD贝叶斯优化函数
% 使用高斯过程优化超参数，智能探索参数空间
%
% 输入:
%   X: n×d 数据矩阵
%   Y_true: n×1 真实标签
%   n_folds: 交叉验证折数
%   max_trials: 最大试验次数
%   verbose: 是否显示详细信息
%
% 输出:
%   best_params: 最优超参数组合
%   all_results: 所有参数组合的结果
%   search_history: 搜索过程历史

% 定义参数搜索范围
param_ranges = struct();
param_ranges.lambda = [0.001, 5.0];    % lambda: 0.001 ~ 5.0
param_ranges.tau = [0.0001, 1.0];     % tau: 0.0001 ~ 1.0
param_ranges.k = [3, 30];              % k: 3 ~ 30

if verbose
    fprintf('贝叶斯优化: %d次试验\n', max_trials);
    fprintf('参数范围:\n');
    fprintf('  lambda: [%.4f, %.4f]\n', param_ranges.lambda(1), param_ranges.lambda(2));
    fprintf('  tau: [%.4f, %.4f]\n', param_ranges.tau(1), param_ranges.tau(2));
    fprintf('  k: [%d, %d]\n', param_ranges.k(1), param_ranges.k(2));
    fprintf('开始搜索...\n\n');
end

% 初始化结果存储
all_results = struct('params', {}, 'metrics', {}, 'fold_scores', {});
best_score = -inf;
best_params = struct();

% 创建交叉验证索引
cv_indices = crossvalind('Kfold', length(Y_true), n_folds);

% 初始化高斯过程（如果可用）
use_gp = false;
try
    % 检查是否有Statistics and Machine Learning Toolbox
    if exist('fitrgp', 'file')
        use_gp = true;
        if verbose
            fprintf('使用高斯过程进行贝叶斯优化\n');
        end
    else
        if verbose
            fprintf('未检测到高斯过程工具箱，使用简化版贝叶斯优化\n');
        end
    end
catch
    if verbose
        fprintf('高斯过程初始化失败，使用简化版贝叶斯优化\n');
    end
end

% 贝叶斯优化
for trial = 1:max_trials
    if trial <= 5
        % 前5次使用随机采样进行初始化
        current_params = sample_random_params(param_ranges);
        if verbose
            fprintf('初始化试验 %d: lambda=%.4f, tau=%.4f, k=%d\n', ...
                trial, current_params.lambda, current_params.tau, current_params.k);
        end
    else
        % 后续使用贝叶斯优化
        if use_gp && trial > 5
            current_params = sample_bayes_params(param_ranges, all_results, verbose);
        else
            current_params = sample_random_params(param_ranges);
        end
        
        if verbose
            fprintf('贝叶斯试验 %d: lambda=%.4f, tau=%.4f, k=%d\n', ...
                trial, current_params.lambda, current_params.tau, current_params.k);
        end
    end
    
    % 设置模型参数
    current_params.maxIter = 30;
    current_params.verbose = false;
    current_params.nCluster = length(unique(Y_true));
    current_params.innerU_MaxIters = 20;
    
    % 交叉验证
    fold_scores = zeros(n_folds, 1);
    fold_metrics = struct('ACC', zeros(n_folds,1), 'NMI', zeros(n_folds,1), ...
                        'F1', zeros(n_folds,1), 'ARI', zeros(n_folds,1));
    
    valid_folds = 0;
    for fold = 1:n_folds
        try
            % 划分训练集和验证集
            train_idx = (cv_indices ~= fold);
            val_idx = (cv_indices == fold);
            
            X_train = X(train_idx, :);
            Y_train = Y_true(train_idx);
            X_val = X(val_idx, :);
            Y_val = Y_true(val_idx);
            
            % 训练模型
            [labels_pred, ~, ~] = main(X_train, current_params);
            
            % 在验证集上评估
            [~, nmi] = compute_nmi(Y_train, labels_pred);
            acc = Accuracy(labels_pred, double(Y_train));
            [f1, ~, ~] = compute_f(Y_train, labels_pred);
            [ari, ~, ~, ~] = RandIndex(Y_train, labels_pred);
            
            % 存储指标
            fold_metrics.ACC(fold) = acc;
            fold_metrics.NMI(fold) = nmi;
            fold_metrics.F1(fold) = f1;
            fold_metrics.ARI(fold) = ari;
            
            % 综合得分
            fold_scores(fold) = 0.3 * acc + 0.4 * nmi + 0.2 * f1 + 0.1 * ari;
            valid_folds = valid_folds + 1;
            
        catch ME
            warning('试验 %d, 折 %d 训练失败: %s', trial, fold, ME.message);
            fold_scores(fold) = -inf;
            fold_metrics.ACC(fold) = 0;
            fold_metrics.NMI(fold) = 0;
            fold_metrics.F1(fold) = 0;
            fold_metrics.ARI(fold) = 0;
        end
    end
    
    % 计算平均指标
    if valid_folds > 0
        valid_scores = fold_scores(fold_scores > -inf);
        valid_acc = fold_metrics.ACC(fold_metrics.ACC > 0);
        valid_nmi = fold_metrics.NMI(fold_metrics.NMI > 0);
        valid_f1 = fold_metrics.F1(fold_metrics.F1 > 0);
        valid_ari = fold_metrics.ARI(fold_metrics.ARI > 0);
        
        mean_score = mean(valid_scores);
        mean_acc = mean(valid_acc);
        mean_nmi = mean(valid_nmi);
        mean_f1 = mean(valid_f1);
        mean_ari = mean(valid_ari);
        
        % 存储结果
        result = struct();
        result.params = current_params;
        result.metrics = struct('ACC', mean_acc, 'NMI', mean_nmi, 'F1', mean_f1, 'ARI', mean_ari, 'Score', mean_score);
        result.fold_scores = fold_scores;
        result.fold_metrics = fold_metrics;
        result.valid_folds = valid_folds;
        
        all_results(trial) = result;
        
        % 更新最优参数
        if mean_score > best_score && ~isnan(mean_score) && ~isinf(mean_score)
            best_score = mean_score;
            best_params = current_params;
            best_params.score = mean_score;
            
            if verbose
                fprintf('  -> 新的最优得分: %.4f\n', mean_score);
            end
        end
        
        if verbose
            fprintf('  平均得分: %.4f (ACC: %.3f, NMI: %.3f, F1: %.3f, ARI: %.3f)\n', ...
                mean_score, mean_acc, mean_nmi, mean_f1, mean_ari);
        end
    else
        warning('试验 %d: 所有折都失败', trial);
    end
end

% 搜索历史
search_history = struct();
search_history.method = 'bayes';
search_history.max_trials = max_trials;
search_history.evaluated_trials = length(all_results);
search_history.best_score = best_score;
search_history.parameter_ranges = param_ranges;
search_history.use_gaussian_process = use_gp;

if verbose
    fprintf('\n=== 贝叶斯优化完成 ===\n');
    fprintf('完成了 %d 次试验\n', length(all_results));
    fprintf('最优得分: %.4f\n', best_score);
end

end

function params = sample_random_params(param_ranges)
% 随机采样参数
params = struct();
params.lambda = param_ranges.lambda(1) + ...
    (param_ranges.lambda(2) - param_ranges.lambda(1)) * rand();
params.tau = param_ranges.tau(1) + ...
    (param_ranges.tau(2) - param_ranges.tau(1)) * rand();
params.k = round(param_ranges.k(1) + ...
    (param_ranges.k(2) - param_ranges.k(1)) * rand());
end

function params = sample_bayes_params(param_ranges, all_results, verbose)
% 基于高斯过程的贝叶斯采样
try
    % 提取已有的参数和得分
    n_results = length(all_results);
    if n_results < 3
        % 样本太少，使用随机采样
        params = sample_random_params(param_ranges);
        return;
    end
    
    % 构建训练数据
    X_train = zeros(n_results, 3);
    y_train = zeros(n_results, 1);
    
    for i = 1:n_results
        X_train(i, 1) = all_results(i).params.lambda;
        X_train(i, 2) = all_results(i).params.tau;
        X_train(i, 3) = all_results(i).params.k;
        y_train(i) = all_results(i).metrics.Score;
    end
    
    % 标准化参数
    X_mean = mean(X_train, 1);
    X_std = std(X_train, 1);
    X_std(X_std == 0) = 1;  % 避免除零
    X_norm = (X_train - X_mean) ./ X_std;
    
    % 训练高斯过程
    gp_model = fitrgp(X_norm, y_train, 'KernelFunction', 'ardsquaredexponential', ...
                      'Standardize', false, 'Verbose', 0);
    
    % 生成候选点
    n_candidates = 100;
    candidates = zeros(n_candidates, 3);
    
    for i = 1:n_candidates
        candidates(i, :) = sample_random_params(param_ranges);
    end
    
    % 标准化候选点
    candidates_norm = (candidates - X_mean) ./ X_std;
    
    % 预测均值和方差
    [y_pred, y_std] = predict(gp_model, candidates_norm);
    
    % 使用UCB（Upper Confidence Bound）策略选择下一个点
    ucb = y_pred + 0.1 * y_std;  % 探索-利用平衡参数
    
    [~, best_idx] = max(ucb);
    best_candidate = candidates(best_idx, :);
    
    % 构建参数结构
    params = struct();
    params.lambda = best_candidate(1);
    params.tau = best_candidate(2);
    params.k = best_candidate(3);
    
    if verbose
        fprintf('    贝叶斯采样: lambda=%.4f, tau=%.4f, k=%d\n', ...
            params.lambda, params.tau, params.k);
    end
    
catch ME
    % 如果贝叶斯优化失败，回退到随机采样
    if verbose
        fprintf('    贝叶斯采样失败: %s，使用随机采样\n', ME.message);
    end
    params = sample_random_params(param_ranges);
end
end
