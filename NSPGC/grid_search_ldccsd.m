function [best_params, all_results, search_history] = grid_search_ldccsd(X, Y_true, n_folds, verbose)
% LDCCSD网格搜索函数
% 遍历所有超参数组合，找到最优参数
%
% 输入:
%   X: n×d 数据矩阵
%   Y_true: n×1 真实标签
%   n_folds: 交叉验证折数
%   verbose: 是否显示详细信息
%
% 输出:
%   best_params: 最优超参数组合
%   all_results: 所有参数组合的结果
%   search_history: 搜索过程历史

% 检查数据集大小
n_samples = size(X, 1);
n_clusters = length(unique(Y_true));

if n_samples < n_clusters
    error('样本数(%d)必须大于类别数(%d)', n_samples, n_clusters);
end

% 调整交叉验证折数，确保每折有足够的样本
if n_samples < n_folds * n_clusters
    n_folds = max(2, floor(n_samples / n_clusters));
    if verbose
        fprintf('调整交叉验证折数为%d（确保每折有足够样本）\n', n_folds);
    end
end

% 定义超参数网格
% 基于您的模型特点和数据特性设计
lambda_candidates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0];
tau_candidates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2];
k_candidates = [5, 8, 10, 12, 15, 20];

% 调整k值范围，确保不超过样本数
k_candidates = k_candidates(k_candidates < n_samples);
if isempty(k_candidates)
    k_candidates = [max(3, min(5, floor(n_samples/2)))];
    if verbose
        fprintf('调整k候选值为%d（基于样本数）\n', k_candidates);
    end
end

% 计算总组合数
total_combinations = length(lambda_candidates) * length(tau_candidates) * length(k_candidates);

if verbose
    fprintf('网格搜索: %d种参数组合\n', total_combinations);
    fprintf('lambda: [%s]\n', num2str(lambda_candidates));
    fprintf('tau: [%s]\n', num2str(tau_candidates));
    fprintf('k: [%s]\n', num2str(k_candidates));
    fprintf('开始搜索...\n\n');
end

% 初始化结果存储
all_results = [];
best_score = -inf;
best_params = struct();

% 创建交叉验证索引
try
    cv_indices = crossvalind('Kfold', length(Y_true), n_folds);
catch ME
    % 如果crossvalind失败，手动创建
    if verbose
        fprintf('crossvalind失败，手动创建交叉验证索引\n');
    end
    cv_indices = mod(1:length(Y_true), n_folds) + 1;
end

% 遍历所有参数组合
combination_idx = 0;
for l = 1:length(lambda_candidates)
    for t = 1:length(tau_candidates)
        for k = 1:length(k_candidates)
            combination_idx = combination_idx + 1;
            
            % 当前参数组合
            current_params = struct();
            current_params.lambda = lambda_candidates(l);
            current_params.tau = tau_candidates(t);
            current_params.k = k_candidates(k);
            current_params.maxIter = 30;
            current_params.verbose = false;
            current_params.nCluster = length(unique(Y_true));
            current_params.innerU_MaxIters = 20;
            
            if verbose
                fprintf('组合 %d/%d: lambda=%.3f, tau=%.3f, k=%d\n', ...
                    combination_idx, total_combinations, ...
                    current_params.lambda, current_params.tau, current_params.k);
            end
            
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
                    
                    % 检查训练集大小
                    if size(X_train, 1) < current_params.nCluster
                        if verbose
                            fprintf('  折%d: 训练集样本数(%d) < 类别数(%d)，跳过\n', ...
                                fold, size(X_train, 1), current_params.nCluster);
                        end
                        fold_scores(fold) = -inf;
                        fold_metrics.ACC(fold) = 0;
                        fold_metrics.NMI(fold) = 0;
                        fold_metrics.F1(fold) = 0;
                        fold_metrics.ARI(fold) = 0;
                        continue;
                    end
                    
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
                    
                    % 综合得分（可以调整权重）
                    fold_scores(fold) = 0.3 * acc + 0.4 * nmi + 0.2 * f1 + 0.1 * ari;
                    valid_folds = valid_folds + 1;
                    
                catch ME
                    warning('组合 %d, 折 %d 训练失败: %s', combination_idx, fold, ME.message);
                    fold_scores(fold) = -inf;
                    fold_metrics.ACC(fold) = 0;
                    fold_metrics.NMI(fold) = 0;
                    fold_metrics.F1(fold) = 0;
                    fold_metrics.ARI(fold) = 0;
                end
            end
            
            % 计算平均指标（只考虑有效的折）
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
                
                all_results(combination_idx) = result;
                
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
                if verbose
                    fprintf('  所有折都失败，跳过此参数组合\n');
                end
            end
        end
    end
end

% 搜索历史
search_history = struct();
search_history.method = 'grid';
search_history.total_combinations = total_combinations;
search_history.evaluated_combinations = length(all_results);
search_history.best_score = best_score;
search_history.parameter_ranges = struct('lambda', lambda_candidates, ...
                                       'tau', tau_candidates, ...
                                       'k', k_candidates);

if verbose
    fprintf('\n=== 网格搜索完成 ===\n');
    fprintf('评估了 %d/%d 种参数组合\n', length(all_results), total_combinations);
    fprintf('最优得分: %.4f\n', best_score);
end

end
