function run_hyperparameter_search()
% run_hyperparameter_search: 优化版单数据集超参数搜索与性能评估
% 
% 该脚本用于:
% 1. 加载特定数据集
% 2. 通过网格搜索找到异簇模型的最佳参数组合
% 3. 使用张量优化的算法实现(main_heterogeneous_tensor)提升性能
% 4. 按ACC排序输出Top5结果
% 5. 支持开关控制是否进行超参数搜索
% 6. 支持手动输入参数进行复现
% 
% 优化内容:
% - Y-update: 向量化邻域计算，复杂度从O(n²c)降至O(nk_NN·c)
% - 梯度计算: 完全向量化稀疏矩阵操作
% - 相似度矩阵: 优化距离计算和k-NN搜索

clc; clear; close all;

% ========================================
% 配置参数
% ========================================

% 固定随机种子确保复现性
MASTER_SEED = 2;
rng(MASTER_SEED);

% 超参数搜索开关
ENABLE_SEARCH = true;  % 设置为false可跳过搜索，使用手动参数

% 手动参数配置（当ENABLE_SEARCH=false时使用，用于复现最佳实验结果）
MANUAL_PARAMS = struct(...
    'lambda', 1, ...
    'alpha', 0.05, ...
    'k', 5, ...
    'tau', 1.5 ...
);

% 网格搜索参数范围
HP_GRID = struct(...
    'lambda', [0.5,1.0, 1.5, 2.0], ...
    'alpha', [ 0.01,0.05, 0.1,0.15], ...
    'k', [5, 8, 10], ...
    'tau', [ 1.5, 2.0, 5.0] ...
);

% 模型参数
MODEL_CONFIG = struct(...
    'maxIter', 1000, ...        % 减少迭代次数以提高性能
    'innerU_MaxIters', 200, ... % 减少内层迭代次数
    'verbose', false ...       % 默认关闭详细输出以提高性能
);

% 性能优化配置
PERFORMANCE_CONFIG = struct(...
    'max_samples', 10000, ...   % 大数据集最大样本数
    'use_subset_for_search', true ... % 搜索时使用子集
);

% ========================================
% 初始化
% ========================================

fprintf('=========================================\n');
fprintf('单数据集异簇模型超参数搜索\n');
fprintf('=========================================\n');

fprintf('主随机种子: %d\n', MASTER_SEED);

if ~ENABLE_SEARCH
    fprintf('手动参数配置(用于复现最佳实验结果): lambda=%.2f, alpha=%.3f, k=%d, tau=%.2f\n', ...
        MANUAL_PARAMS.lambda, MANUAL_PARAMS.alpha, MANUAL_PARAMS.k, MANUAL_PARAMS.tau);
end

% 数据路径
DATASET_FOLDER = fullfile(pwd, '..', 'data');
RESULT_FOLDER = fullfile(pwd, 'hyperparameter_results');

% 指定数据集名称
DATASET_NAME = 'zoo.mat';

fprintf('数据集文件夹: %s\n', DATASET_FOLDER);
fprintf('指定数据集: %s\n', DATASET_NAME);
fprintf('结果文件夹: %s\n', RESULT_FOLDER);

% 创建结果文件夹
if ~exist(RESULT_FOLDER, 'dir')
    mkdir(RESULT_FOLDER);
end

% 清理路径，避免函数冲突
restoredefaultpath;

% 加载依赖函数（使用优先级路径）
lib_path = fullfile(pwd, '..', 'lib');
solution_path = fullfile(pwd, '..', 'SOLUTION');

% 优先添加当前项目的SOLUTION路径
if exist(solution_path, 'dir')
    addpath(solution_path, '-begin');  % 添加到路径开头，确保优先级
end

if exist(lib_path, 'dir')
    addpath(lib_path);
end

% ========================================
% 加载指定数据集
% ========================================
data_path = fullfile(DATASET_FOLDER, DATASET_NAME);
if ~exist(data_path, 'file')
    error('指定数据集文件不存在: %s', data_path);
end

fprintf('\n=========================================\n');
fprintf('处理数据集: %s\n', DATASET_NAME);

% 加载数据
clear X Y_true;
load(data_path);

% 显示加载的数据变量信息
vars = whos;
fprintf('  数据文件中的变量:\n');
for v = 1:length(vars)
    var_info = vars(v);
    fprintf('    %s: %d×%d class: %s\n', var_info.name, var_info.size(1), var_info.size(2), var_info.class);
end

% 处理标签（自动识别标签字段）
label_fields = {'Y', 'y', 'gt', 'label', 'gnd', 'target', 'Data', 'II_Ia_Ib_data'};
data_loaded = false;

for f = 1:length(label_fields)
    if strcmp(label_fields{f}, 'Data') && exist('Data', 'var')
        % Data格式：第一列是标签，其他列是特征
        Y_true = Data(:, 1);
        X = Data(:, 2:end);
        data_loaded = true;
        fprintf('  使用Data格式（第一列为标签）\n');
        break;
    elseif strcmp(label_fields{f}, 'II_Ia_Ib_data') && exist('II_Ia_Ib_data', 'var')
        % II_Ia_Ib_data格式：第一列是标签，其他列是特征
        Y_true = II_Ia_Ib_data(:, 1);
        X = II_Ia_Ib_data(:, 2:end);
        data_loaded = true;
        fprintf('  使用II_Ia_Ib_data格式（第一列为标签）\n');
        break;
    elseif exist(label_fields{f}, 'var')
        Y_true = eval(label_fields{f});
        if ~exist('X', 'var')
            % 尝试查找其他可能的特征变量
            feature_vars = {'X', 'data', 'features', 'train_data', 'test_data'};
            for fv = 1:length(feature_vars)
                if exist(feature_vars{fv}, 'var')
                    X = eval(feature_vars{fv});
                    fprintf('  找到特征变量: %s\n', feature_vars{fv});
                    break;
                end
            end
            
            if ~exist('X', 'var')
                warning('找到标签但未找到特征X，跳过此字段');
                continue;
            end
        end
        data_loaded = true;
        fprintf('  使用标签字段: %s\n', label_fields{f});
        break;
    end
end

% 如果仍未加载数据，尝试更通用的方法
if ~data_loaded
    fprintf('  尝试通用数据加载方法...\n');
    % 查找可能的特征和标签变量
    vars = whos;
    numeric_vars = [];
    for v = 1:length(vars)
        var_info = vars(v);
        if strcmp(var_info.class, 'double') && (var_info.size(1) > 1) && (var_info.size(2) > 1)
            numeric_vars(end+1) = var_info.name;
        end
    end
    
    % 如果找到两个以上的数值变量，尝试将最大的作为特征，其他作为标签候选
    if length(numeric_vars) >= 2
        % 按大小排序
        var_sizes = zeros(length(numeric_vars), 1);
        for v = 1:length(numeric_vars)
            var_info = eval(['whos(''', numeric_vars{v}, ''');']);
            var_sizes(v) = var_info(1).size(1) * var_info(1).size(2);
        end
        [sorted_sizes, sorted_idx] = sort(var_sizes, 'descend');
        
        % 假设最大的是特征矩阵
        X = eval(numeric_vars{sorted_idx(1)});
        fprintf('  假设特征变量: %s (%d×%d)\n', numeric_vars{sorted_idx(1)}, size(X,1), size(X,2));
        
        % 查找合适的标签变量
        for v = 2:length(sorted_idx)
            candidate = eval(numeric_vars{sorted_idx(v)});
            % 检查是否可以作为标签（一维向量）
            if (size(candidate,1) == 1 || size(candidate,2) == 1) && ...
               (size(candidate,1) == size(X,1) || size(candidate,2) == size(X,1))
                Y_true = candidate;
                fprintf('  假设标签变量: %s (%d×%d)\n', numeric_vars{sorted_idx(v)}, size(Y_true,1), size(Y_true,2));
                data_loaded = true;
                break;
            end
        end
    end
end

if ~data_loaded
    error('数据集 %s 未找到合适的标签字段', DATASET_NAME);
end

% 数据预处理
Y_true = reshape(Y_true, [], 1);  % 确保列向量
% 检查X和Y_true的维度是否匹配，如果不匹配尝试转置
if size(X, 1) ~= size(Y_true, 1)
    fprintf('  检测到维度不匹配，尝试转置X矩阵...\n');
    fprintf('  原始X维度: %d×%d, Y_true维度: %d×%d\n', size(X, 1), size(X, 2), size(Y_true, 1), size(Y_true, 2));
    
    % 尝试转置X
    if size(X, 2) == size(Y_true, 1)
        X = X';
        fprintf('  转置后X维度: %d×%d\n', size(X, 1), size(X, 2));
    else
        error('无法通过转置对齐维度');
    end
end

if size(X, 1) ~= size(Y_true, 1)
    error('数据集 %s 特征与标签样本数不匹配', DATASET_NAME);
end

% 数据有效性检查
if any(isnan(X(:))) || any(isinf(X(:)))
    warning('数据集 %s 包含NaN或Inf值，进行清理...', DATASET_NAME);
    X(isnan(X) | isinf(X)) = 0;
end

if any(isnan(Y_true)) || any(isinf(Y_true))
    warning('数据集 %s 标签包含NaN或Inf值，进行清理...', DATASET_NAME);
    Y_true(isnan(Y_true) | isinf(Y_true)) = 1;
end

% 标签映射为连续整数
[Y_true, ~] = grp2idx(Y_true);
n_cluster = length(unique(Y_true));
n_sample = size(X, 1);
n_feat = size(X, 2);

% 数据集有效性检查
if n_cluster >= n_sample
    error('数据集 %s: 类别数(%d) >= 样本数(%d)，这通常不正确。', DATASET_NAME, n_cluster, n_sample);
end

if n_cluster < 2
    error('数据集 %s: 类别数(%d) < 2，无法进行聚类。', DATASET_NAME, n_cluster);
end

fprintf('  数据维度: %d样本 × %d特征 × %d类别\n', n_sample, n_feat, n_cluster);

% 数据标准化
X = zscore(X);

% 性能优化：对于大数据集，使用子集进行超参数搜索
X_search = X;
Y_true_search = Y_true;
if n_sample > PERFORMANCE_CONFIG.max_samples && ENABLE_SEARCH && PERFORMANCE_CONFIG.use_subset_for_search
    fprintf('  数据集较大，使用子集进行超参数搜索...\n');
    % 随机采样保持类别分布
    selected_indices = datasample_subset(Y_true, PERFORMANCE_CONFIG.max_samples);
    X_search = X(selected_indices, :);
    Y_true_search = Y_true(selected_indices);
    fprintf('  搜索子集大小: %d样本\n', size(X_search, 1));
end

fprintf('  数据预处理完成\n');

% ========================================
% 超参数搜索或使用手动参数
% ========================================
if ENABLE_SEARCH
    fprintf('  开始网格搜索 (%d个参数组合)\n', ...
        length(HP_GRID.lambda)*length(HP_GRID.alpha)*length(HP_GRID.k)*length(HP_GRID.tau));
    fprintf('  注意: 搜索过程中只显示参数组合，最终将输出前5个最佳结果及其性能指标\n');
    
    % 生成所有参数组合
    hp_combinations = [];
    combo_idx = 1;
    for l_idx = 1:length(HP_GRID.lambda)
        lambda = HP_GRID.lambda(l_idx);
        for a_idx = 1:length(HP_GRID.alpha)
            alpha = HP_GRID.alpha(a_idx);
            for k_idx = 1:length(HP_GRID.k)
                k = HP_GRID.k(k_idx);
                for t_idx = 1:length(HP_GRID.tau)
                    tau = HP_GRID.tau(t_idx);
                    hp_combinations(combo_idx, :) = [lambda, alpha, k, tau];
                    combo_idx = combo_idx + 1;
                end
            end
        end
    end
    
    total_combos = size(hp_combinations, 1);
    fprintf('  总参数组合数: %d\n', total_combos);
    
    % 存储搜索结果
    search_results = repmat(struct(...
        'lambda', 0, ...
        'alpha', 0, ...
        'k', 0, ...
        'tau', 0, ...
        'acc', 0, ...
        'nmi', 0, ...
        'f1', 0, ...
        'ari', 0, ...
        'ri', 0, ...
        'pair_precision', 0, ...
        'pair_recall', 0, ...
        'pair_f1', 0), total_combos, 1);
    
    % 遍历所有参数组合
    for combo_idx = 1:total_combos
        lambda = hp_combinations(combo_idx, 1);
        alpha = hp_combinations(combo_idx, 2);
        k = hp_combinations(combo_idx, 3);
        tau = hp_combinations(combo_idx, 4);
        
        % 显示进度（每10个组合显示一次）
        if mod(combo_idx, 10) == 1 || combo_idx == total_combos
            fprintf('    搜索进度: %d/%d (%.1f%%)\n', combo_idx, total_combos, combo_idx/total_combos*100);
        end
        
        % 固定随机种子确保复现性
        rng(MASTER_SEED);
        
        % 配置参数（搜索时使用与最终评估相同的迭代次数以确保一致性）
        params = struct(...
            'lambda', lambda, ...
            'alpha', alpha, ...
            'k', k, ...
            'tau', tau, ...
            'maxIter', MODEL_CONFIG.maxIter, ...  % 使用与最终评估相同的迭代次数
            'innerU_MaxIters', MODEL_CONFIG.innerU_MaxIters, ...  % 使用与最终评估相同的内层迭代次数
            'nCluster', n_cluster, ...
            'verbose', false ...  % 搜索时关闭详细输出
        );
        
        try
            % 运行张量优化模型（使用完整数据集以确保一致性）
            tic;
            [labels_pred, ~, ~] = main_heterogeneous_tensor(X, params);
            runtime = toc;
            mapped_labels = bestMap_fixed(Y_true, labels_pred);
            acc = sum(Y_true == mapped_labels) / n_sample;
            
            % 计算其他评估指标
            [~, nmi] = compute_nmi(Y_true, labels_pred);
            [f1, precision, recall] = compute_f(Y_true, labels_pred);
            [ari, ri, ~, ~] = RandIndex(Y_true, labels_pred);
            
            % 计算配对指标
            n = length(Y_true);
            same_true = bsxfun(@eq, Y_true, Y_true');
            same_pred = bsxfun(@eq, labels_pred, labels_pred');
            iu = triu(ones(n),1);
            TP = sum(sum(same_true & same_pred & iu));
            TN = sum(sum(~same_true & ~same_pred & iu));
            FP = sum(sum(~same_true & same_pred & iu));
            FN = sum(sum(same_true & ~same_pred & iu));
            pair_precision = TP / (TP + FP + eps);
            pair_recall    = TP / (TP + FN + eps);
            pair_f1        = 2*pair_precision*pair_recall/(pair_precision+pair_recall+eps);
            
            % 在搜索过程中只输出参数组合，不输出指标
            fprintf('    参数组合 %d: lambda=%.2f, alpha=%.3f, k=%d, tau=%.2f ' ,...
                combo_idx, lambda, alpha, k, tau);
        catch ME
            fprintf('    警告: 组合(%d) 失败: %s\n', combo_idx, ME.message);
            acc = 0;
            nmi = 0;
            f1 = 0;
            ari = 0;
            ri = 0;
            pair_precision = 0;
            pair_recall = 0;
            pair_f1 = 0;
        end
        
        % 保存结果
        search_results(combo_idx) = struct(...
            'lambda', lambda, ...
            'alpha', alpha, ...
            'k', k, ...
            'tau', tau, ...
            'acc', acc, ...
            'nmi', nmi, ...
            'f1', f1, ...
            'ari', ari, ...
            'ri', ri, ...
            'pair_precision', pair_precision, ...
            'pair_recall', pair_recall, ...
            'pair_f1', pair_f1 ...
        );
    end
    
    % 按ACC排序
    acc_values = [search_results.acc];
    [sorted_acc, sorted_idx] = sort(acc_values, 'descend');
    
    % 保存当前数据集的结果
    all_results.dataset = DATASET_NAME;
    all_results.top_results = search_results(sorted_idx(1:min(5, length(sorted_idx))));
    
    % 显示Top5结果
    fprintf('\n=========================================\n');
    fprintf('搜索完成！前5个最佳参数组合及其性能指标:\n');
    fprintf('=========================================\n');
    fprintf('  %-6s %-8s %-7s %-4s %-6s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n', ...
        '排名', 'lambda', 'alpha', 'k', 'tau', 'ACC', 'NMI', 'F1', 'ARI', 'RI', 'P-Prec', 'P-F1');
    fprintf('  %s\n', repmat('-', 1, 95));
    for i = 1:min(5, length(sorted_idx))
        r = search_results(sorted_idx(i));
        fprintf('  %-6d %-8.4f %-7.4f %-4d %-6.1f %-8.4f %-8.4f %-8.4f %-8.4f %-8.4f %-8.4f %-8.4f\n', ...
            i, r.lambda, r.alpha, r.k, r.tau, r.acc, r.nmi, r.f1, r.ari, r.ri, r.pair_precision, r.pair_f1);
    end
    fprintf('=========================================\n');
    
    % 保存结果到文件
    result_file = fullfile(RESULT_FOLDER, sprintf('%s_results.mat', DATASET_NAME(1:end-4)));
    save(result_file, 'search_results', 'sorted_idx', 'HP_GRID', 'MASTER_SEED');
    fprintf('  结果已保存到: %s\n', result_file);
    
    % 使用最佳参数进行最终评估
    best_params = search_results(sorted_idx(1));
else
    % 使用手动参数（用于复现最佳实验结果）
    best_params = struct(...
        'lambda', MANUAL_PARAMS.lambda, ...
        'alpha', MANUAL_PARAMS.alpha, ...
        'k', MANUAL_PARAMS.k, ...
        'tau', MANUAL_PARAMS.tau, ...
        'acc', 0, ...
        'nmi', 0, ...
        'f1', 0, ...
        'ari', 0, ...
        'ri', 0, ...
        'pair_precision', 0, ...
        'pair_recall', 0, ...
        'pair_f1', 0 ...
    );
    
    fprintf('  使用手动参数(用于复现最佳实验结果): lambda=%.2f, alpha=%.3f, k=%d, tau=%.2f\n', ...
        best_params.lambda, best_params.alpha, best_params.k, best_params.tau);
end

% ========================================
% 使用最佳参数进行最终评估并保存前5个最优结果
% ========================================
fprintf('  使用最佳参数进行最终评估并保存前5个最优结果...\n');

% 创建数据集特定的结果文件夹
dataset_result_folder = fullfile(RESULT_FOLDER, DATASET_NAME(1:end-4));
if ~exist(dataset_result_folder, 'dir')
    mkdir(dataset_result_folder);
end

% 获取前5个最优参数组合
if ENABLE_SEARCH
    top5_indices = sorted_idx(1:min(5, length(sorted_idx)));
else
    % 如果不进行搜索，只使用手动参数
    top5_indices = 1;
    search_results = repmat(best_params, 1, 1);
end

% 保存前5个最优结果
for i = 1:length(top5_indices)
    if ENABLE_SEARCH
        current_params = search_results(top5_indices(i));
    else
        current_params = best_params;
    end
    
    fprintf('    评估第%d个最优参数组合...\n', i);
    
    % 固定随机种子确保复现性
    rng(MASTER_SEED);
    
    % 配置参数（最终评估时使用完整的迭代次数）
    params = struct(...
        'lambda', current_params.lambda, ...
        'alpha', current_params.alpha, ...
        'k', current_params.k, ...
        'tau', current_params.tau, ...
        'maxIter', MODEL_CONFIG.maxIter, ...
        'innerU_MaxIters', MODEL_CONFIG.innerU_MaxIters, ...
        'nCluster', n_cluster, ...
        'verbose', true ...  % 最终评估时开启详细输出
    );
    
    try
        % 运行张量优化模型
        tic;
        [labels_pred, Y, history] = main_heterogeneous_tensor(X, params);
        final_runtime = toc;
        mapped_labels = bestMap_fixed(Y_true, labels_pred);
        final_acc = sum(Y_true == mapped_labels) / n_sample;
        
        % 计算其他评估指标
        [~, nmi] = compute_nmi(Y_true, labels_pred);
        [f1, precision, recall] = compute_f(Y_true, labels_pred);
        [ari, ri, ~, ~] = RandIndex(Y_true, labels_pred);
        
        % 计算配对指标
        n = length(Y_true);
        same_true = bsxfun(@eq, Y_true, Y_true');
        same_pred = bsxfun(@eq, labels_pred, labels_pred');
        iu = triu(ones(n),1);
        TP = sum(sum(same_true & same_pred & iu));
        TN = sum(sum(~same_true & ~same_pred & iu));
        FP = sum(sum(~same_true & same_pred & iu));
        FN = sum(sum(same_true & ~same_pred & iu));
        pair_precision = TP / (TP + FP + eps);
        pair_recall    = TP / (TP + FN + eps);
        pair_f1        = 2*pair_precision*pair_recall/(pair_precision+pair_recall+eps);
        
        % 输出最终评估的各项指标（包含运行时间）
        fprintf('    第%d个最优参数组合评估结果:\n', i);
        fprintf('  最终评估结果 (使用最优参数): ACC=%.4f, NMI=%.4f, F1=%.4f, ARI=%.4f, 张量运行时间=%.2fs\n', ...
            final_acc, nmi, f1, ari, final_runtime);
        fprintf('      配对精度=%.4f, 配对召回率=%.4f, 配对F1=%.4f\n', ...
            pair_precision, pair_recall, pair_f1);
        fprintf('      张量优化算法运行时间: %.2f秒\n', final_runtime);
        
        % 保存完整结果
        result_struct = struct(...
            'dataset_name', DATASET_NAME(1:end-4), ...
            'params', params, ...
            'random_seed', MASTER_SEED, ...
            'performance', struct(...
                'acc', final_acc, ...
                'nmi', nmi, ...
                'f1', f1, ...
                'ari', ari, ...
                'ri', ri, ...
                'pair_precision', pair_precision, ...
                'pair_recall', pair_recall, ...
                'pair_f1', pair_f1), ...
            'history', history, ...
            'labels_pred', labels_pred, ...
            'Y_true', Y_true, ...
            'Y', Y ...
        );
        
        % 保存结果到文件
        result_file = fullfile(dataset_result_folder, sprintf('top%d_result.mat', i));
        save(result_file, 'result_struct');
        fprintf('    第%d个最优结果已保存到: %s\n', i, result_file);
        
        % 可视化（可选）
        figure;
        subplot(2,3,1);
        plot(history.obj, 'LineWidth',1.5);
        xlabel('Iteration'); ylabel('Objective'); 
        title(sprintf('Objective trajectory - %s', DATASET_NAME(1:end-4)));
        grid on;
        
        subplot(2,3,2);
        plot(history.v_stat, 'LineWidth',1.5);
        xlabel('Iteration'); ylabel('v values');
        legend('min(v)','median(v)','max(v)');
        title(sprintf('Evolution of v distribution - %s', DATASET_NAME(1:end-4)));
        grid on;
        
        subplot(2,3,3);
        bar(sum(Y,1)); 
        title(sprintf('Final cluster sizes - %s', DATASET_NAME(1:end-4))); 
        xlabel('cluster'); ylabel('#samples');
        grid on;
        
        subplot(2,3,4);
        if isfield(history, 'hetero_effect')
            plot(history.hetero_effect, 'LineWidth',1.5);
            xlabel('Iteration'); ylabel('Hetero Effect');
            title(sprintf('Heterogeneous Effect - %s', DATASET_NAME(1:end-4)));
            grid on;
        else
            text(0.5, 0.5, 'No hetero effect data', 'HorizontalAlignment', 'center', 'FontSize', 12);
            title(sprintf('Heterogeneous Effect - %s', DATASET_NAME(1:end-4)));
        end
        
        subplot(2,3,5);
        [C, ~] = confusionmat(Y_true, labels_pred);
        imagesc(C);
        colorbar;
        title('Confusion Matrix');
        xlabel('Predicted Labels'); ylabel('True Labels');
        grid on;
        
        subplot(2,3,6);
        metric_names = {'ACC', 'NMI', 'F1', 'ARI'};
        metric_values = [final_acc, nmi, f1, ari];
        bar(metric_values, 'FaceColor', [0.2, 0.6, 0.8]);
        set(gca, 'XTickLabel', metric_names);
        ylabel('Value');
        title('Performance Metrics Summary');
        grid on;
        % 添加数值标签
        for j = 1:length(metric_values)
            text(j, metric_values(j)+0.02, sprintf('%.3f', metric_values(j)), ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        end
        
        % 保存图表
        fig_file = fullfile(dataset_result_folder, sprintf('top%d_visualization.png', i));
        saveas(gcf, fig_file);
        fprintf('    第%d个可视化图表已保存到: %s\n', i, fig_file);
        close(gcf);
        
    catch ME
        warning('数据集 %s 第%d个最优参数评估失败: %s\n', DATASET_NAME, i, ME.message);
    end
end

% ========================================
% 生成汇总报告
% ========================================
fprintf('\n=========================================\n');
fprintf('汇总报告\n');
fprintf('=========================================\n');

summary_file = fullfile(RESULT_FOLDER, 'summary_report.txt');
fid = fopen(summary_file, 'w');
if fid == -1
    error('无法创建汇总报告文件');
end

fprintf(fid, '单数据集异簇模型超参数搜索汇总报告\n');
fprintf(fid, '=========================================\n');
fprintf(fid, '主随机种子: %d\n', MASTER_SEED);
fprintf(fid, '数据集: %s\n', DATASET_NAME);
 
if ~ENABLE_SEARCH
    fprintf(fid, '手动参数配置(用于复现最佳实验结果): lambda=%.2f, alpha=%.3f, k=%d, tau=%.2f\n', ...
MANUAL_PARAMS.lambda, MANUAL_PARAMS.alpha, MANUAL_PARAMS.k, MANUAL_PARAMS.tau);
end
fprintf(fid, '生成时间: %s\n\n', datestr(now));

fprintf('\n数据集: %s\n', DATASET_NAME);
fprintf(fid, '\n数据集: %s\n', DATASET_NAME);

% 显示Top5结果（如果进行了搜索）
if ENABLE_SEARCH && isfield(all_results, 'top_results')
    top_results = all_results.top_results;
    
    fprintf('Top5参数组合:\n');
    fprintf('%-6s %-8s %-7s %-4s %-6s %-8s %-8s %-8s %-8s\n', '排名', 'lambda', 'alpha', 'k', 'tau', 'ACC', 'NMI', 'F1', 'ARI');
    fprintf('%s\n', repmat('-', 1, 75));
    fprintf(fid, 'Top5参数组合:\n');
    fprintf(fid, '%-6s %-8s %-7s %-4s %-6s %-8s %-8s %-8s %-8s\n', '排名', 'lambda', 'alpha', 'k', 'tau', 'ACC', 'NMI', 'F1', 'ARI');
    fprintf(fid, '%s\n', repmat('-', 1, 75));
    
    for j = 1:length(top_results)
        r = top_results(j);
        fprintf('%-6d %-8.4f %-7.4f %-4d %-6.1f %-8.4f %-8.4f %-8.4f %-8.4f\n', ...
            j, r.lambda, r.alpha, r.k, r.tau, r.acc, r.nmi, r.f1, r.ari);
        fprintf(fid, '%-6d %-8.4f %-7.4f %-4d %-6.1f %-8.4f %-8.4f %-8.4f %-8.4f\n', ...
            j, r.lambda, r.alpha, r.k, r.tau, r.acc, r.nmi, r.f1, r.ari);
    end
end

fclose(fid);
fprintf('\n汇总报告已保存到: %s\n', summary_file);

% 清理路径
if exist(lib_path, 'dir')
    rmpath(lib_path);
end
if exist(solution_path, 'dir')
    rmpath(genpath(solution_path));
end

fprintf('\n=========================================\n');
fprintf('数据集处理完成！\n');
fprintf('=========================================\n');

end

% 子函数：数据子采样以保持类别分布
function selected_indices = datasample_subset(labels, max_samples)
    unique_labels = unique(labels);
    n_classes = length(unique_labels);
    samples_per_class = floor(max_samples / n_classes);
    
    selected_indices = [];
    for i = 1:n_classes
        class_indices = find(labels == unique_labels(i));
        n_class_samples = min(length(class_indices), samples_per_class);
        if n_class_samples > 0
            selected_class_indices = datasample(class_indices, n_class_samples, 'Replace', false);
            selected_indices = [selected_indices; selected_class_indices];
        end
    end
    
    % 如果还有剩余容量，随机补充一些样本
    if length(selected_indices) < max_samples
        remaining_indices = setdiff(1:length(labels), selected_indices);
        n_remaining = min(length(remaining_indices), max_samples - length(selected_indices));
        if n_remaining > 0
            additional_indices = datasample(remaining_indices, n_remaining, 'Replace', false);
            selected_indices = [selected_indices; additional_indices];
        end
    end
    
    % 随机打乱顺序
    selected_indices = selected_indices(randperm(length(selected_indices)));
end