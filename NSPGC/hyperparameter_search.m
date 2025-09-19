function [best_params, all_results, search_history] = hyperparameter_search(X, Y_true, search_method, varargin)
% LDCCSD超参数搜索主函数
% 支持三种搜索策略：网格搜索、随机搜索、贝叶斯优化
%
% 输入:
%   X: n×d 数据矩阵
%   Y_true: n×1 真实标签
%   search_method: 搜索方法 ('grid', 'random', 'bayes')
%   varargin: 可选参数
%
% 输出:
%   best_params: 最优超参数组合
%   all_results: 所有参数组合的结果
%   search_history: 搜索过程历史

% 解析可选参数
p = inputParser;
addParameter(p, 'n_folds', 3, @isnumeric);           % 交叉验证折数
addParameter(p, 'max_trials', 100, @isnumeric);       % 最大试验次数
addParameter(p, 'verbose', true, @islogical);         % 是否显示详细信息
addParameter(p, 'save_results', true, @islogical);    % 是否保存结果
addParameter(p, 'results_dir', './hyperparameter_results', @ischar);
parse(p, varargin{:});

n_folds = p.Results.n_folds;
max_trials = p.Results.max_trials;
verbose = p.Results.verbose;
save_results = p.Results.save_results;
results_dir = p.Results.results_dir;

% 创建结果目录
if save_results && ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% 数据预处理
n_samples = size(X, 1);
n_clusters = length(unique(Y_true));

if verbose
    fprintf('=== LDCCSD超参数搜索开始 ===\n');
    fprintf('数据集: %d个样本, %d个特征, %d个类别\n', size(X,1), size(X,2), n_clusters);
    fprintf('搜索方法: %s\n', search_method);
    fprintf('交叉验证: %d折\n', n_folds);
end

% 根据搜索方法选择策略
switch lower(search_method)
    case 'grid'
        [best_params, all_results, search_history] = grid_search_ldccsd(X, Y_true, n_folds, verbose);
    case 'random'
        [best_params, all_results, search_history] = random_search_ldccsd(X, Y_true, n_folds, max_trials, verbose);
    case 'bayes'
        [best_params, all_results, search_history] = bayes_optimization_ldccsd(X, Y_true, n_folds, max_trials, verbose);
    otherwise
        error('不支持的搜索方法: %s. 支持的方法: grid, random, bayes', search_method);
end

% 最终验证最优参数
if verbose
    fprintf('\n=== 最终验证最优参数 ===\n');
    fprintf('最优参数: lambda=%.4f, tau=%.4f, k=%d\n', ...
        best_params.lambda, best_params.tau, best_params.k);
    fprintf('最优得分: %.4f\n', best_params.score);
end

% 保存结果
if save_results
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    results_file = fullfile(results_dir, sprintf('hyperparameter_search_%s_%s.mat', search_method, timestamp));
    save(results_file, 'best_params', 'all_results', 'search_history', 'X', 'Y_true');
    
    % 保存CSV格式的详细结果
    csv_file = fullfile(results_dir, sprintf('hyperparameter_results_%s_%s.csv', search_method, timestamp));
    save_results_to_csv(all_results, csv_file);
    
    if verbose
        fprintf('结果已保存到: %s\n', results_file);
        fprintf('详细结果已保存到: %s\n', csv_file);
    end
end

end

function save_results_to_csv(all_results, csv_file)
% 将结果保存为CSV格式
if isempty(all_results)
    return;
end

% 提取字段
params_fields = fieldnames(all_results(1).params);
metric_fields = fieldnames(all_results(1).metrics);

% 创建表头
header = [params_fields; metric_fields];
header = header';

% 创建数据矩阵
n_results = length(all_results);
data = zeros(n_results, length(header));

for i = 1:n_results
    row_idx = 1;
    
    % 参数值
    for j = 1:length(params_fields)
        data(i, row_idx) = all_results(i).params.(params_fields{j});
        row_idx = row_idx + 1;
    end
    
    % 指标值
    for j = 1:length(metric_fields)
        data(i, row_idx) = all_results(i).metrics.(metric_fields{j});
        row_idx = row_idx + 1;
    end
end

% 写入CSV文件
fid = fopen(csv_file, 'w');
if fid == -1
    warning('无法创建CSV文件: %s', csv_file);
    return;
end

% 写入表头
fprintf(fid, '%s,', header{1:end-1});
fprintf(fid, '%s\n', header{end});

% 写入数据
for i = 1:n_results
    fprintf(fid, '%.6f,', data(i, 1:end-1));
    fprintf(fid, '%.6f\n', data(i, end));
end

fclose(fid);
end
