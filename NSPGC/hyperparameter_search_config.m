function search_config = hyperparameter_search_config()
% 超参数搜索配置文件
% 返回一个包含所有搜索配置的结构体
%
% 输出:
%   search_config: 包含所有配置参数的结构体

% ===== 基本搜索配置 =====
search_config = struct();

% 是否启用超参数搜索
search_config.enable_search = true;        % true: 启用, false: 使用默认参数

% 搜索方法选择
search_config.search_method = 'grid';     % 'grid': 网格搜索, 'random': 随机搜索, 'bayes': 贝叶斯优化

% 交叉验证设置
search_config.n_folds = 3;                % 交叉验证折数 (建议: 小数据集2-3折, 大数据集3-5折)

% 最大试验次数（仅对随机搜索和贝叶斯优化有效）
search_config.max_trials = 50;            % 随机搜索和贝叶斯优化的最大试验次数

% 输出设置
search_config.verbose = true;             % 是否显示详细信息
search_config.save_results = true;        % 是否保存搜索结果

% 结果保存目录
search_config.results_dir = './hyperparameter_results';  % 结果保存目录

% 样本数量限制
search_config.max_samples_for_search = 300; % 超参数搜索时的最大样本数（加快搜索速度）
search_config.max_samples_for_training = 500; % 最终训练时的最大样本数

% ===== 网格搜索配置 =====
search_config.grid_search = struct();
search_config.grid_search.lambda_candidates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0];
search_config.grid_search.tau_candidates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2];
search_config.grid_search.k_candidates = [5, 8, 10, 12, 15, 20];

% ===== 随机搜索配置 =====
search_config.random_search = struct();
search_config.random_search.lambda_range = [0.001, 5.0];    % lambda搜索范围
search_config.random_search.tau_range = [0.0001, 1.0];     % tau搜索范围
search_config.random_search.k_range = [3, 30];              % k搜索范围

% ===== 贝叶斯优化配置 =====
search_config.bayes_search = struct();
search_config.bayes_search.lambda_range = [0.001, 5.0];    % lambda搜索范围
search_config.bayes_search.tau_range = [0.0001, 1.0];     % tau搜索范围
search_config.bayes_search.k_range = [3, 30];              % k搜索范围
search_config.bayes_search.init_trials = 5;                 % 初始化随机试验次数
search_config.bayes_search.exploration_weight = 0.1;        % 探索-利用平衡参数

% ===== 模型参数配置 =====
search_config.model_params = struct();
search_config.model_params.maxIter = 30;                    % 最大迭代次数
search_config.model_params.innerU_MaxIters = 20;            % 内部优化最大迭代次数
search_config.model_params.verbose = false;                 % 模型训练时是否显示详细信息

% ===== 评估指标权重 =====
search_config.evaluation_weights = struct();
search_config.evaluation_weights.ACC = 0.3;                 % 准确率权重
search_config.evaluation_weights.NMI = 0.4;                 % 标准化互信息权重
search_config.evaluation_weights.F1 = 0.2;                  % F1分数权重
search_config.evaluation_weights.ARI = 0.1;                 % 调整兰德指数权重

% ===== 性能优化配置 =====
search_config.performance = struct();
search_config.performance.parallel_search = false;           % 是否启用并行搜索（需要Parallel Computing Toolbox）
search_config.performance.max_workers = 4;                  % 最大并行工作进程数
search_config.performance.timeout_per_trial = 300;          % 每次试验的超时时间（秒）

% ===== 早停配置 =====
search_config.early_stopping = struct();
search_config.early_stopping.enable = false;                % 是否启用早停机制
search_config.early_stopping.patience = 10;                 % 连续无改善的最大次数
search_config.early_stopping.min_improvement = 0.001;       % 最小改善阈值

% ===== 数据预处理配置 =====
search_config.preprocessing = struct();
search_config.preprocessing.standardize_features = true;     % 是否标准化特征
search_config.preprocessing.handle_missing = true;           % 是否处理缺失值
search_config.preprocessing.handle_outliers = false;         % 是否处理异常值
search_config.preprocessing.outlier_threshold = 3;           % 异常值阈值（标准差倍数）

% ===== 结果分析配置 =====
search_config.analysis = struct();
search_config.analysis.generate_plots = true;               % 是否生成可视化图表
search_config.analysis.plot_dir = './hyperparameter_plots'; % 图表保存目录
search_config.analysis.plot_format = 'png';                 % 图表保存格式
search_config.analysis.save_csv = true;                     % 是否保存CSV格式结果
search_config.analysis.save_mat = true;                     % 是否保存MAT格式结果

% ===== 日志配置 =====
search_config.logging = struct();
search_config.logging.log_level = 'info';                   % 日志级别: 'debug', 'info', 'warning', 'error'
search_config.logging.log_file = './hyperparameter_search.log'; % 日志文件路径
search_config.logging.console_output = true;                % 是否在控制台输出日志

end

% ===== 配置验证函数 =====
function validate_config(search_config)
% 验证配置参数的有效性

% 检查搜索方法
valid_methods = {'grid', 'random', 'bayes'};
if ~ismember(search_config.search_method, valid_methods)
    error('无效的搜索方法: %s. 有效方法: %s', search_config.search_method, strjoin(valid_methods, ', '));
end

% 检查交叉验证折数
if search_config.n_folds < 2 || search_config.n_folds > 10
    warning('交叉验证折数 %d 可能不合适，建议范围: 2-10', search_config.n_folds);
end

% 检查样本数量限制
if search_config.max_samples_for_search < 50
    warning('超参数搜索样本数 %d 可能过少，建议至少50个样本', search_config.max_samples_for_search);
end

% 检查参数范围
if search_config.random_search.lambda_range(1) >= search_config.random_search.lambda_range(2)
    error('Lambda范围设置错误: 最小值应小于最大值');
end

if search_config.random_search.tau_range(1) >= search_config.random_search.tau_range(2)
    error('Tau范围设置错误: 最小值应小于最大值');
end

if search_config.random_search.k_range(1) >= search_config.random_search.k_range(2)
    error('K范围设置错误: 最小值应小于最大值');
end

% 检查权重设置
weights = [search_config.evaluation_weights.ACC, ...
           search_config.evaluation_weights.NMI, ...
           search_config.evaluation_weights.F1, ...
           search_config.evaluation_weights.ARI];
       
if abs(sum(weights) - 1.0) > 1e-6
    warning('评估指标权重总和不为1.0，当前总和: %.3f', sum(weights));
end

fprintf('配置验证通过\n');
end
