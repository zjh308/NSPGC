function analyze_hyperparameter_results(all_results, search_history, varargin)
% 分析超参数搜索结果并生成可视化图表
%
% 输入:
%   all_results: 所有参数组合的结果
%   search_history: 搜索过程历史
%   varargin: 可选参数

% 解析可选参数
p = inputParser;
addParameter(p, 'save_plots', true, @islogical);
addParameter(p, 'plot_dir', './hyperparameter_plots', @ischar);
addParameter(p, 'plot_format', 'png', @ischar);
parse(p, varargin{:});

save_plots = p.Results.save_plots;
plot_dir = p.Results.plot_dir;
plot_format = p.Results.plot_format;

% 创建图表目录
if save_plots && ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end

% 检查结果是否为空
if isempty(all_results)
    warning('没有结果可分析');
    return;
end

fprintf('=== 超参数搜索结果分析 ===\n');
fprintf('搜索方法: %s\n', search_history.method);
fprintf('评估组合数: %d\n', length(all_results));

% 提取数据
n_results = length(all_results);
lambda_vals = zeros(n_results, 1);
tau_vals = zeros(n_results, 1);
k_vals = zeros(n_results, 1);
scores = zeros(n_results, 1);
acc_scores = zeros(n_results, 1);
nmi_scores = zeros(n_results, 1);
f1_scores = zeros(n_results, 1);
ari_scores = zeros(n_results, 1);

for i = 1:n_results
    lambda_vals(i) = all_results(i).params.lambda;
    tau_vals(i) = all_results(i).params.tau;
    k_vals(i) = all_results(i).params.k;
    scores(i) = all_results(i).metrics.Score;
    acc_scores(i) = all_results(i).metrics.ACC;
    nmi_scores(i) = all_results(i).metrics.NMI;
    f1_scores(i) = all_results(i).metrics.F1;
    ari_scores(i) = all_results(i).metrics.ARI;
end

% 1. 参数分布分析
fprintf('\n--- 参数分布分析 ---\n');
fprintf('Lambda: 范围[%.4f, %.4f], 均值=%.4f, 标准差=%.4f\n', ...
    min(lambda_vals), max(lambda_vals), mean(lambda_vals), std(lambda_vals));
fprintf('Tau: 范围[%.4f, %.4f], 均值=%.4f, 标准差=%.4f\n', ...
    min(tau_vals), max(tau_vals), mean(tau_vals), std(tau_vals));
fprintf('K: 范围[%d, %d], 均值=%.1f, 标准差=%.1f\n', ...
    min(k_vals), max(k_vals), mean(k_vals), std(k_vals));

% 2. 得分分布分析
fprintf('\n--- 得分分布分析 ---\n');
fprintf('综合得分: 范围[%.4f, %.4f], 均值=%.4f, 标准差=%.4f\n', ...
    min(scores), max(scores), mean(scores), std(scores));
fprintf('ACC: 范围[%.4f, %.4f], 均值=%.4f\n', ...
    min(acc_scores), max(acc_scores), mean(acc_scores));
fprintf('NMI: 范围[%.4f, %.4f], 均值=%.4f\n', ...
    min(nmi_scores), max(nmi_scores), mean(nmi_scores));
fprintf('F1: 范围[%.4f, %.4f], 均值=%.4f\n', ...
    min(f1_scores), max(f1_scores), mean(f1_scores));
fprintf('ARI: 范围[%.4f, %.4f], 均值=%.4f\n', ...
    min(ari_scores), max(ari_scores), mean(ari_scores));

% 3. 最优参数组合
[best_score, best_idx] = max(scores);
best_lambda = lambda_vals(best_idx);
best_tau = tau_vals(best_idx);
best_k = k_vals(best_idx);

fprintf('\n--- 最优参数组合 ---\n');
fprintf('最优得分: %.4f\n', best_score);
fprintf('最优Lambda: %.4f\n', best_lambda);
fprintf('最优Tau: %.4f\n', best_tau);
fprintf('最优K: %d\n', best_k);
fprintf('对应指标: ACC=%.4f, NMI=%.4f, F1=%.4f, ARI=%.4f\n', ...
    acc_scores(best_idx), nmi_scores(best_idx), f1_scores(best_idx), ari_scores(best_idx));

% 4. 参数敏感性分析
fprintf('\n--- 参数敏感性分析 ---\n');
analyze_parameter_sensitivity(lambda_vals, tau_vals, k_vals, scores);

% 5. 生成可视化图表
generate_visualization_plots(all_results, search_history, save_plots, plot_dir, plot_format);

fprintf('\n=== 分析完成 ===\n');
if save_plots
    fprintf('图表已保存到: %s\n', plot_dir);
end

end

function analyze_parameter_sensitivity(lambda_vals, tau_vals, k_vals, scores)
% 分析参数敏感性
% 计算每个参数与得分的相关性

% Lambda敏感性
lambda_corr = corrcoef(lambda_vals, scores);
fprintf('Lambda与得分的相关性: %.4f\n', lambda_corr(1,2));

% Tau敏感性
tau_corr = corrcoef(tau_vals, scores);
fprintf('Tau与得分的相关性: %.4f\n', tau_corr(1,2));

% K敏感性
k_corr = corrcoef(k_vals, scores);
fprintf('K与得分的相关性: %.4f\n', k_corr(1,2));

% 参数重要性排序
correlations = [abs(lambda_corr(1,2)), abs(tau_corr(1,2)), abs(k_corr(1,2))];
param_names = {'Lambda', 'Tau', 'K'};
[sorted_corr, sorted_idx] = sort(correlations, 'descend');

fprintf('参数重要性排序:\n');
for i = 1:length(sorted_corr)
    fprintf('  %s: %.4f\n', param_names{sorted_idx(i)}, sorted_corr(i));
end

end

function generate_visualization_plots(all_results, search_history, save_plots, plot_dir, plot_format)
% 生成可视化图表

% 提取数据
n_results = length(all_results);
lambda_vals = zeros(n_results, 1);
tau_vals = zeros(n_results, 1);
k_vals = zeros(n_results, 1);
scores = zeros(n_results, 1);

for i = 1:n_results
    lambda_vals(i) = all_results(i).params.lambda;
    tau_vals(i) = all_results(i).params.tau;
    k_vals(i) = all_results(i).params.k;
    scores(i) = all_results(i).metrics.Score;
end

% 1. 得分分布直方图
figure('Position', [100, 100, 800, 600]);
subplot(2,3,1);
histogram(scores, 20, 'FaceColor', 'skyblue', 'EdgeColor', 'black');
xlabel('综合得分');
ylabel('频次');
title('得分分布');
grid on;

% 2. Lambda vs 得分
subplot(2,3,2);
scatter(lambda_vals, scores, 50, scores, 'filled');
colorbar;
xlabel('Lambda');
ylabel('综合得分');
title('Lambda vs 得分');
grid on;

% 3. Tau vs 得分
subplot(2,3,3);
scatter(tau_vals, scores, 50, scores, 'filled');
colorbar;
xlabel('Tau');
ylabel('综合得分');
title('Tau vs 得分');
grid on;

% 4. K vs 得分
subplot(2,3,4);
scatter(k_vals, scores, 50, scores, 'filled');
colorbar;
xlabel('K');
ylabel('综合得分');
title('K vs 得分');
grid on;

% 5. 3D散点图：Lambda-Tau-得分
subplot(2,3,5);
scatter3(lambda_vals, tau_vals, scores, 50, scores, 'filled');
xlabel('Lambda');
ylabel('Tau');
zlabel('综合得分');
title('Lambda-Tau-得分 3D视图');
colorbar;
grid on;

% 6. 热力图：参数组合得分
subplot(2,3,6);
% 创建参数网格
lambda_unique = unique(lambda_vals);
tau_unique = unique(tau_vals);
score_matrix = zeros(length(tau_unique), length(lambda_unique));

for i = 1:length(tau_unique)
    for j = 1:length(lambda_unique)
        idx = (tau_vals == tau_unique(i)) & (lambda_vals == lambda_unique(j));
        if any(idx)
            score_matrix(i, j) = mean(scores(idx));
        end
    end
end

imagesc(score_matrix);
colorbar;
xlabel('Lambda索引');
ylabel('Tau索引');
title('参数组合得分热力图');
set(gca, 'XTick', 1:length(lambda_unique), 'XTickLabel', arrayfun(@(x) sprintf('%.3f', x), lambda_unique, 'UniformOutput', false));
set(gca, 'YTick', 1:length(tau_unique), 'YTickLabel', arrayfun(@(x) sprintf('%.3f', x), tau_unique, 'UniformOutput', false));

sgtitle(sprintf('main超参数搜索结果分析 (%s方法)', search_history.method), 'FontSize', 16);

% 保存图表
if save_plots
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    plot_file = fullfile(plot_dir, sprintf('hyperparameter_analysis_%s_%s.%s', ...
        search_history.method, timestamp, plot_format));
    saveas(gcf, plot_file);
    fprintf('主图表已保存: %s\n', plot_file);
end

% 7. 收敛曲线（如果有多轮搜索）
if isfield(search_history, 'evaluated_trials') && search_history.evaluated_trials > 1
    figure('Position', [200, 200, 600, 400]);
    
    % 按试验顺序排序
    [~, sort_idx] = sort(scores, 'descend');
    sorted_scores = scores(sort_idx);
    
    plot(1:length(sorted_scores), sorted_scores, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
    xlabel('试验次数');
    ylabel('最优得分');
    title(sprintf('超参数搜索收敛曲线 (%s方法)', search_history.method));
    grid on;
    
    % 标记最优值
    hold on;
    [max_score, max_idx] = max(sorted_scores);
    plot(max_idx, max_score, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'red');
    text(max_idx, max_score, sprintf(' 最优: %.4f', max_score), 'FontSize', 12);
    hold off;
    
    if save_plots
        conv_file = fullfile(plot_dir, sprintf('convergence_curve_%s_%s.%s', ...
            search_history.method, timestamp, plot_format));
        saveas(gcf, conv_file);
        fprintf('收敛曲线已保存: %s\n', conv_file);
    end
end

end
