function generate_convergence_plot(history, dataset_name, save_path)
% 生成符合ICASSP会议标准的收敛性图表
% 用于展示算法的理论收敛性和实际性能

if nargin < 3
    save_path = '';
end

% 创建收敛性分析图
figure('Position', [100, 100, 1200, 800]);

% === 子图1: 目标函数收敛曲线 ===
subplot(2,3,1);
iterations = 1:length(history.obj);
% 过滤掉零值，保持真实的目标函数轨迹
valid_obj = history.obj(history.obj ~= 0);
valid_iterations = 1:length(valid_obj);

plot(valid_iterations, valid_obj, 'b-', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Iteration');
ylabel('Objective Value');
title('Objective Function Convergence');
grid on;
% 标注收敛点
if length(valid_obj) < length(history.obj)  % 如果提前收敛
    plot(length(valid_obj), valid_obj(end), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    text(length(valid_obj), valid_obj(end), sprintf('  Converged at iter %d', length(valid_obj)), ...
        'FontSize', 10, 'Color', 'red');
end

% === 子图2: 目标函数变化率 ===
subplot(2,3,2);
if length(valid_obj) > 1
    obj_changes = abs(diff(valid_obj)) ./ (abs(valid_obj(1:end-1)) + 1e-10);
    semilogy(2:length(valid_obj), obj_changes, 'g-', 'LineWidth', 2);
    hold on;
    % 收敛阈值线
    yline(1e-5, 'r--', 'LineWidth', 2, 'Label', 'Convergence Threshold (1e-5)');
    xlabel('Iteration');
    ylabel('Relative Objective Change');
    title('Convergence Rate Analysis');
    grid on;
    legend('|Δobj|/|obj|', 'Threshold', 'Location', 'best');
end

% === 子图3: SPL调度参数τ ===
subplot(2,3,3);
% 过滤掉零值的τ
valid_tau = history.tau(1:length(valid_obj));
plot(valid_iterations, valid_tau, 'm-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('SPL Age Parameter τ');
title('Self-Paced Learning Schedule');
grid on;

% === 子图4: v向量统计 ===
subplot(2,3,4);
% 过滤掉零值的v统计
valid_v_stat = history.v_stat(1:length(valid_obj), :);
plot(valid_iterations, valid_v_stat(:,1), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Min(v)');
hold on;
plot(valid_iterations, valid_v_stat(:,2), 'g-', 'LineWidth', 2, 'DisplayName', 'Median(v)');
plot(valid_iterations, valid_v_stat(:,3), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Max(v)');
xlabel('Iteration');
ylabel('Sample Weight Statistics');
title('SPL Weight Evolution');
legend('Location', 'best');
grid on;

% === 子图5: 异簇效果 ===
subplot(2,3,5);
% 过滤掉零值的异簇效果
valid_hetero = history.hetero_effect(1:length(valid_obj));
plot(valid_iterations, valid_hetero, 'c-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Heterogeneous Effect');
title('Inter-cluster Regularization Effect');
grid on;

% === 子图6: 收敛性总结 ===
subplot(2,3,6);
% 创建收敛性指标表格
convergence_metrics = {
    'Total Iterations', sprintf('%d', length(valid_obj));
    'Final Objective', sprintf('%.3e', valid_obj(end));
    'Final τ', sprintf('%.3f', valid_tau(end));
    'Final Hetero Effect', sprintf('%.4f', valid_hetero(end));
};

if length(valid_obj) > 1
    final_change = abs(valid_obj(end) - valid_obj(end-1)) / (abs(valid_obj(end-1)) + 1e-10);
    convergence_metrics{end+1,1} = 'Final Δobj';
    convergence_metrics{end,2} = sprintf('%.2e', final_change);
    
    % 判断是否提前收敛（实际迭代数少于总迭代数）
    if length(valid_obj) < length(history.obj) || final_change < 1e-5
        convergence_status = 'CONVERGED';
        status_color = 'green';
    else
        convergence_status = 'MAX ITER REACHED';
        status_color = 'red';
    end
else
    convergence_status = 'SINGLE ITERATION';
    status_color = 'orange';
end

convergence_metrics{end+1,1} = 'Status';
convergence_metrics{end,2} = convergence_status;

% 显示表格
axis off;
text(0.1, 0.9, 'Convergence Summary', 'FontSize', 14, 'FontWeight', 'bold');
for i = 1:size(convergence_metrics, 1)
    y_pos = 0.8 - (i-1) * 0.1;
    if strcmp(convergence_metrics{i,1}, 'Status')
        text(0.1, y_pos, sprintf('%s: %s', convergence_metrics{i,1}, convergence_metrics{i,2}), ...
            'FontSize', 12, 'Color', status_color, 'FontWeight', 'bold');
    else
        text(0.1, y_pos, sprintf('%s: %s', convergence_metrics{i,1}, convergence_metrics{i,2}), ...
            'FontSize', 11);
    end
end

% === 整体标题 ===
sgtitle(sprintf('SPGC Convergence Analysis - %s Dataset', dataset_name), ...
    'FontSize', 16, 'FontWeight', 'bold');

% === 保存图表 ===
if ~isempty(save_path)
    % 确保目录存在
    [save_dir, ~, ~] = fileparts(save_path);
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    
    % 保存为高质量PNG和EPS格式（适合论文）
    print(save_path, '-dpng', '-r300');
    [path_no_ext, ~, ~] = fileparts(save_path);
    print([path_no_ext, '.eps'], '-depsc', '-r300');
    
    fprintf('收敛性分析图表已保存到: %s\n', save_path);
    fprintf('EPS格式已保存到: %s.eps\n', path_no_ext);
end

end
