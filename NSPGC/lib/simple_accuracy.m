function acc = simple_accuracy(true_labels, pred_labels)
% 简单的准确率计算函数，用于调试
%
% 输入:
%   true_labels - 真实标签向量
%   pred_labels - 预测标签向量
%
% 输出:
%   acc - 准确率

% 确保输入是列向量
true_labels = true_labels(:);
pred_labels = pred_labels(:);

% 检查维度是否匹配
if length(true_labels) ~= length(pred_labels)
    error('真实标签和预测标签的长度不匹配');
end

% 计算准确率
acc = sum(true_labels == pred_labels) / length(true_labels);

fprintf('简单准确率计算:\n');
fprintf('  样本数: %d\n', length(true_labels));
fprintf('  正确数: %d\n', sum(true_labels == pred_labels));
fprintf('  准确率: %.4f\n', acc);
end