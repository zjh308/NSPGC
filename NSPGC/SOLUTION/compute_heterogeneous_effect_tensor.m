function hetero_effect = compute_heterogeneous_effect_tensor(v, Y, S)
% 计算异簇推离效果指标：张量优化版本
% 独立函数文件，避免嵌套函数调用问题

hetero_diffs = [];
homo_diffs = [];

n = size(Y,1);
eps_stable = 1e-12;

% 优化版本：只计算稀疏矩阵中的非零元素
[I, J, vals] = find(S);
valid_pairs = I < J;  % 只考虑上三角部分避免重复
I = I(valid_pairs);
J = J(valid_pairs);
vals = vals(valid_pairs);

for idx = 1:length(I)
    i = I(idx);
    j = J(idx);
    
    if vals(idx) > eps_stable
        weight_diff = abs(v(i) - v(j));
        if norm(Y(i,:) - Y(j,:)) < 1e-6  % 同簇
            homo_diffs(end+1) = weight_diff; %#ok<AGROW>
        else  % 异簇
            hetero_diffs(end+1) = weight_diff; %#ok<AGROW>
        end
    end
end

% 使用更稳健的计算方式
if ~isempty(hetero_diffs) && ~isempty(homo_diffs)
    hetero_mean = mean(hetero_diffs);
    homo_mean = mean(homo_diffs);
    
    if homo_mean < 1e-6
        hetero_effect = 1.0;
    else
        hetero_effect = hetero_mean / homo_mean;
        hetero_effect = min(hetero_effect, 100.0);
        hetero_effect = max(hetero_effect, 0.1);
    end
else
    hetero_effect = 1.0;
end

end
