function acc = clustering_accuracy(true_labels, pred_labels)
    % 调整标签顺序以最大化匹配
    D = max(max(pred_labels), max(true_labels));
    w = zeros(D);
    for i = 1:length(true_labels)
        w(pred_labels(i), true_labels(i)) = w(pred_labels(i), true_labels(i)) + 1;
    end
    
    % 使用匈牙利算法最大匹配，提供多种备选方案
    try
        % 首先尝试使用hungarian函数
        if exist('hungarian', 'file')
            [~, cost] = hungarian(-w);
        elseif exist('munkres', 'file')
            [~, cost] = munkres(-w);
        else
            % 如果都没有，使用简单的贪心匹配
            warning('未找到hungarian或munkres函数，使用简单匹配');
            cost = simple_matching(w);
        end
    catch ME
        warning('匈牙利算法失败，使用简单匹配: %s', ME.message);
        cost = simple_matching(w);
    end
    
    acc = cost / length(true_labels);
end

function cost = simple_matching(w)
    % 简单的贪心匹配算法作为备选
    [n, m] = size(w);
    cost = 0;
    
    % 贪心选择最大匹配
    for i = 1:min(n, m)
        [max_val, max_idx] = max(w(:));
        if max_val <= 0
            break;
        end
        
        [row, col] = ind2sub(size(w), max_idx);
        cost = cost + max_val;
        
        % 标记已使用的行和列
        w(row, :) = 0;
        w(:, col) = 0;
    end
end
