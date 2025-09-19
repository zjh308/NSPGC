function [indx] = my_kmeans(U, numclass)
    % 归一化处理
    U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1, size(U, 2));
    
    % 调用 kmeans 函数
    indx = kmeans(U_normalized, numclass, 'MaxIter', 100, 'Replicates', 50, 'EmptyAction', 'drop');
    
    % 确保输出是列向量
    indx = indx(:);
end