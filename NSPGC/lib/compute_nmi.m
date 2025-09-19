function [A nmi avgent] = compute_nmi (T, H)
    % 添加输入参数检查
    if nargin < 2
        error('compute_nmi requires two input arguments: T and H');
    end
    
    if ~isequal(size(T), size(H))
        error('T and H must have the same size. T: %d×%d, H: %d×%d', ...
            size(T,1), size(T,2), size(H,1), size(H,2));
    end
    
    N = length(T);
    
    % 使用稳定的方式获取唯一值
    try
        classes = unique(T(:));  % 确保是一维数组
        clusters = unique(H(:)); % 确保是一维数组
    catch ME
        error('计算unique值时出错: %s', ME.message);
    end
    
    num_class = length(classes);
    num_clust = length(clusters);
    
    % 检查是否有空的类或簇
    if num_class == 0 || num_clust == 0
        error('发现空的类或簇: num_class=%d, num_clust=%d', num_class, num_clust);
    end

    %% 初始化所有变量
    try
        D = zeros(num_class, 1);  % 每个类别的点数
        B = zeros(num_clust, 1);  % 每个聚类的点数
        A = zeros(num_clust, num_class);
        miarr = zeros(num_clust, num_class);  % 初始化miarr矩阵
    catch ME
        error('初始化矩阵时出错: %s. num_class=%d, num_clust=%d', ME.message, num_class, num_clust);
    end
    
    %%compute number of points in each class
    for j=1:num_class
        index_class = (T(:)==classes(j));
        D(j) = sum(index_class);
    end      
    
    %%mutual information
    mi = 0;
    avgent = 0;
    for i=1:num_clust
        %number of points in cluster 'i'
        index_clust = (H(:)==clusters(i));
        B(i) = sum(index_clust);
        for j=1:num_class
            index_class = (T(:)==classes(j));
            %%compute number of points in class 'j' that end up in cluster 'i'
            A(i,j) = sum(index_class.*index_clust);
            if (A(i,j) ~= 0)
                miarr(i,j) = A(i,j)/N * log2 (N*A(i,j)/(B(i)*D(j)));
                %%average entropy calculation                
                avgent = avgent - (B(i)/N) * (A(i,j)/B(i)) * log2 (A(i,j)/B(i));
            else
                miarr(i,j) = 0;
            end
            mi = mi + miarr(i,j);
        end        
    end
    
    %%class entropy
    class_ent = 0;
    for i=1:num_class
        class_ent = class_ent + D(i)/N * log2(N/D(i));
    end
    
    %%clustering entropy
    clust_ent = 0;
    for i=1:num_clust
        clust_ent = clust_ent + B(i)/N * log2(N/B(i));
    end
        
    %%normalized mutual information
    % 添加保护防止除零错误
    if (clust_ent + class_ent) == 0
        nmi = 0;
    else
        nmi = 2*mi / (clust_ent + class_ent);
    end
end