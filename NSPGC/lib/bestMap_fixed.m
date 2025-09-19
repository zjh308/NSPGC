function [newL2] = bestMap_fixed(L1,L2)
%bestmap_fixed: permute labels of L2 to match L1 as good as possible
%   [newL2] = bestMap_fixed(L1,L2);
%   修复版本：能正确处理任意标签值，避免索引超出范围错误
%
%   version 2.1 --Fixed version for arbitrary label values
%   version 2.0 --May/2007
%   version 1.0 --November/2003
%
%   Written by Deng Cai (dengcai AT gmail.com)
%   Fixed by Assistant for arbitrary label handling

%===========    

L1 = L1(:);
L2 = L2(:);
if size(L1) ~= size(L2)
    error('size(L1) must == size(L2)');
end

Label1 = unique(L1);
nClass1 = length(Label1);
Label2 = unique(L2);
nClass2 = length(Label2);

% 构建方形混淆矩阵（Hungarian算法要求）
nClass = max(nClass1, nClass2);
G = zeros(nClass, nClass);
for i=1:nClass1
    for j=1:nClass2
        G(i,j) = length(find(L1 == Label1(i) & L2 == Label2(j)));
    end
end

% 使用匈牙利算法找到最优匹配
[c,t] = hungarian(-G);

% 重新映射标签
newL2 = zeros(size(L2));
for i=1:nClass2
    % 安全地访问Label1，避免索引超出范围
    if c(i) <= nClass1
        newL2(L2 == Label2(i)) = Label1(c(i));
    else
        % 如果索引超出范围，使用第一个标签
        newL2(L2 == Label2(i)) = Label1(1);
        warning('Label mapping index out of range, using first label');
    end
end

end