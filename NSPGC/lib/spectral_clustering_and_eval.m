function [acc, nmi, purity, pred_labels] = spectral_clustering_and_eval(G, true_labels, num_clusters)
% G             - 特征矩阵（n × c），由 LCSPL 学到的
% true_labels   - 真实标签（n × 1）
% num_clusters  - 聚类数（即类别数）

    % Step 1: 归一化特征
    G = normalize(G, 2);  % 每行单位范数（样本级）

    % Step 2: K-means 聚类
    rng(1);  % 固定随机种子保证可重复性
    pred_labels = kmeans(G, num_clusters, 'Replicates', 10, 'MaxIter', 500, 'Display', 'off');

    % Step 3: 评估指标
    acc    = clustering_accuracy(true_labels, pred_labels);
    nmi    = compute_nmi(true_labels, pred_labels);
    purity = compute_purity(true_labels, pred_labels);

    % 打印结果
    fprintf('Clustering ACC: %.4f, NMI: %.4f, Purity: %.4f\n', acc, nmi, purity);
end
