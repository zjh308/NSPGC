function purity = compute_purity(true_labels, pred_labels)
    clusters = unique(pred_labels);
    total_correct = 0;
    for i = 1:length(clusters)
        idx = find(pred_labels == clusters(i));
        true_in_cluster = true_labels(idx);
        mode_label = mode(true_in_cluster);
        total_correct = total_correct + sum(true_in_cluster == mode_label);
    end
    purity = total_correct / length(true_labels);
end
