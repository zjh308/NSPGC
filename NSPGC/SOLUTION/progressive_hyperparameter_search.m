function result = progressive_hyperparameter_search(X, Y_true, config)
% Progressive hyperparameter search: coarse-to-fine with live logging and resumable state
% 支持异簇推离参数alpha的超参数搜索
% Inputs:
%   X (nxd), Y_true (nx1), config struct fields:
%     - results_dir (char)
%     - dataset_name (char)
%     - seeds_coarse (1xS)
%     - seeds_fine (1xS)
%     - enable_balance (logical) optional
%     - enable_heterogeneous (logical) optional - 是否启用异簇推离模型搜索
%     - verbose (logical)
% Outputs:
%   result struct: best_params, leaderboard (table-like struct array), logs_path

if ~isfield(config,'results_dir'), config.results_dir = './hyperparameter_results'; end
if ~exist(config.results_dir,'dir'), mkdir(config.results_dir); end
if ~isfield(config,'dataset_name'), config.dataset_name = 'dataset'; end
if ~isfield(config,'verbose'), config.verbose = true; end
if ~isfield(config,'enable_heterogeneous'), config.enable_heterogeneous = true; end  % 默认启用异簇推离搜索

% enforce labels as column and consecutive is optional, metrics are label-invariant via bestMap_fixed
if size(Y_true,1)==1, Y_true = Y_true'; end
n = size(X,1);

% Stage A: warmup to determine tau_min that avoids v->0
% 根据经验，tau至少从1.5开始搜索
tau_grid_all = [1.5, 2.0, 3.0, 5.0, 8.0, 10.0]; % 根据经验设置的tau范围
tau_min = tau_grid_all(1);
tau_max = tau_grid_all(end);

warm_seeds = 1;   % cheap warmup
n_use_warm = min( max(256, ceil(n/5)), n );
idx_warm = 1:n_use_warm;  % deterministic small subset

tau_ok = [];
for ti = 1:numel(tau_grid_all)
    tau_try = tau_grid_all(ti);
    ok = local_quick_try(X(idx_warm,:), Y_true(idx_warm), struct('tau',tau_try,'verbose',false));
    if ok, tau_ok(end+1) = tau_try; end %#ok<AGROW>
end
if ~isempty(tau_ok)
    tau_min = min(tau_ok);
end

% Stage B: coarse search on subset/short iters
n_use = min( max(468, ceil(n/3)), n );
idx_coarse = 1:n_use;

% 根据经验设置更精细的参数范围
lambda_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]; % 更大更精细的lambda范围
k_grid = [3, 5, 7, 8, 10, 12];  % k尽量3到10搜索
% 根据memory中的经验，alpha在[0.003, 0.007]范围内效果最佳
alpha_grid = [0.0, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03];  % 包含0.0（原始模型）和异簇推离参数
tau_grid = tau_ok; if isempty(tau_grid), tau_grid = tau_grid_all; end

seeds_coarse = [1 2 3];
if isfield(config,'seeds_coarse'), seeds_coarse = config.seeds_coarse; end

leaderboard = [];
log_path = fullfile(config.results_dir, sprintf('progressive_log_%s.mat', config.dataset_name));
if exist(log_path,'file')
    S = load(log_path);
    if isfield(S,'leaderboard'), leaderboard = S.leaderboard; end
end

combos = {};
for a = 1:numel(tau_grid)
  for b = 1:numel(lambda_grid)
    for c = 1:numel(k_grid)
      if config.enable_heterogeneous
        for d = 1:numel(alpha_grid)
          combos{end+1,1} = [tau_grid(a), lambda_grid(b), k_grid(c), alpha_grid(d)]; %#ok<AGROW>
        end
      else
        combos{end+1,1} = [tau_grid(a), lambda_grid(b), k_grid(c), 0.0]; %#ok<AGROW>
      end
    end
  end
end

for ci = 1:numel(combos)
    tau = combos{ci}(1); lambda = combos{ci}(2); k = combos{ci}(3); alpha = combos{ci}(4);
    % skip if already evaluated
    if local_exists_entry(leaderboard, tau, lambda, k, alpha), continue; end
    [score_mean, score_std, acc_mean, ari_mean, f1_mean] = local_eval_combo(X(idx_coarse,:), Y_true(idx_coarse), tau, lambda, k, alpha, seeds_coarse, 12, 10); %#ok<ASGLU>
    entry = struct('tau',tau,'lambda',lambda,'k',k,'alpha',alpha, ...
        'NMI_mean',score_mean,'NMI_std',score_std,'ACC_mean',acc_mean,'ARI_mean',ari_mean,'F1_mean',f1_mean);
    leaderboard = [leaderboard; entry]; %#ok<AGROW>
    % persist after each combo
    save(log_path,'leaderboard','-v7');
    if config.verbose
        fprintf('[progressive] coarse %d/%d: tau=%.4g lambda=%.4g k=%d alpha=%.4g | NMI=%.3f±%.3f\n', ci, numel(combos), tau, lambda, k, alpha, score_mean, score_std);
        local_print_topk(leaderboard, 10);
    end
end

% Stage C: refine around top-M
M = min(10, numel(leaderboard));
[~,idx_sort] = sort([leaderboard.NMI_mean], 'descend');
topM = leaderboard(idx_sort(1:M));
refined = leaderboard;

for m = 1:numel(topM)
    base = topM(m);
    tau_ref = unique(sort([base.tau/2, base.tau/sqrt(2), base.tau, base.tau*sqrt(2), base.tau*2]));
    lambda_ref = unique(sort([base.lambda/2, base.lambda/sqrt(2), base.lambda, base.lambda*sqrt(2), base.lambda*2]));
    k_ref = unique([max(2,base.k-2) base.k-1 base.k base.k+1 base.k+2]);
    % 对alpha参数也进行精细化搜索
    if isfield(base, 'alpha') && base.alpha > 0
        alpha_ref = unique(sort([base.alpha/2, base.alpha/sqrt(2), base.alpha, base.alpha*sqrt(2), base.alpha*2]));
        alpha_ref = alpha_ref(alpha_ref >= 0);  % 确保非负
    else
        alpha_ref = [0.0, 0.003, 0.005];  % 如果原来是0，尝试异簇参数
    end
    for a = 1:numel(tau_ref)
      for b = 1:numel(lambda_ref)
        for c = 1:numel(k_ref)
          for d = 1:numel(alpha_ref)
            tau = tau_ref(a); lambda = lambda_ref(b); k = k_ref(c); alpha = alpha_ref(d);
            if local_exists_entry(refined, tau, lambda, k, alpha), continue; end
            [score_mean, score_std, acc_mean, ari_mean, f1_mean] = local_eval_combo(X(idx_coarse,:), Y_true(idx_coarse), tau, lambda, k, alpha, seeds_coarse, 12, 10);
            entry = struct('tau',tau,'lambda',lambda,'k',k,'alpha',alpha, ...
                'NMI_mean',score_mean,'NMI_std',score_std,'ACC_mean',acc_mean,'ARI_mean',ari_mean,'F1_mean',f1_mean);
            refined = [refined; entry]; %#ok<AGROW>
            save(log_path,'leaderboard','refined','-v7');
            if config.verbose
                fprintf('[progressive] refine: tau=%.4g lambda=%.4g k=%d alpha=%.4g | NMI=%.3f±%.3f\n', tau, lambda, k, alpha, score_mean, score_std);
                local_print_topk(refined, 10);
            end
          end
        end
      end
    end
end

% Stage D: full-data final on Top-N
[~,idx_sort2] = sort([refined.NMI_mean], 'descend');
leader_final = refined(idx_sort2);
N = min(3, numel(leader_final));
seeds_fine = [1 2 3 4 5]; if isfield(config,'seeds_fine'), seeds_fine = config.seeds_fine; end

best = leader_final(1);
best_params = struct('lambda', best.lambda, 'tau', best.tau, 'k', best.k, ...
    'maxIter', 70, 'verbose', true, 'nCluster', numel(unique(Y_true)), 'innerU_MaxIters', 20);
% 添加alpha参数
if isfield(best, 'alpha')
    best_params.alpha = best.alpha;
else
    best_params.alpha = 0.0;  % 默认为原始模型
end

finals = [];
for i = 1:N
    cand = leader_final(i);
    alpha_final = 0.0;
    if isfield(cand, 'alpha')
        alpha_final = cand.alpha;
    end
    [nm, ns, am, ar, fm] = local_eval_combo(X, Y_true, cand.tau, cand.lambda, cand.k, alpha_final, seeds_fine, 70, 20);
    finals = [finals; struct('tau',cand.tau,'lambda',cand.lambda,'k',cand.k,'alpha',alpha_final,'NMI_mean',nm,'NMI_std',ns,'ACC_mean',am,'ARI_mean',ar,'F1_mean',fm)]; %#ok<AGROW>
    save(log_path,'leaderboard','refined','finals','-v7');
    if config.verbose
        fprintf('[progressive] final: tau=%.4g lambda=%.4g k=%d alpha=%.4g | NMI=%.3f±%.3f\n', cand.tau, cand.lambda, cand.k, alpha_final, nm, ns);
    end
end

% pick best of finals
if ~isempty(finals)
    [~,ix] = sort([finals.NMI_mean],'descend');
    finals = finals(ix);
    best_params.lambda = finals(1).lambda; 
    best_params.tau = finals(1).tau; 
    best_params.k = finals(1).k;
    if isfield(finals(1), 'alpha')
        best_params.alpha = finals(1).alpha;
    end
end

result = struct('best_params', best_params, 'leaderboard', leader_final, 'logs_path', log_path);

end

% ===== Helpers =====
function exists = local_exists_entry(L, tau, lambda, k, alpha)
% 检查是否已经评估过该参数组合（包含alpha）
exists = false;
if isempty(L), return; end
for i=1:numel(L)
    % 兼容旧格式（没有alpha字段）
    l_alpha = 0.0;
    if isfield(L(i), 'alpha')
        l_alpha = L(i).alpha;
    end
    if abs(L(i).tau - tau) < 1e-12 && abs(L(i).lambda - lambda) < 1e-12 && L(i).k == k && abs(l_alpha - alpha) < 1e-12
        exists = true; return;
    end
end
end

function [nmi_mean, nmi_std, acc_mean, ari_mean, f1_mean] = local_eval_combo(X, Y_true, tau, lambda, k, alpha, seeds, maxIter, innerU)
% 评估参数组合（包含alpha）
nmi_all = zeros(numel(seeds),1); acc_all = nmi_all; ari_all = nmi_all; f1_all = nmi_all;
params = struct; 
params.lambda = lambda; 
params.tau = tau; 
params.k = k; 
params.maxIter = maxIter; 
params.verbose = false; 
params.nCluster = numel(unique(Y_true)); 
params.innerU_MaxIters = innerU;
params.alpha = alpha;  % 添加alpha参数

for si = 1:numel(seeds)
    rng(seeds(si));
    try
        % 根据alpha值决定使用哪个模型
        if alpha > 0
            [labels_pred, ~, ~] = main_heterogeneous(X, params);  % 使用异簇推离模型
        else
            [labels_pred, ~, ~] = main(X, params);  % 使用原始模型
        end
        [~, nmi] = compute_nmi(Y_true, labels_pred);
        nmi_all(si) = nmi;
        try
            labels_pred_mapped = bestMap_fixed(Y_true, labels_pred);
            acc_all(si) = length(find(Y_true == labels_pred_mapped))/length(Y_true);
        catch
            acc_all(si) = NaN;
        end
        try
            [AR,~,~,~] = RandIndex(Y_true, labels_pred);
            ari_all(si) = AR;
        catch
            ari_all(si) = NaN;
        end
        try
            [f1,~,~] = compute_f(Y_true, labels_pred);
            f1_all(si) = f1;
        catch
            f1_all(si) = NaN;
        end
    catch
        nmi_all(si) = NaN; acc_all(si) = NaN; ari_all(si) = NaN; f1_all(si) = NaN;
    end
end
nmi_mean = nanmean(nmi_all); nmi_std = nanstd(nmi_all);
acc_mean = nanmean(acc_all); ari_mean = nanmean(ari_all); f1_mean = nanmean(f1_all);
end

function ok = local_quick_try(X, Y_true, opt)
% quick run with tiny iters to validate tau
params = struct; params.lambda = 0.05; params.tau = opt.tau; params.k = 10; params.maxIter = 3; params.verbose = false; params.nCluster = numel(unique(Y_true)); params.innerU_MaxIters = 5;
try
    [~, ~, hist] = main(X, params);
    ok = ~isempty(hist) && all(isfinite(hist.tau(1:min(3,numel(hist.tau)))));
catch
    ok = false;
end
end

function local_print_topk(L, K)
% 打印排行榜（包含alpha参数）
if isempty(L)
    fprintf("[progressive] leaderboard 空\n");
    return;
end
[~, idx] = sort([L.NMI_mean], 'descend');
idx = idx(1:min(K, numel(L)));
fprintf('[progressive] Top-%d (by NMI_mean):\n', numel(idx));
for t = 1:numel(idx)
    e = L(idx(t));
    % 兼容旧格式（没有alpha字段）
    e_alpha = 0.0;
    if isfield(e, 'alpha')
        e_alpha = e.alpha;
    end
    fprintf('  #%d tau=%.4g lambda=%.4g k=%d alpha=%.4g | NMI=%.3f±%.3f ACC=%.3f ARI=%.3f F1=%.3f\n', ...
        t, e.tau, e.lambda, e.k, e_alpha, e.NMI_mean, e.NMI_std, e.ACC_mean, e.ARI_mean, e.F1_mean);
end
end


