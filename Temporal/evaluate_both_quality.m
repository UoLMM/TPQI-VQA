clear, clc
% 数据
data_path = 'D:/Dataset/KoNViD_1k';
meta_data = readtable(fullfile(data_path, "KONVID_1K_metadata.csv"));
flicker_ids = meta_data.flickr_id;
mos = meta_data.mos;

data_length = size(mos, 1);

q_corvage = zeros(data_length, 1);
q_spa_pred = zeros(data_length, 1);
q_fus_pred = zeros(data_length, 1);
q_fus_curvage = zeros(data_length, 1);

load('./features/tip_original_q_temporal.mat')
load('./features/tip_original_curvage.mat')

excel = './new_mos.xlsx';

tStartTotal = tic;
for v = 1 : data_length
    tStart = tic;
    load(['D:/SourceCode/STEM-main/Temporal_Quality/NIQE_TIP/', num2str(flicker_ids(v)), '.mp4_niqe.mat']);

    q_spa_pred(v) = mean(Frame_NIQE);
    %q_fus_pred(v) = mapminmax(q_corvage(v), 0, 1) *mapminmax(q_spa_pred(v), 0, 1);
    q_fus_pred(v) = q_tem_pred(v) .* q_spa_pred(v);
    q_fus_curvage(v) = q_corvage(v) .* q_spa_pred(v);
    ts = toc(tStart);
    %fprintf('Video %d, overall %f seconds elapsed...\n', v, ts);
end
%q_corvage
%q_tem_pred

%线性预测的计算
norm_q_fus = (q_fus_pred-min(q_fus_pred))./(max(q_fus_pred)-min(q_fus_pred));
q_fus_pred_mos = (1-norm_q_fus).*(max(mos)-min(mos)) + min(mos);

norm_mos = (mos-min(mos))./(max(mos)-min(mos));
cycle_mos = (1-norm_mos).*(max(q_fus_pred)-min(q_fus_pred)) + min(q_fus_pred);

q_tem_pred_mos = cycle_mos ./ q_spa_pred;

norm_q_tem_pred_mos = (q_tem_pred_mos-min(q_tem_pred_mos))./(max(q_tem_pred_mos)-min(q_tem_pred_mos));
cycle_q_tem_pred_mos = (1-norm_q_tem_pred_mos).*(max(mos)-min(mos)) + min(mos);

norm_q_tem_pred = (q_tem_pred-min(q_tem_pred))./(max(q_tem_pred)-min(q_tem_pred));
cycle_q_tem_pred = (1-norm_q_tem_pred).*(max(mos)-min(mos)) + min(mos);

%曲率的计算
norm_q_curvage = (q_fus_curvage-min(q_fus_curvage))./(max(q_fus_curvage)-min(q_fus_curvage));
norm_q_curvage_mos = (1-norm_q_curvage).*(max(mos)-min(mos)) + min(mos);

norm_mos = (mos-min(mos))./(max(mos)-min(mos));
cycle_curvage_mos = (1-norm_mos).*(max(q_fus_curvage)-min(q_fus_curvage)) + min(q_fus_curvage);

q_tem_pred_curvage_mos = cycle_curvage_mos ./ q_spa_pred;

norm_q_tem_pred_curvage_mos = (q_tem_pred_curvage_mos-min(q_tem_pred_curvage_mos))./(max(q_tem_pred_curvage_mos)-min(q_tem_pred_curvage_mos));
cycle_q_tem_pred_curvage_mos = (1-norm_q_tem_pred_curvage_mos).*(max(mos)-min(mos)) + min(mos);

norm_q_corvage = (q_corvage-min(q_corvage))./(max(q_corvage)-min(q_corvage));
cycle_q_corvage_pred = (1-norm_q_corvage).*(max(mos)-min(mos)) + min(mos);

%xlswrite(excel, flicker_ids(1:1200).','C1');
%xlswrite(excel, cycle_q_tem_pred_mos(1:1200).','D1');


rating_metrics(1, mos, q_fus_pred_mos);

rating_metrics(2, cycle_q_tem_pred_mos, cycle_q_tem_pred);
% modelfun = @(b, x)(b(2) + (b(1) - b(2)) ./ (1 + exp(-1 .* (x - b(3)) ./ abs(b(4)))));
% beta0 = 2 * rand(4, 1) - 1;
% beta = nlinfit(q_fus_pred, mos, modelfun, beta0);
% q_fit = modelfun(beta, q_fus_pred);
% %q_fit(q_fit<2.4) = 2.5
% 
% 
% %cycle_modelfun = @(b, x1, x2)(b(3)+abs(b(4)).*log((x1 - b(2))./(b(1)-x1))-x2)
% cycle_q_temporal = cycle_logistic_fun(beta, mos, q_spa_pred);
% 
% modelfun2 = @(c, x)(c(2) + (c(1) - c(2)) ./ (1 + exp(-1 .* (x - c(3)) ./ abs(c(4)))));
% cbeta0 = 2 * rand(4, 1) - 1;
% cbeta = nlinfit(cycle_q_temporal, mos, modelfun2, cbeta0);
% cycle_q_fit = modelfun2(cbeta, cycle_q_temporal);
% 
% 
% modelfun3 = @(d, x)(d(2) + (d(1) - d(2)) ./ (1 + exp(-1 .* (x - d(3)) ./ abs(d(4)))));
% dbeta0 = 2 * rand(4, 1) - 1;
% dbeta = nlinfit(q_tem_pred, cycle_q_fit, modelfun3, dbeta0);
% cycle_q_fit_pred = modelfun3(dbeta, q_tem_pred);
% 
% rating_metrics(cycle_q_fit_pred, cycle_q_fit);

