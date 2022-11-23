% 数据
clc;clear;close all; 
data_path = 'D:/Dataset/KoNViD_1k';
meta_data = readtable(fullfile(data_path, "KONVID_1K_metadata.csv"));
flicker_ids = meta_data.flickr_id;
mos = meta_data.mos;
excel = './new_mos2.xlsx';

data_length = size(mos, 1);
k = 6;
d = 10;
Threshold = 2; %给定阈值

q_corvage_pred = zeros(data_length, 1);
q_tem_pred = zeros(data_length, 1);
q_spa_pred = zeros(data_length, 1);
q_fus_pred = zeros(data_length, 1);
q_fus_pred2 = zeros(data_length, 1);

douglas_v1 = zeros(data_length, 1);
douglas_lgn = zeros(data_length, 1);
douglas_full = zeros(data_length, 1);
load(['D:/SourceCode/Features/KoNViD/konvid1k_niqe_feats.mat']);

tStartTotal = tic;
for v = 1 : data_length
    tStart = tic;
    load(['D:/SourceCode/Features/KoNViD/LGN_Features/LGN_TIP32/', num2str(flicker_ids(v)), '.mp4.mat']);
    load(['D:/SourceCode/Features/KoNViD/V1_Features/V1_Gabor0.2/', num2str(flicker_ids(v)), '.mat']);
    features_norm22 = feats_mat{v};
    lgn_features = LGN_features_level6;
    lgn_features(isnan(lgn_features) | isinf(lgn_features)) = 0;
    [~,lgn_score] = pca(lgn_features);
    lgn_features = lgn_score(:, 1:d);

    v1_features = V1_features;
    v1_features(isnan(v1_features) | isinf(v1_features)) = 0;
    [~,v1_score] = pca(v1_features);
    v1_features = v1_score(:, 1:d);

    douglas_feature = lgn_features;%(20:120,:);
    num_frames = size(douglas_feature, 1);

    % temporal quality estimation


    %dst_lgn = compute_curvature_simp(lgn_features, Threshold);
    %dst_v1 = compute_curvature_simp(v1_features, Threshold);
    %douglas_lgn(v)=size(dst_lgn, 1);
    %douglas_v1(v)=size(dst_v1, 1);
    %douglas_full(v)=num_frames;

    curv_lgn_dst = compute_curvature(lgn_features);
    curv_v1_dst = compute_curvature(v1_features);

    q_curvage = log(mean(curv_v1_dst));%log(mean(dst_v1)) + log(mean(dst_lgn));
    q_temporal = log(mean(curv_lgn_dst));%log(mean(curv_v1_dst)) + log(mean(curv_lgn_dst));
    q_corvage_pred(v) = q_curvage;
    q_tem_pred(v) = q_temporal;
    q_spa_pred(v) = mean(features_norm22);
    q_fus_pred(v) = q_spa_pred(v)+q_corvage_pred(v);%q_spa_pred(v)+
    q_fus_pred2(v) = q_spa_pred(v)+q_tem_pred(v);%q_spa_pred(v)+%+q_corvage_pred(v);

    ts = toc(tStart);
    fprintf('Video %d, overall %f seconds elapsed...\n', v, ts);
end

% figure(10)
% scatter(1:1:data_length, douglas, '.');
% ylabel('ground-truth');
% xlabel('predicted');
% 
% tstotal = toc(tStartTotal);
% fprintf('Video 1 - %d, overall %f seconds elapsed...\n', data_length, tstotal);
% xlswrite(excel, douglas(1:1200).','C1');
fprintf('Simplification: %f\n', sum(douglas_lgn)/sum(douglas_full));

modelfun = @(b, x)(b(2) + (b(1) - b(2)) ./ (1 + exp(-1 .* (x - b(3)) ./ abs(b(4)))));
beta0 = 2 * rand(4, 1) - 1;
beta = nlinfit(q_fus_pred, mos, modelfun, beta0);
q_fit = modelfun(beta, q_fus_pred);

rating_metrics(1, mos, q_fit);

modelfun2 = @(c, x)(c(2) + (c(1) - c(2)) ./ (1 + exp(-1 .* (x - c(3)) ./ abs(c(4)))));
cbeta0 = 2 * rand(4, 1) - 1;
cbeta = nlinfit(q_fus_pred2, mos, modelfun2, cbeta0);
q_fit = modelfun2(cbeta, q_fus_pred2);

rating_metrics(2, mos, q_fit);



function curvatures = compute_curvature(lgn_features)
    % input: 一个视频由lgn模型提取特征后得到的矩阵，nxm，n为帧数，m为特征维数
    % output: 计算后的curvatures的向量
    len = size(lgn_features, 1);
    curvatures = zeros(len - 2, 1);

    for fn = 2 : len - 1  % fn: frame number
        prev = lgn_features(fn - 1, :);
        current = lgn_features(fn , :);
        next = lgn_features(fn + 1, :);
        numerator = dot((next - current), (current - prev));
        denominator = norm(next - current) * norm(current - prev)+0.000001;
        cos_alpha = acos(numerator / denominator)*(norm(abs(next-prev)));
        curvatures(fn - 1) = cos_alpha;
    end
end

function curvatures = compute_curvature_simp(lgn_features, Threshold)
    lgn_features_norm = zscore(lgn_features);
    [r,c] = size(lgn_features_norm);
    
    clear A;
    A(1,1) = 1;
    A(2,1) = r;
    A(1,2:c+1) = lgn_features_norm(1,1:c);
    A(2,2:c+1) = lgn_features_norm(r,1:c);
    A(1,c+2) = 0;
    A(2,c+2) = 0;

    [A] = DouglasPeucker(lgn_features_norm,A,Threshold,1,r,1); % 递归
    [num,~] = size(A);
    %keypoint(A(1:num))=1;%
    A = sortrows(A,1);
    curvatures = zeros(num - 2, 1);

    for t = 2: 1 : num - 1
        prev = lgn_features(A(t-1),:);
        current = lgn_features(A(t),:);
        next = lgn_features(A(t+1),:);
        numerator = dot((next - current),(current - prev));
        size(numerator);
        denominator = norm(abs(next-current))*norm(abs(current-prev))+0.000001;
        curvatures(t-1) = acos(numerator/denominator)*(sqrt(norm(abs(next-prev))));
    end
end

function D=getDistanceFromPointsToLine(points,p1,p2)
    D = 0;
    [r,~] = size(points);
    for t = 1: 1 : r
        point = points(t);
        norm_dis=sqrt(sum((p1-point).^2));
        numerator = dot((p1-point),(p1-p2));
        size(numerator);
        denominator = norm(abs(p1-point))*norm(abs(p1-p2))+0.000001;
        theta = acos(numerator/denominator);
        D = D + norm_dis*sin(theta);
    end
end

%     lgn_features = LGN_features_level6;
%     lgn_features(isnan(lgn_features) | isinf(lgn_features)) = 0;
%     num_frames = size(lgn_features, 1);
% 
%     % 对LGN提取的特征使用PCA降维
%     %[~,score] = pca(lgn_features);
%     %lgn_features = score(:, 1:d);
% 
%     % temporal quality estimation
%     dt = zeros(num_frames - 2, 1);
% 
%     for t = 2: 1 : num_frames - 1
%         prev = lgn_features(t-1,:);
%         current = lgn_features(t,:);
%         next = lgn_features(t+1,:);
%         numerator = dot((next - current),(current - prev));
%         size(numerator);
%         denominator = norm(abs(next-current))*norm(abs(current-prev));
%         if denominator==0
%             dt(t) = 0;
%         else
%             dt(t) = acos(numerator/denominator);
%         end
%     end
%     dt(isnan(dt) | isinf(dt)) = 0;
% 
%     q_corvage(v) = abs(mean(dt));
