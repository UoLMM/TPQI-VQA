%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 数据
data_path = 'D:/Dataset/KoNViD_1k/';
meta_data = readtable(fullfile(data_path, "KONVID_1K_metadata.csv"));
flicker_ids = meta_data.flickr_id;
mos = meta_data.mos;

data_length = size(mos, 1);
curvatures_v1 = zeros(data_length, 1);
curvatures_lgn = zeros(data_length, 1);
curvatures = zeros(data_length, 1);
tStartTotal = tic;

gaborArray = gaborFilterBank(5,8,39,39);  % Generates the Gabor filter bank

for v = 1 : data_length
    tStart = tic;

    video_name = fullfile(data_path, "KoNViD_1k_videos/", ...
        [num2str(flicker_ids(v)), '.mp4']);
    vid = VideoReader(video_name);

    img_features = extract_img_features(vid);
    lgn_features = extract_lgn_features(vid);
    V1_features = extract_v1_features(vid, gaborArray);

    lgn_features(isnan(lgn_features) | isinf(lgn_features)) = 0;

    curvature = compute_curvature(img_features);
    curvatures(v) = mean(curvature);
    curvature_lgn = compute_curvature(lgn_features);
    curvatures_lgn(v) = mean(curvature_lgn);

    [~,score] = pca(V1_features);
    V1_features = score(:, 1:100);
    curvature_v1 = compute_curvature(V1_features);
    curvatures_v1(v) = mean(curvature_v1);

    ts = toc(tStart);
    fprintf('Video %d, curvature: %f, curvature_lgn: %f, curvature_v1: %f ...\n', v, curvatures(v), curvatures_lgn(v), curvatures_v1(v));

end
tstotal = toc(tStartTotal);
fprintf('Video 1 - %d, overall %f seconds elapsed...\n', data_length, tstotal);

function img_features = extract_img_features(vid)
    % video_name is the file path to the video
    %vid = VideoReader(video_name);
    num_frames = vid.NumFrames;
    img_features = [];
    for f = 1:num_frames
        frame = read(vid, f);
        frame_gray_norm = normalize_gray_image(rgb2gray(frame));
        img_features = [img_features; frame_gray_norm(:).'];
    end
end

function v1_features = extract_v1_features(vid, gaborArray)
    % video_name is the file path to the video
    %vid = VideoReader(video_name);
    num_frames = vid.NumFrames;
    v1_features = [];
    for f = 1:num_frames
        frame = read(vid, f);
        frame_gray_norm = gaborFeatures(imresize(frame, 0.2),gaborArray,4,4);%normalize_gray_image(rgb2gray(frame));
        v1_features = [v1_features; frame_gray_norm(:).'];
    end
end

function lgn_features = extract_lgn_features(vid)
    num_frames = vid.NumFrames;
    lgn_features = [];
    for f = 1:num_frames
        frame = read(vid, f);
        frame_gray = double(rgb2gray(frame));
        [y, ~] = frame_LGN_features(frame_gray);
        y = y{1, 6};
        y = y(:);
        lgn_features = [lgn_features; y.'];
    end
end