%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 数据
% data_path = 'D:/Dataset/KoNViD_1k/';
% meta_data = readtable(fullfile(data_path, "KONVID_1K_metadata.csv"));
data_path = 'D:/Dataset/LIVE-VQC/';
meta_data = readtable(fullfile(data_path, "livevqc_metadata.csv"));
flicker_ids = meta_data.File;
mos = meta_data.mos;

data_length = size(mos, 1);
curvatures_v1 = zeros(data_length, 1);
tStartTotal = tic;

gaborArray = gaborFilterBank(6,8,39,39);  % Generates the Gabor filter bank

time = 0;
for v = 573 : data_length
%     video_name = fullfile(data_path, "KoNViD_1k_videos/", ...
%          [num2str(flicker_ids(v)), '.mp4']);
    video_name = fullfile(data_path, "Video/", meta_data.File{v});
    vid = VideoReader(video_name);
    tStart = tic;

    num_frame = meta_data.nb_frames(v);
    V1_features = extract_v1_features(vid, gaborArray, num_frame);
    [~,score] = pca(V1_features);
    V1_features = score;
    time = time+toc(tStart);

    save(strcat('D:/SourceCode/STEM-main/Temporal_Quality/V1_Gabor1_live540/',meta_data.File{v},'.mat'),'V1_features')

    ts = toc(tStart);
    fprintf('Video %d, overall %f seconds elapsed...\n', v, ts);

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

function v1_features = extract_v1_features(vid, gaborArray, num_frame)
    % video_name is the file path to the video
    %vid = VideoReader(video_name);
    num_frames = vid.NumFrames;
    v1_features = [];
    for f = 1:num_frame
        frame = read(vid, f);
        frame = imresize(frame, [960, 540]);
        frame_gray_norm = gaborFeatures(imresize(frame, 1),gaborArray,4,4);%normalize_gray_image(rgb2gray(frame));
        v1_features = [v1_features; frame_gray_norm(:).'];
    end
end

function lgn_features = extract_lgn_features(vid)
    num_frames = vid.NumFrames;
    lgn_features = [];
    time = 0
    for f = 1:10
        frame = read(vid, f);
        figure; imshow(frame)
        frame_gray = double(rgb2gray(frame));
        frame_gray = imresize(frame_gray, [3840, 2160]);
        tStart = tic;
        [y, ~] = frame_LGN_features(frame_gray);
        time = time+toc(tStart);
        y = y{1, 6};
        y = y(:);
        lgn_features = [lgn_features; y.'];
    end
    fprintf('270P, overall %f seconds elapsed...\n', time/10);
end