%%
% Compute features for a set of video files from datasets
% 
% close all; 
% clear;
% warning('off','all');

%%
% parameters
algo_name = 'niqe'; % algorithm name, eg, 'V-BLIINDS'
data_name = 'live';  % dataset name, eg, 'KONVID_1K'
load('./modelparameters.mat');

% data_path = 'D:/Dataset/LIVE-VQC/';
% meta_data = readtable(fullfile(data_path, "livevqc_metadata.csv"));
% load(['D:/SourceCode/Features/LiveVQC/livevqc_niqe_feats.mat']);
data_path = 'D:/Dataset/Qualcomm/';

filelist = readtable(fullfile(data_path, "live_qualcomm_metadata.csv"));
% data_path = 'D:/Dataset/LIVE-VQC/';
% meta_data = readtable(fullfile(data_path, "livevqc_metadata.csv"));
filename = filelist.File_name;% meta_data = readtable(fullfile(data_path,"CVD2014_ratings/", "CVD_metadata.xlsx"));
%flicker_ids = meta_data.flickr_id;
% filename = meta_data.File_name;

%% *You need to customize here*
% if strcmp(data_name, 'konvid1k')
%     data_path = 'D:/Dataset/KoNViD_1k/KoNViD_1k_videos/';
% elseif strcmp(data_name, 'livevqc')
%     data_path = 'D:/Dataset/LIVE-VQC/Video';
% elseif strcmp(data_name, 'youtubeugc')
%     data_path = 'E:/datasets/KoNViD_1k_videos/KoNViD_1k_videos/';
% elseif strcmp(data_name, 'cvd')
%     data_path = 'E:/datasets/KoNViD_1k_videos/KoNViD_1k_videos/';
% end

%%
% create temp dir to store decoded videos
video_tmp = 'tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = './features';
% metadata至少要有：视频名字，视频帧高，视频帧宽
%dirVideos = dir(fullfile(data_path,'*.mp4'));

num_videos = size(filename, 1);
out_feat_name = fullfile(feat_path, [data_name,'_',algo_name,'_feats.mat']);
feats_mat = {};
%===================================================

%% extract features
% parfor i = 1 : num_videos % for parallel speedup
for i = 1 : num_videos
%     try
    % get video full path and decoded video name
    if strcmp(data_name, 'konvid1k')
        video_name = fullfile(data_path,  [num2str(filelist.flickr_id(i)),'.mp4']);
        yuv_name = fullfile(video_tmp, [num2str(filelist.flickr_id(i)), '.yuv']);
    elseif strcmp(data_name, 'livevqc')
        video_name = fullfile(data_path, filelist.File{i});
        yuv_name = fullfile(video_tmp, [filelist.File{i}, '.yuv']);
    elseif strcmp(data_name, 'youtubeugc')
        video_name = fullfile(data_path, filelist.category{i},...
            [num2str(filelist.resolution(i)),'P'],[filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    elseif strcmp(data_name, 'cvd')
        video_name = fullfile(data_path, filelist.pafold{i}, "/", filelist.fold{i}, "/", filelist.Content{i}, "/", filelist.File_name{i}+".avi");
        yuv_name = fullfile(video_tmp, [filelist.File_name{i}, '.yuv']);
    elseif strcmp(data_name, 'YoutubeUGC')
        video_name = fullfile(data_path, dirVideos(i,1).name);
        yuv_name = fullfile(video_tmp, [dirVideos(i,1).name, '.yuv']);
    elseif strcmp(data_name, 'live')
        yuv_name = fullfile(data_path, filelist.types{i}, filelist.File_name{i});

    end

    fprintf('\n---\nComputing features for %d-th sequence: %s\n', i, yuv_name);
                                                                                                                        % decode video and store in temp dir
%     if ~strcmp(video_name, yuv_name) 
%         cmd = strcat('ffmpeg -loglevel error -y -i', 32, video_name, 32, ' -pix_fmt yuv420p -vsync 0', 32, yuv_name);
%         system(cmd);
%     end  
%     vid = VideoReader(video_name);

    % get video meta data
    width = 1920;%vid.Width;
    height = 1080;%vid.Height;


    % calculate video features
    tic
    feats_mat{i} = calc_niqe_feats(yuv_name, width, height, mu_prisparam, ...
        cov_prisparam);
    toc
    % clear cache
    if ~strcmp(data_name, 'LIVE_VQA')
        delete(yuv_name)
    end
%     catch
%         feats_mat(i,:) = NaN;
%     end
end
% save feature matrix
save(out_feat_name, 'feats_mat');