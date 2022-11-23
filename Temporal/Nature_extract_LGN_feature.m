%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 数据
% data_path = 'D:/Dataset/CVD/';
% meta_data = readtable(fullfile(data_path,"CVD2014_ratings/", "CVD_metadata.xlsx"));
% %flicker_ids = meta_data.flickr_id;
% filename = meta_data.File_name;
% data_path = 'D:/Dataset/YoutubeUGC/';
% 
% dirVideos = dir(fullfile(data_path,'*.mp4'));

data_path = 'D:/Dataset/Qualcomm/';
filelist = readtable(fullfile(data_path, "live_qualcomm_metadata.csv"));
filename = filelist.File_name;

data_length = size(filename, 1);
curvatures_v1 = zeros(data_length, 1);
tStartTotal = tic;

for v = 89 : data_length
    tStart = tic;
    %video_name = fullfile(data_path, dirVideos(v,1).name);
    video_name = fullfile(data_path, filelist.types{v}, filelist.File_name{v});

    %vid = VideoReader(video_name);
    %meta_data.nb_frames(v);
    [NumFrames, LGN_features] = extract_lgn_features(video_name);
    LGN_features(isnan(LGN_features) | isinf(LGN_features)) = 0;

    [~,score] = pca(LGN_features);
    if NumFrames>size(score,2)
        LGN_features = score;
    else
        LGN_features = score(:, 1:NumFrames-1);
    end

    save(strcat('D:/SourceCode/STEM-main/Temporal_Quality/Qualcomm_LGN_TIP32/',filelist.File_name{v},'.mat'),'LGN_features')

    ts = toc(tStart);
    fprintf('Video %d, overall %f seconds elapsed...\n', v, ts);

end
tstotal = toc(tStartTotal);
fprintf('Video 1 - %d, overall %f seconds elapsed...\n', data_length, tstotal);

function img_features = extract_img_features(vid, num)
    % video_name is the file path to the video
    %vid = VideoReader(video_name);
    num_frames = num;
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
        frame_gray_norm = gaborFeatures(imresize(frame, 0.25),gaborArray,4,4);%normalize_gray_image(rgb2gray(frame));
        v1_features = [v1_features; frame_gray_norm(:).'];
    end
end

function [frame_end, lgn_features] = extract_lgn_features(video_name)
    test_file = fopen(video_name,'r');
    if test_file == -1
        fprintf('Test YUV file not found.');
        niqe_score = [];
        return;
    end
    % Open test video file
    fseek(test_file, 0, 1);
    file_length = ftell(test_file);
    % get frame number
    width = 1920;
    height = 1080;
    frame_end = floor(file_length/width/height/1.5) - 1;
    lgn_features = [];
    for f = 1:frame_end
        yuvframe = YUVread(test_file,[width height],f);
        frame_gray = double(rgb2gray(imread('image.png')));
        [y, ~] = frame_LGN_features(frame_gray);
        y = y{1, 6};
        y = y(:);
        lgn_features = [lgn_features; y.'];
    end
end


function YUV = YUVread(f,dim,frnum)

    % This function reads a frame #frnum (0..n-1) from YUV file into an
    % 3D array with Y, U and V components
    
    fseek(f,dim(1)*dim(2)*1.5*frnum,'bof');
    
    % Read Y-component
    Y=fread(f,dim(1)*dim(2),'uchar');
    if length(Y)<dim(1)*dim(2)
        YUV = [];
        return;
    end
    Y=cast(reshape(Y,dim(1),dim(2)),'double');
    
    % Read U-component
    U=fread(f,dim(1)*dim(2)/4,'uchar');
    if length(U)<dim(1)*dim(2)/4
        YUV = [];
        return;
    end
    U=cast(reshape(U,dim(1)/2,dim(2)/2),'double');
    U=imresize(U,2.0);
    
    % Read V-component
    V=fread(f,dim(1)*dim(2)/4,'uchar');
    if length(V)<dim(1)*dim(2)/4
        YUV = [];
        return;
    end    
    V=cast(reshape(V,dim(1)/2,dim(2)/2),'double');
    V=imresize(V,2.0);
    U = U-128;
    V=V-128;
    % Combine Y, U, and V
    YUV(:,:,1)=Y'+1.402.*V';
    YUV(:,:,2)=Y'-0.34414.*U'-0.71414.*V';
    YUV(:,:,3)=Y'+1.772.*U';
    
    YUV = round(YUV);
    YUV(YUV>255)=255;
    YUV(YUV<0)=0;
    imwrite(YUV/255, 'image.png');
    %figure; imshow(YUV/255)
end