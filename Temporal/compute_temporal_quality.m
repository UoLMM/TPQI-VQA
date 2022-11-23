function q_temporal = compute_temporal_quality(video_name, k, d, pre_computed)
    % video_name is the file path to the video
    % k, d, alpha, pf is optional
    vid = VideoReader(video_name);
    num_frames = vid.NumFrames;

    % 默认参数
    % the number of past frames considered in the linear model
    if (~exist('k', 'var'))
        k = 6;
    end
    % the dimension of the low dimensional representation of the LGN feature map
    if (~exist('d', 'var'))
        d = 10;
    end
    % whether the lgn features are calculated in advance
    if (~exist('pre_computed', 'var'))
        pre_computed = false;
    end

    if (~pre_computed)
        lgn_features = [];
        for f = 1:num_frames
            frame = read(vid, f);
            frame_gray = double(rgb2gray(frame));
    
            % 使用LGN提取特征
            [y, ~] = frame_LGN_features(frame_gray);
            y = y{1, 6};
            y = y(:);
            lgn_features = [lgn_features;y.'];
        end
    else
%         load()
    end

    % 对LGN提取的特征使用PCA降维
    [~,score] = pca(lgn_features);
    lgn_features = score(:, 1:d);

    % temporal quality estimation
    dt = zeros(num_frames, 1);
    for t = k + 1:num_frames
        features_win = lgn_features(t - k : t - 1, :);
        feature_cur = lgn_features(t, :);
        b = robustfit(features_win.', feature_cur.');
        sz = size(features_win);
        feature_pred = b(1) + b(2:sz + 1).' * features_win;

        dt(t) = norm(feature_cur - feature_pred, 1);
    end
    q_temporal = log(mean(dt));
end