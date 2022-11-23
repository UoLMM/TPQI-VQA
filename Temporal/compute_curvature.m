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
        denominator = norm(next - current) * norm(current - prev);
        cos_alpha = acos(numerator / denominator);
        curvatures(fn - 1) = cos_alpha;
    end
end