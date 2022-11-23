function niqe_score = calc_niqe_feats(test_video, width, height, mu_prisparam, cov_prisparam)
    % Try to open test_video; if cannot, return
    test_file = fopen(test_video,'r');
    if test_file == -1
        fprintf('Test YUV file not found.');
        niqe_score = [];
        return;
    end
    % Open test video file
    fseek(test_file, 0, 1);
    file_length = ftell(test_file);
    fprintf('Video file size: %d bytes (%d frames)\n',file_length, ...
            floor(file_length/width/height/1.5));
    % get frame number
    frame_start = 0; 
    frame_end = floor(file_length/width/height/1.5) - 1;
    % params
    blocksizerow    = 96;
    blocksizecol    = 96;
    blockrowoverlap = 0;
    blockcoloverlap = 0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Loop through all the frames in the frame_range to compute frame features 
    %
    fprintf('Computing frame features every second frame on frames %d..%d\n', ...
            frame_start, frame_end);
     
    niqe_score = zeros(frame_end - frame_start + 1, 1);
    time = 0
    for i = frame_start : frame_end
        frame = YUVread(test_file,[width height],i);
%         imshow(frame);
        tStart = tic;

        frame_gray = uint8(frame(:, :, 1));
        frame_gray = imresize(frame_gray, [width,height]);
%         niqe_score(i) = niqe(frame_gray);
        niqe_score(i + 1) = computequality(frame_gray, blocksizerow, blocksizecol, ...
            blockrowoverlap, blockcoloverlap, mu_prisparam, cov_prisparam);
        time = time+toc(tStart);

    end
    fprintf('270P, overall %f seconds elapsed...\n', time/(frame_end-frame_start));


    fclose(test_file);
end

% Read one frame from YUV file
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
    
    % Combine Y, U, and V
    YUV(:,:,1)=Y';
    YUV(:,:,2)=U';
    YUV(:,:,3)=V';
end