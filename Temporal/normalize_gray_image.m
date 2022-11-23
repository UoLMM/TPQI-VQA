function dblImageS1 = normalize_gray_image(gray_img)
    % Get a double image in the range 0 to +1
    originalMinValue = double(min(min(gray_img)));
    originalMaxValue = double(max(max(gray_img)));
    originalRange = originalMaxValue - originalMinValue;

    desiredMin = 0;
    desiredMax = 1;
    desiredRange = desiredMax - desiredMin;
    dblImageS1 = desiredRange * (double(gray_img) - originalMinValue) / originalRange + desiredMin;
end