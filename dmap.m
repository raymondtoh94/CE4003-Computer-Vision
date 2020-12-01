function map = dmap(imgleft, imgright, tempheight, tempwidth)
[height, width] = size(imgleft);

map = ones(height - tempheight + 1, width - tempwidth + 1);
imgstart = round(tempwidth / 2);
th = floor(tempheight / 2);
tw = floor(tempwidth / 2);
for row = imgstart:height - imgstart - 1
    for col = imgstart:width - imgstart - 1
        
        T = imgleft(row-th:row+tw,col-th:col+tw);
        left = col-14;
        right = col;
        if left < imgstart
            left = imgstart;
        end
        ssd_min = Inf;
        xr_min = left;
        for xr = left:right
            I = imgright(row-th:row+tw,xr-th:xr+tw);
            I_flipped = rot90(I, 2);
            ssd_1 = ifft2(fft2(I) .* fft2(I_flipped));
            ssd_1 = ssd_1(tempheight, tempwidth);
            ssd_2 = ifft2(fft2(T) .* fft2(I_flipped));
            ssd_2 = ssd_2(tempheight, tempwidth) * 2;
            ssd = ssd_1 - ssd_2;
            if ssd < ssd_min
                ssd_min = ssd;
                xr_min = xr;
            end
        end
        map(row, col) = col - xr_min;
    end
end