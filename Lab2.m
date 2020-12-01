%% ------------------3.1-------------------- %%
%Load img
Pc = imread('macritchie.jpg');

%Check for RGB or Gray_scale
whos Pc

%Convert to grayscale
Pc = rgb2gray(Pc);

%Show img
imshow(Pc, []);

%Create hori and verti sobel mask
horiSobel = [-1 -2 -1; 
              0  0  0;
              1  2  1];
          
vertiSobel = [-1 0 1; 
              -2 0 2;
              -1 0 1];
          
%convolution img with sobel mask
img = conv2(double(Pc),double(horiSobel));
imghori = abs(img)/4;

imgverti = conv2(double(Pc),double(vertiSobel));
imgverti = abs(imgverti)/4;

%show result
figure;imshow(imghori, []);title('Horizontal Sobel Filter')  
figure;imshow(imgverti, []);title('Vertical Sobel Filter')  

%squaring operation
imgedge = sqrt(imghori.^2+imgverti.^2);
figure;imshow(imgedge, []);title('Combined Sobel  Filter')

%threshold for imgedge
threshold100 = imgedge>100;
threshold75 = imgedge>75;
threshold50 = imgedge>50;

%different results for threshold
figure;imshow(threshold100, []);title('t = 100')
figure;imshow(threshold75, []);title('t = 75')
figure;imshow(threshold50, []);title('t = 50')  

%Use Canny edge detection with tl=0.04, th=0.1, sigma=1.0, 2.5, 5
canny1 = edge(Pc, 'canny', [0.04 0.1], 1.0);
canny2 = edge(Pc, 'canny', [0.04 0.1], 2.5);
canny3 = edge(Pc, 'canny', [0.04 0.1], 5.0);

%show results
figure;imshow(canny1, []);title('sigma = 1')  
figure;imshow(canny2, []);title('sigma = 2.5')  
figure;imshow(canny3, []);title('sigma = 5')  

%Change values of TL for canny detection with fix value sigma = 3
canny4 = edge(Pc, 'canny', [0.01 0.1], 3.0);
canny5 = edge(Pc, 'canny', [0.05 0.1], 3.0);
canny6 = edge(Pc, 'canny', [0.09 0.1], 3.0);

%show results
figure;imshow(canny4, []);title('lower threshold = 0.01')
figure;imshow(canny5, []);title('lower threshold = 0.05')
figure;imshow(canny6, []);title('lower threshold = 0.09')

%% ------------------3.2-------------------- %%
canny = edge(Pc, 'canny', [0.04 0.1], 1.0);
figure;imshow(canny, []);

[Hough, xp] = radon(canny);
%show hough space result
figure;imagesc(uint8(Hough));title('Hough Space');
xlabel('THETA');
ylabel('RHO');

%Get max intensity
maxpeak = max(Hough, [], 'all');
[rho, theta] = find(Hough >= maxpeak);
fprintf('rho = %d\ntheta = %d\n', rho, theta);

radius = xp(rho);
[A, B] = pol2cart(theta*pi/180, radius);
B = -B;
C = A*(A+179)+B*(B+145);
fprintf('A = %d\nB = %d\nC = %d\n', A, B, C);

% Pc dimension 290 x 358
xl = 0;
yl = (C-A*xl)/B;
xr = 358-1;
yr = (C-A*xr)/B;
fprintf('yl = %d\nyr = %d\n', yl, yr);

figure;imshow(Pc, []);
line([xl xr], [yl yr], 'Color', 'blue');

%% ------------------3.3-------------------- %%
%show corridor left and right
left = imread('corridorl.jpg'); 
left = rgb2gray(left);
right = imread('corridorr.jpg'); 
right = rgb2gray(right);

figure;imshow(left, []);title('corridorl.jpg')  
figure;imshow(right, []);title('corridorr.jpg') 

%show disparitymap for corridor image
D = dmap(left, right, 11, 11);
ref = imread('corridor_disp.jpg');
figure;imshow(D, [-15 15]);title('Result') 
figure;imshow(ref, []);title('Reference') 

%show triclopsi2 left and right
left2 = imread('triclopsi2l.jpg'); 
left2 = rgb2gray(left2);
right2 = imread('triclopsi2r.jpg'); 
right2 = rgb2gray(right2);
figure;imshow(left2, []);title('triclopsi2l.jpg')  
figure;imshow(right2, []);title('triclopsi2r.jpg')  

%show disparitymap for triclopsi2
D2 = dmap(left2, right2, 11, 11);
ref2 = imread('triclopsid.jpg');

figure;imshow(D2, [-15 15]);title('Result')
figure;imshow(ref2, []);title('Reference')    