%What i learn when doing lab
%Cannot give assignment to min or max, Example(min = .....)
%min_val = uint8(zeros(size(gray_image)+double(min(gray_image(:)))
%Make Matrix of MxN of 255 - kron(ones(size(gray_image)),255)

%------------------2.1--------------------%
%Load Image
image = imread('mrt-train.jpg');

%Check for RGB or Gray_scale
whos image

%Convert to grayscale
gray_image = rgb2gray(image);

%Show Image
imshow(gray_image);

%Get Min = 13 and Max = 204 Gray Intensity
min(gray_image(:)), max(gray_image(:))

%Stretch to 0-255 Intensity
stretched_image = (255/191)*imsubtract(gray_image,13);

%Check for Min, Max -> 0,255
min(stretched_image(:)), max(stretched_image(:))

%Show Stretched Image
imshow(stretched_image,[]);


%------------------2.2--------------------%
%Show Image Intensity Historygram of Gray Image
imhist(gray_image, 10)
imhist(gray_image, 256)

%Do Histogram Equalization for Gray Image
histeq_image = histeq(gray_image, 255);

%Show Histogram Equalization for Gray Image
imhist(histeq_image, 10)
imhist(histeq_image, 256)

imshow(histeq_image)
%Histogram become more uniform because it spread out the most frequent
%intensity values which allows for area with lower local contrast to gain
%a higher contrast.

%------------------2.3--------------------%
%Load Image
ntu_gn = imread('lib-gn.jpg');

%Display Image
imshow(ntu_gn);

%Declare Variable
sigma_1 = 1.0;
sigma_2 = 2.0;

%Axis for filter
dim=5;
range = -floor(dim/2) : floor(dim/2);
[X, Y] = meshgrid(range,range);



%Write h(x,y,sigma) function in matladb -> h.m
%To use function -> h(x,y,sigma)
%Fill values in 5x5 Filter
G1 = h(X,Y,sigma_1);
G2 = h(X,Y,sigma_2);

%Normalize Filter
G1 = G1/sum(G1(:));
G2 = G2/sum(G1(:));

%View my Gaus_Filter
mesh(G1)
mesh(G2)

%Convolution NTU Image with Filter
G1OP = uint8(conv2(ntu_gn, G1));
G2OP = uint8(conv2(ntu_gn, G2));

%Display Resultant Image
figure;imshow(G1OP)
figure;imshow(G2OP)

%Using Gaussian averaging filter, with the increase of sigma, the noise is reduced at the expense of
%the clarity and brightness of the image, resulting in a blurrer and darker
%image. In this example, original image will still be perferred as i
%believe human eye able to pick up more information compared to the
%filtered image.

%Load Image
ntu_sp = imread('lib-sp.jpg');

%Display Image
figure;imshow(ntu_sp)

%Filter with another NTU Image with Filter
G1OP2 = uint8(conv2(ntu_sp, G1));
G2OP2 = uint8(conv2(ntu_sp, G2));

%Display Resultant Image
figure;imshow(G1OP2)
figure;imshow(G2OP2)

%After applying to speckle noise, speckle noise still able to be seen from
%human eye. Image also became blury. Hence, gaussian averaging filter still work better in removing gaussian noise

%------------------2.4--------------------%
%Applying Median Filter to NTU Image
M1 = medfilt2(ntu_gn,[3,3]); %G_noise
M2 = medfilt2(ntu_gn,[5,5]); %G_noise
M3 = medfilt2(ntu_sp,[3,3]); %S_noise
M4 = medfilt2(ntu_sp,[5,5]); %S_noise

%Display Resultant Image
figure;imshow(M1)
figure;imshow(M2)
figure;imshow(M3)
figure;imshow(M4)

%Median Filter for neightbour sizes of 3x3 work extremely good on NTU image
%with speckle noise, as compared to 5x5 median filter was a little blury.
%Median filter does not work well for Gaussian Noise image as the noise
%still can be seen after applying the filter. In Conclusion, Gaussian
%Average Filter work better for image with gaussian noise, and Median
%Filter work better for image with speckle noise.

%------------------2.5--------------------%
%Load and display Image
pck = imread('pck-int.jpg');
imshow(pck)

%Transform Image using Fast Fourier Transform 2D
fftpck = fft2(pck);

%Shift DC component to centre of image
imagesc(fftshift(real(fftpck.^0.1)))
colormap('default')

%Display image with no fftshift
imagesc(real(fftpck.^0.1))

%Capture X Y Corrdinate of two distinct frequency peak
%coord1 = [9.0161, 241.1676], coord2 = [249.1882, 17.5552]
%[x y] = ginput(2)

%Remove Two distinct frequency peak in spatial domain
fftpck_zero = fftpck;
dim=floor(5/2);

%Notice [X Y], in domain will be [Y X]
fftpck_zero(241-dim:241+dim,9-dim:9+dim)=0;
fftpck_zero(17-dim:17+dim,249-dim:249+dim)=0;
imagesc(real(fftpck_zero.^0.1))

%Perform Inverse Fourier Transform
fftpck_zero_ifft = uint8(ifft2(fftpck_zero));
imshow(fftpck_zero_ifft)

%The interference patterns were reduced. When we do a fourier transform on
%image, we are able to see the origin on the top left hand corner of the
%image due to low frequency and the interference pattern frequency was in
%the top right and bottom left edge due to Euler's identity. By applying
%fftshift, the low frequency component of image will be at the centre of
%image and frequency will go higher as it approaches the edge of image.

%As there are still interference frequency shown in spatial domain, we can
%futher improve the image by zeroing the frequency by doing this.

%Make another copy
fftpck_zero2 = fftpck;

%Notice [X Y], in domain will be [Y X]
%coord1 = [9.0161, 241.1676], coord2 = [249.1882, 17.5552]
fftpck_zero2(241-dim:241+dim,9-dim:9+dim)=0;
fftpck_zero2(17-dim:17+dim,249-dim:249+dim)=0;

%Extend the zero to edge
fftpck_zero2(:,9)=0;
fftpck_zero2(241,:)=0;
fftpck_zero2(:,249)=0;
fftpck_zero2(17,:)=0;

%Display the zero
imagesc(real(fftpck_zero2.^0.1))

%Perform Inverse Fourier Transform
fftpck_zero2_ifft = uint8(ifft2(fftpck_zero2));
imshow(real(fftpck_zero2_ifft))

%----------2.5(f)----------%
pc = imread('primate-caged.jpg');

%Check for RGB or grayscale image
whos pc

%Change to grayscale
pc = rgb2gray(pc);

%Fourier Transform 2D
fftpc = fft2(pc);

%Check for spectrum, we can see there are 4 tiny spectrum causes the effect
imagesc(fftshift(real(fftpc.^0.5)));

fftpc_zero = fftpc;
imagesc(real(fftpc_zero.^0.5));

%[x y] = ginput(2)
x1=251; y1=11; fftpc_zero(x1-3:x1+3,y1-3:y1+3) = 0;
x2=10; y2=237; fftpc_zero(x2-3:x2+3,y2-3:y2+3) = 0;
x3=247; y3=21; fftpc_zero(x3-3:x3+3,y3-3:y3+3) = 0;
x4=6; y4=247; fftpc_zero(x4-3:x4+3,y4-3:y4+3) = 0;
imagesc(real(fftpc_zero.^0.5));

%Perform Inverse Fourier Transform
filted_fftpc_zero = uint8(ifft2(fftpc_zero));
imshow(real(filted_fftpc_zero))


%------------------2.6--------------------%
%Load and display Image
book = imread('book.jpg');
imshow(book)

[x, y] = ginput(4);

%A4 Dimension
op = [0;0;210;0;210;297;0;297];

%8x8 Matrix
A = [
    [x(1),y(1),1,0,0,0,-op(1)*x(1),-op(1)*y(1)];
    [0,0,0,x(1),y(1),1,-op(2)*x(1),-op(2)*y(1)];
    [x(2),y(2),1,0,0,0,-op(3)*x(2),-op(3)*y(2)];
    [0,0,0,x(2),y(2),1,-op(4)*x(2),-op(4)*y(2)];
    [x(3),y(3),1,0,0,0,-op(5)*x(3),-op(5)*y(3)];
    [0,0,0,x(3),y(3),1,-op(6)*x(3),-op(6)*y(3)];
    [x(4),y(4),1,0,0,0,-op(7)*x(4),-op(7)*y(4)];
    [0,0,0,x(4),y(4),1,-op(8)*x(4),-op(8)*y(4)];
];

%8x1 Matrix = 8x8 \ 8x1
u = inv(A)*op;

U = reshape([u;1],3,3)';
w = U*[x'; y'; ones(1,4)];
w = w ./ (ones(3,1)*w(3,:));
T = maketform('projective', U');
P2 = imtransform(book, T, 'XData', [0 210], 'YData', [0 297]);

imshow(P2)