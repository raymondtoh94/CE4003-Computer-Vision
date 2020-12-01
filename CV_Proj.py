import pytesseract
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import ndimage

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\raymo\AppData\Local\Tesseract-OCR\tesseract.exe'

def adaptive_threshold(img, block_size, offset=0, method='gaussian', param=None, cval=0):
    if block_size % 2 == 0:
        raise ValueError(f"BlockSize {block_size} must be odd! Given block_size {block_size} is even.".format(block_size))

    if img.ndim == 3 and img.shape[-1] != 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    thresh_img = np.zeros(img.shape, 'double')
    
    if method == 'generic':
        ndimage.generic_filter(img, param, block_size, output=thresh_img, cval=cval)
        
    elif method == 'gaussian':
        if param is None:
            # default sigma that cover 99% distribution
            sigma = (block_size - 1) / 6
        else:
            sigma = param
        ndimage.gaussian_filter(img, sigma, output=thresh_img, cval=cval)
        
    elif method == 'mean':
        mask = 1 / block_size * np.ones((block_size,))
        # split 2d to 1d for faster convolve
        ndimage.convolve1d(img, mask, axis=0, output=thresh_img, cval=cval)
        ndimage.convolve1d(thresh_img, mask, axis=1, output=thresh_img, cval=cval)
    
    elif method == 'median':
        
        ndimage.median_filter(img, block_size, output=thresh_img, cval=cval)
    else:
        print(f"{method} is invalid.")

    return thresh_img - offset

def gaus_threshold(img):
    if img.ndim == 3 and img.shape[-1] != 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #otsuimg1(1,193), otsuimg2(34,31), adimg1(15,9), adimg2(65,5), kmeanimg1(23,3), kmeanimg2(23,3)
    T = adaptive_threshold(img, 15, offset = 9, method = "gaussian") 
    img = (img > T).astype("uint8") * 255
    return img

def threshold_otsu(img):
    if img.ndim == 3 and img.shape[-1] != 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    pixel_counts, bin_centers = np.histogram(img.ravel(), 255)
    bin_centers = bin_centers[:255]
    pixel_counts = pixel_counts.astype(float) 
    weight1 = np.cumsum(pixel_counts) 
    weight2 = np.cumsum(pixel_counts[::-1])[::-1] 
    mean1 = np.cumsum(pixel_counts * bin_centers) / weight1 
    mean2 = (np.cumsum((pixel_counts * bin_centers)[::-1]) / weight2[::-1])[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2 
    idx = np.argmax(variance12) 
    t = bin_centers[:-1][idx] 
    
    row = []
    for i in img:
        col = []
        for j in i:
            if j < t:
                col.append(0)
            else:
                col.append(255)
        row.append(col)
    img = np.array(row, dtype="uint8")
    return img

def classic_adaptive_threshold(img, blockdims=(2,2)):
    if img.ndim == 3 and img.shape[-1] != 1:
         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         
    #Split into blocks of image
    block = []
    for i in np.vsplit(img,blockdims[0]):
        for j in np.hsplit(i,blockdims[1]):
            block.append(threshold_otsu(j))
    block = np.array(block)
    
    #Join blocks to form image
    newblock = []
    result = []
    counter=1
    for i in range(0,len(block),blockdims[1]):
        test = np.hstack((block[i:blockdims[1]*counter]))
        counter+=1
        newblock.append(test)
    result.append(np.vstack((newblock[:])))
    result = np.squeeze(np.array(result), axis=0)
    return result

def ocr_acc(img_str, string_str):
    count=0
    if len(img_str) > len(string_str):
        for i in range(len(string_str)):
            if string_str[i] in img_str:
                count+=1
    else:
        for i in range(len(img_str)):
            if string_str[i] in img_str:
                count+=1
    return count/len(img_str)*100

def kmean_cluster_img1(img):
    if img.ndim == 3 and img.shape[-1] != 1:
         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = img.reshape((-1, 1))
    image = np.float32(image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # number of clusters (K)
    k = 2
    _, labels, (centers) = cv2.kmeans(image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
         
    # convert back to 8 bit values
    centers = np.uint8(centers)
    
    # flatten the labels array
    labels = labels.flatten()
    
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)
    
    #Image split based on segmented_image
    topLeft = (600,1)
    bottomRight = (965,229)
    x, y = topLeft[0], topLeft[1]
    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
    # Extract Region of Interest
    ROI = img[y:y+h, x:x+w]
    blur = gaus_threshold(ROI)
    img[y:y+h, x:x+w] = blur
    return img

def kmean_cluster_img2(img):
    if img.ndim == 3 and img.shape[-1] != 1:
         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = img.reshape((-1, 1))
    image = np.float32(image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # number of clusters (K)
    k = 3
    _, labels, (centers) = cv2.kmeans(image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
         
    # convert back to 8 bit values
    centers = np.uint8(centers)
    
    # flatten the labels array
    labels = labels.flatten()
    
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)
    
    topLeft = (0,0)
    bottomRight = (589,290)
    # img = filter(img,kernel_low_3)
    x, y = topLeft[0], topLeft[1]
    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
    ROI = img[y:y+h, x:x+w]
    blur = gaus_threshold(ROI)
    img[y:y+h, x:x+w] = blur
    
    # Make the 2nd cluster rectangle (centre rectangle)
    topLeft = (0,290)
    bottomRight = (589,470)
    x, y = topLeft[0], topLeft[1]
    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
    ROI = img[y:y+h, x:x+w]
    blur = gaus_threshold(ROI)
    img[y:y+h, x:x+w] = blur
    
    #Make a rectangle first ( bottom half)
    topLeft = (0,470)
    bottomRight = (589,782)
    x, y = topLeft[0], topLeft[1]
    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
    ROI = img[y:y+h, x:x+w]
    blur = gaus_threshold(ROI)
    img[y:y+h, x:x+w] = blur
    return img

if __name__ == "__main__":
    #Load images
    img1 = cv2.imread('sample01.png')
    img2 = cv2.imread('sample02.png')
    
    #Correct String for 100% Accuracy
    img1_str = "Parking: You may park anywhere on the campus where there are no signs prohibiting par-\nking. Keep in mind the carpool hours and park accordingly so you do not get blocked in the\nafternoon\n\nUnder School Age Children:While we love the younger children, it can be disruptive and\ninappropriate to have them on campus during school hours. There may be special times\nthat they may be invited or can accompany a parent volunteer, but otherwise we ask that\nyou adhere to our _ policy for the benefit of the students and staff.\n\x0c".split(' ')
    img2_str = "Sonnet for Lena\n\nO dear Lena, your beauty is so vant\nIt is hard sometimes to describe it fast.\nI thought the entire world I would impress\nIf only your portrait i could compress.\nAlas! First when i tried to use VQ\nI found that your cheeks belong to only you.\nYour silky hair contains a thousand lines\nHard to match with sums of discrete cosines.\nAnd for your lips, sensual and tactual\nThirteen Crays found not the proper fractal.\nAnd while these setbacks are all quite servere\nI might have fixed them with hacks here or there\nBut when filters took sparkle from your eyes\nI said, ‘Damn all this. I'll just digitize.”\n\nThomas Colthurst\n\x0c".split(' ')

    #Replace function for different type of results
    #All function required to change gaus_threshold hyperparameter
    
    #Classic Otsu
    #img = threshold_otsu(img1)
    
    #Classic Adaptive_Otsu
    #img = classic_adaptive_threshold(img2, blockdims=(34,31))
    
    #Adaptive_Gaus or others
    #(Mean/Median) required to change method in adaptive_threshold function written in gaus_thredhold
    
    img = gaus_threshold(img1)
    
    #K-mean Cluster
    #img = kmean_cluster_img1(img1)
    
    #Show figures and Calculate Accuracy
    string = pytesseract.image_to_string(img)
    string_str = string.split(' ')
    
    #Update the ocr_acc first parameter to either img1_str or img2_str
    print(f"OCR Accuracy - {ocr_acc(img1_str, string_str):.2f}%")
    plt.figure(figsize = (20,10))
    plt.imshow(img, cmap="gray")
    #plt.savefig("img.png")