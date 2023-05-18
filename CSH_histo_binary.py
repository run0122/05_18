import cv2
import numpy as np
import matplotlib.pyplot as plt

blk_size = 9        # 블럭 사이즈
C = 5 

img1 = cv2.imread('../img2/beach.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img = cv2.imread('../img2/beach.jpg', cv2.IMREAD_GRAYSCALE)

ret, thresh_simple = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
thresh_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                      cv2.THRESH_BINARY, blk_size, C)
ret, thresh_otsu = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

hist = cv2.calcHist([img], [0], None, [256], [0, 255])

imgs = {'Original': img1, 'Gray': img, 'threshold:127': thresh_simple, 
        'threshold:otsu': thresh_otsu, 'threshold:adaptive': thresh_adaptive}

plt.figure(figsize=(10, 10))

for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(3, 2, i + 1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    if key == 'Gray':
        plt.subplot(3, 2, 6)
        plt.title('Histogram')
        plt.plot(hist, color='black')
        plt.xlim([0, 256])
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

plt.tight_layout()
plt.show()