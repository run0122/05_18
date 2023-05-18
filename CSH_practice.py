import cv2
import numpy as np
import matplotlib.pyplot as plt

# 영상 읽기
img1 = cv2.imread('../img2/beach.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img = cv2.imread('../img2/beach.jpg', cv2.IMREAD_GRAYSCALE)

blk_size = 9        # 블럭 사이즈
C = 5 

ret, thresh_simple = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
thresh_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk_size, C)
ret, thresh_otsu = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

hist = cv2.calcHist([img], [0], None, [256], [0, 255])

# 바이너리 이미지로 변환
ret, imthres = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 가장 바깥 컨투어만 수집
contours, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

# 컨투어를 크기에 따라 정렬
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 최대 5개의 컨투어만 선택
contours = contours[:5]

# Create a subplot with 4 rows and 2 columns
fig, axs = plt.subplots(4, 2, figsize=(12, 16))

# Plot the original image
axs[0, 0].imshow(img1)
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

# Plot the grayscale image
axs[0, 1].imshow(img, cmap='gray')
axs[0, 1].set_title('Gray')
axs[0, 1].axis('off')

# Plot the simple thresholding result
axs[1, 0].imshow(thresh_simple, cmap='gray')
axs[1, 0].set_title('Threshold: 127')
axs[1, 0].axis('off')

# Plot the Otsu's thresholding result
axs[1, 1].imshow(thresh_otsu, cmap='gray')
axs[1, 1].set_title('Threshold: Otsu')
axs[1, 1].axis('off')

# Plot the adaptive thresholding result
axs[2, 0].imshow(thresh_adaptive, cmap='gray')
axs[2, 0].set_title('Threshold: Adaptive')
axs[2, 0].axis('off')

# Copy the original image to draw contours on
img_with_contours = img1.copy()

for i, contour in enumerate(contours):
    # 컨투어의 중심점 계산
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # 컨투어 그리기
    cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 0), 3)

    # 중심점에 숫자 표시
    cv2.putText(img_with_contours, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Plot the image with contours
axs[2, 1].imshow(img_with_contours)
axs[2, 1].set_title('Image with Contours')
axs[2, 1].axis('off')

# Plot the histogram
axs[3, 1].plot(hist, color='black')
axs[3, 1].set_xlim([0, 256])
axs[3, 1].set_xlabel('Pixel Value')
axs[3, 1].set_ylabel('Frequency')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the subplots
plt.show()