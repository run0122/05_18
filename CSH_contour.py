import cv2
import numpy as np

# 영상 읽기
img = cv2.imread('../img2/beach.jpg', cv2.IMREAD_GRAYSCALE)

# 바이너리 이미지로 변환
ret, imthres = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 가장 바깥 컨투어만 수집
contours, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

# 컨투어를 크기에 따라 정렬
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 최대 5개의 컨투어만 선택
contours = contours[:5]

for i, contour in enumerate(contours):
    # 컨투어의 중심점 계산
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # 컨투어 그리기
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)

    # 중심점에 숫자 표시
    cv2.putText(img, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# 화면 출력
cv2.imshow('Largest Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
