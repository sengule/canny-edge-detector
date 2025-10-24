import cv2
from detector import canny_edge_detector

img = cv2.imread('test.png',cv2.IMREAD_GRAYSCALE)

#Detect edge
result = canny_edge_detector(img)

cv2.imshow("Original", img)
cv2.imshow("Canny Edge Detector", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
