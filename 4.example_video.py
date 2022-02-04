import cv2
import numpy as np
import matplotlib.pyplot as plt

capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
font = cv2.FONT_HERSHEY_SIMPLEX

def imgThreshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # GRAY 灰度图像 非黑即白
    # cv2.imshow("GRAY", GRAY)
    rosource,binary=cv2.threshold(gray,121,255,cv2.THRESH_BINARY)
    # cv2.imshow("binary", binary)
    return binary

while (capture.isOpened()):
    retval, image = capture.read()  #读取彩色图像
    img = imgThreshold(image)
    cv2.putText(img, 'Please wait...', (400, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("img", img)
    key = cv2.waitKey(1)
    if key == 27:
        (x, y) = img.shape
        print("x=", x, "y=", y)
        capture.release()
        cv2.destroyAllWindows()
        break
capture.release()
cv2.destroyAllWindows()

