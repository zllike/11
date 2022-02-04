import cv2
import numpy as np
import matplotlib.pyplot as plt

#图像二值化处理
#输入读取的图片数据 -- img
#返回二值化处理后的图像 -- binary
def imgThreshold(img):
    GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # GRAY 灰度图像 非黑即白
    # cv2.imshow("GRAY", GRAY)
    rosource,binary=cv2.threshold(GRAY,121,255,cv2.THRESH_BINARY)
    # cv2.imshow("binary", binary)
    return binary

#对图片进行水平分割,返回的是分割好的照片数组
# img 输入的是 灰度图像二值化处理 后的结果
def horizontalCut(img):
    (x,y)=img.shape #返回的分别是矩阵的行数和列数，x是行数，y是列数
    pointCount=np.zeros(y,dtype=np.uint8)#每行黑色的个数
    x_axes=np.arange(0,y)
    for i in range(0,x):
        for j in range(0,y):
            if(img[i,j]==0):
                pointCount[i]=pointCount[i]+1
    plt.plot(x_axes,pointCount)
    start=[]#开始索引数组
    end=[]#结束索引数组
    #对照片进行分割
    for index in range(1,y):
        #上个为0当前不为0，即为开始
        if((pointCount[index]!=0)&(pointCount[index-1]==0)):
            start.append(index)
        #上个不为0当前为0，即为结束
        elif((pointCount[index]==0)&(pointCount[index-1]!=0)):
             end.append(index)
    img1=img[start[0]:end[0],:]
    img2=img[start[1]:end[1],:]
    img3=img[start[2]:end[2],:]
    cv2.imwrite("img1.jpg", img1)
    imgArr=[img1,img2,img3]
    for m in range(3):
        cv2.imshow(str(m),imgArr[m])
    plt.show()
    return imgArr

img = cv2.imread("1.jpg")  # 读取原始图像
cv2.imshow("img", img)
binary = imgThreshold(img)  #二值化处理函数
horizontalCut(binary)
cv2.waitKey()  # 按下任何键盘按键后
cv2.destroyAllWindows() # 释放所有窗体
