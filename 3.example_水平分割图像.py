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

# 对图片进行垂直分割，传入的是二值图
def verticalCut(img,img_num):
    (x,y)=img.shape #返回的分别是矩阵的行数和列数，x是行数，y是列数
    pointCount=np.zeros(y,dtype=np.float32)#每列黑色的个数
    x_axes=np.arange(0,y)
    #i是列数，j是行数
    for i in range(0,x):
        for j in range(0,y):
            if(img[i,j]==0):
                pointCount[j]=pointCount[j]+1
    figure=plt.figure(str(img_num))
    plt.plot(x_axes,pointCount)
    # plt.plot(pointCount, x_axes)
    start = []
    end = []
    # 对照片进行分割
    print(pointCount)
    plt.show()
    for index in range(1, y-1):
        # 上个为0当前不为0，即为开始
        if ((pointCount[index-1] == 0) & (pointCount[index] != 0)):
            start.append(index)
        # 上个不为0当前为0，即为结束
        elif ((pointCount[index] != 0) & (pointCount[index +1] == 0)):
            end.append(index)
    imgArr=[]
    for idx in range(0,len(start)):
        tempimg=img[ :,start[idx]:end[idx]]
        cv2.imshow(str(img_num)+"_"+str(idx), tempimg)
        cv2.imwrite(img_num+'_'+str(idx)+'.jpg',tempimg)
        imgArr.append(tempimg)
    return imgArr


img = cv2.imread("img1.jpg")  # 读取原始图像
cv2.imshow("img", img)
binary = imgThreshold(img)  #二值化处理函数
verticalCut(binary, "1")
cv2.waitKey()  # 按下任何键盘按键后
cv2.destroyAllWindows() # 释放所有窗体