import MyUtils
import cv2
from matplotlib import pyplot as plt
import numpy as np

class Frame:
    SD=-1
    hist = []
    index = -1
    img = None
    isCut= False

def ColorHistTwinComparisonDetect(VideoName):
    Tb = 70000  # 较高的阈值
    Ts = 20000  # 较低的阈值

    Fs = -1  # 渐变开始的下标
    Fe = -1  # 渐变结束的下标

    isbegin = False
    myFrames = []
    # step = 50
    step = 30
    list_sd = []
    num = 0

    result_path = "HistTwinComparisonResult"

    MyUtils.makeDir(result_path)
    cap = cv2.VideoCapture(VideoName)  # 提取视频
    index = -1
    plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    while (cap.isOpened()):

        ret, frame = cap.read()
        if ret == False:
            break
        index += 1
        if index>= 3563:
            break

        img = frame
        temp_frame = Frame()
        temp_frame.index = index
        temp_frame.img = img.copy()
        myFrames.append(temp_frame)

        img = cv2.GaussianBlur(img, (9, 9), 0.0)  # 做高斯模糊
        img= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s,v = cv2.split(img)
        hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])  # 计算直方图
        hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])  # 计算直方图
        hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])  # 计算直方图
        weight= [0.5,0.3,0.2]
        hist = weight[0] * hist_h + weight[1] * hist_s + weight[2] * hist_v

        plt.plot(hist)
        plt.xlim([0, 256])

        temp_frame.hist = hist

        if len(myFrames) >= 2:
            sd = 0
            last_Frame = myFrames[len(myFrames) - 2]
            last_hist = last_Frame.hist

            for i in range(0, 256):
                sd += abs(hist[i]-last_hist[i])

            last_Frame.SD = sd
            list_sd.append(sd)
            # print(index, sd)

            if sd >= Tb: # 帧间差大于Tb, 是突变
                num += 1
                last_Frame.isCut = True
                isbegin=False
            elif sd >= Ts:# 帧间差大于Ts，小于Tb，是可能的渐变起点
                Fs = last_Frame.index
                isbegin = True  # 进入渐变过程
                # print("Fs", Fs)
            elif sd < Ts:
                if isbegin == True: # 如果处于渐变过程
                    diffsd= 0
                    for i in range(0, 256): # 累积差
                        diffsd += abs(last_Frame.hist[i] - myFrames[Fs].hist[i])
                    if diffsd>= Tb: # 累积差超过Tb，算作渐变结束点
                        Fe = last_Frame.index
                        last_Frame.isCut= True
                        isbegin= False
                        print("Fe", Fe)
                    elif last_Frame.index - Fs > step: # 帧间差小于Ts，累积差小于Tb，步长超过渐变的范围，放弃渐变开始点
                        isbegin = False

    cap.release()
    plt.show()

    print(np.average(list_sd))
    print(np.max(list_sd))
    print(num)

    for i in range (0,len(myFrames)-1):
        if myFrames[i].isCut==True:
            print(myFrames[i].index)
            cv2.imwrite(result_path+"/" + str(myFrames[i].index) + ".png", myFrames[i].img)


ColorHistTwinComparisonDetect("movie.mp4")

