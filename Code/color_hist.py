
import cv2
from matplotlib import pyplot as plt
import numpy as np
import MyUtils

myFrames = []
list_similarity = []
list_distance = []

class Frame:
    distance= 0
    similarity= 0
    img= None
    s= None
    hist=[]
    index= -1


def ColorHistDetect(VideoName):


    result_path = "similarResult"
    # result_path2 = "distResult"
    MyUtils.makeDir(result_path)
    # MyUtils.makeDir(result_path2)
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
        if index >= 3563:
            break

        img = frame
        temp_frame = Frame()
        temp_frame.index = index
        temp_frame.img = img.copy()
        myFrames.append(temp_frame)

        #cv2.imwrite("original/" + str(index) + ".png", img)

        img = cv2.GaussianBlur(img, (9, 9), 0.0)  # 做高斯模糊
        # b, g, r = cv2.split(img)
        # hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])  # 计算直方图
        # hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])  # 计算直方图
        # hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])  # 计算直方图
        # weight= [0.33,0.33,0.33]
        # hist= weight[0]*hist_b +weight[1]* hist_g + weight[2]* hist_r

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
            s = 0
            h = 0
            x2=0
            last_Frame = myFrames[len(myFrames) - 2]
            # cv2.imshow(str(count)+"lastframe",last_Frame.img)
            last_hist = last_Frame.hist

            for i in range(0, 256):
                s += min(hist[i], last_hist[i])
                h += last_hist[i]
                temp_max= max(hist[i], last_hist[i])
                if temp_max >0:
                    x2+= ((hist[i]-last_hist[i])*(hist[i]-last_hist[i]))/ max(hist[i], last_hist[i])

            last_Frame.similarity = s / h  # 获得直方图的交，表示相似度
            last_Frame.distance=x2 #获得直方图的差异

            list_similarity.append(last_Frame.similarity)
            list_distance.append(last_Frame.distance)


    cap.release()
    plt.show()


    similarThresholdWay(list_similarity, result_path)
    #  calKmeans()


def similarThresholdWay(list_similarity,result_path):
    print(list_similarity)
    min_value = min(list_similarity)

    # threshold = 0.6
    threshold = 0.7
    print("min: ", min_value)
    num = 0

    for i in range(0, len(myFrames) - 1):
        if myFrames[i].similarity <= threshold:
            num += 1
            print(myFrames[i].index)
            cv2.imwrite(result_path + "/" + str(myFrames[i].index) + ".png", myFrames[i].img)

    print("num: ", num)



def calcAndDrawHist(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256)

    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)

    return histImg


def testHist(path):
    img = cv2.imread(path)
    b, g, r = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    histImgB = calcAndDrawHist(gray, [255, 0, 0])
    histImgG = calcAndDrawHist(g, [0, 255, 0])
    histImgR = calcAndDrawHist(r, [0, 0, 255])

    cv2.imshow(path+"histImgB", histImgB)
    cv2.imshow(path+"histImgG", histImgG)
    cv2.imshow(path+"histImgR", histImgR)
    cv2.imshow(path+"Img", img)

# testHist("MyOriginal/672.png")
# testHist("MyOriginal/673.png")
# cv2.waitKey(0)
# cv2.destroyAllWindows()

ColorHistDetect("movie.mp4")