import cv2
import numpy as np
import MyUtils

class Frame:
    distance= 0
    similarity= 0
    isCut= False
    isKeyFrame = False
    img= None
    s= None
    dilate= None
    hist=[]
    index= -1
    shot_id=-1
    ECR= 0
    hu=[]
    hu_dist=0
def humoments(img_gray):
    # 标准矩定义为m_pq = sumsum(x^p * y^q * f(x, y))
    row, col = img_gray.shape
    #计算图像的0阶几何矩
    m00 = img_gray.sum()
    m10 = m01 = 0
    #　计算图像的二阶、三阶几何矩
    m11 = m20 = m02 = m12 = m21 = m30 = m03 = 0
    for i in range(row):
        m10 += (i * img_gray[i]).sum()
        m20 += (i ** 2 * img_gray[i]).sum()
        m30 += (i ** 3 * img_gray[i]).sum()
        for j in range(col):
            m11 += i * j * img_gray[i][j]
            m12 += i * j ** 2 * img_gray[i][j]
            m21 += i ** 2 * j * img_gray[i][j]
    for j in range(col):
        m01 += (j * img_gray[:, j]).sum()
        m02 += (j ** 2 * img_gray[:, j]).sum()
        m30 += (j ** 3 * img_gray[:, j]).sum()
    # 由标准矩我们可以得到图像的"重心"
    u10 = m10 / m00
    u01 = m01 / m00
    # 计算图像的二阶中心矩、三阶中心矩
    y00 = m00
    y10 = y01 = 0
    y11 = m11 - u01 * m10
    y20 = m20 - u10 * m10
    y02 = m02 - u01 * m01
    y30 = m30 - 3 * u10 * m20 + 2 * u10 ** 2 * m10
    y12 = m12 - 2 * u01 * m11 - u10 * m02 + 2 * u01 ** 2 * m10
    y21 = m21 - 2 * u10 * m11 - u01 * m20 + 2 * u10 ** 2 * m01
    y03 = m03 - 3 * u01 * m02 + 2 * u01 ** 2 * m01
    # 计算图像的归格化中心矩
    n20 = y20 / m00 ** 2
    n02 = y02 / m00 ** 2
    n11 = y11 / m00 ** 2
    n30 = y30 / m00 ** 2.5
    n03 = y03 / m00 ** 2.5
    n12 = y12 / m00 ** 2.5
    n21 = y21 / m00 ** 2.5
    # 计算图像的七个不变矩
    h1 = n20 + n02
    h2 = (n20 - n02) ** 2 + 4 * n11 ** 2
    h3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h5 = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + (3 * n21 - n03) * (n21 + n03) \
        * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    h6 = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03)
    h7 = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + (3 * n12 - n30) * (n21 + n03) \
        * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    inv_m7 = [h1, h2, h3, h4, h5, h6, h7]
    inv_m7 = np.log(np.abs(inv_m7)) # 取对数
    return inv_m7

def humomentDetect(VideoName):
    myFrames=[]
    result_path = "humomentDetectResult"
    MyUtils.makeDir(result_path)
    cap = cv2.VideoCapture(VideoName)  # 提取视频
    index = -1
    list_hu_dist = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        index += 1
        # if index < 600:
        #     continue
        if index >= 3563:
            break
        img = frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(img_gray)
        humoments = cv2.HuMoments(moments)
        humoments = np.log(np.abs(humoments)) # 7个不变矩

        temp_frame = Frame()
        temp_frame.hu= humoments
        temp_frame.index = index
        temp_frame.img = img.copy()

        myFrames.append(temp_frame)
        if len(myFrames) >= 2:
            last_Frame = myFrames[len(myFrames) - 2]
            tempdist= (last_Frame.hu[0]- temp_frame.hu[0])**2 +(last_Frame.hu[1]- temp_frame.hu[1])**2 +(last_Frame.hu[2] - temp_frame.hu[2]) ** 2
            last_Frame.hu_dist= tempdist
            print(tempdist)
            list_hu_dist.append(tempdist)

    max_value = max(list_hu_dist)
    print("max_value",max_value)
    threshold = 10
    num = 0
    for i in range(0, len(myFrames) - 1):
        if myFrames[i].hu_dist >= threshold:
            num += 1
            print(myFrames[i].index)
            cv2.imwrite(result_path + "/" + str(myFrames[i].index) + ".png", myFrames[i].img)

    print(num)

humomentDetect("movie.mp4")