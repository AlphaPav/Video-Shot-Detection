import cv2
import numpy as np
import MyUtils

class Frame:
    flow=None
    flowsum=-1
    mag= None
    ang= None
    index=-1
    img = None
    angdist= None
    magdist = None
    flowdist= None

def optFlowDetect(VideoName):
    result_path = "optResult"

    MyUtils.makeDir(result_path)
    MyFrames = []
    list_flowsum= []
    list_flowdist=[]

    cap = cv2.VideoCapture(VideoName)
    index =0
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    while(1):
        ret, frame2 = cap.read()
        index +=1
        print("index",index)
        if ret == False:
            break
        # if index< 300:
        #     continue
        # if index >= 1000:
        #     break
        temp_frame = Frame()
        temp_frame.index = index

        # temp_frame.img = frame2.copy()
        MyFrames.append(temp_frame)

        temp = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        # args: prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
        # 当前帧的flow
        flow = cv2.calcOpticalFlowFarneback(prvs,temp, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        temp_frame.flow = flow
        u= flow[...,0]
        v= flow[...,1]

        sum=0.0
        for i in range(0,u.shape[0]):
            for j in range(0,u.shape[1]):
                sum+= u[i][j]* u[i][j] + v[i][j]* v[i][j]
        temp_frame.flowsum= sum
        list_flowsum.append(temp_frame.flowsum)


        if len(MyFrames)>=2: # 下标从 1,2,3,4 开始算起
            last_frame= MyFrames[len(MyFrames)-2]
            # print("last_frame.index",last_frame.index)
            last_frame.flowdist = abs(temp_frame.flowsum - last_frame.flowsum)
            list_flowdist.append(last_frame.flowdist)
        # show
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # 笛卡尔坐标转极坐标cartToPolar()
        temp_frame.mag = mag
        temp_frame.ang = ang
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2',rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs = temp

    cap.release()
    cv2.destroyAllWindows()
    print("while done")
    avg  = np.mean(list_flowdist)
    print("flowdist mean", avg)

    avgflowsum=  np.mean(list_flowsum)
    print("flowsum mean" ,  avgflowsum)

    # max= np.max(list_flowdist)
    # threshold= 5000000 # 173张
    # threshold= 6000000 # 150张
    threshold = 8000000  # 132张
    num=0
    for i in range(0,len(MyFrames)-1):
        print(MyFrames[i].index, ':' , MyFrames[i].flowdist)
        if MyFrames[i].flowdist >= threshold:
            # print(MyFrames[i].index)
            print(MyFrames[i].index, "dist",  MyFrames[i].flowdist )
            img= cv2.imread("MyOriginal/"+str(MyFrames[i].index)+'.png')
            cv2.imwrite(result_path+'/'+str(MyFrames[i].index)+'.png', img)
            num+=1
    print(num)



optFlowDetect("movie.mp4")