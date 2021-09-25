import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon  # 多边形

def Cal_area_2poly(data1, data2):
    """
    任意两个图形的相交面积的计算
    :param data1: 当前物体
    :param data2: 待比较的物体
    :return: 当前物体与待比较的物体的面积交集
    """

    poly1 = Polygon(data1).convex_hull  # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area
def compute_polygon_area(points):
    point_num = len(points)
    if(point_num < 3): return 0.0
    s = points[0][1] * (points[point_num-1][0] - points[1][0])
    #for i in range(point_num): # (int i = 1 i < point_num ++i):
    for i in range(1, point_num): # 有小伙伴发现一个bug，这里做了修改，但是没有测试，需要使用的亲请测试下，以免结果不正确。
        s += points[i][1] * (points[i-1][0] - points[(i+1) % point_num][0])
    return abs(s/2.0)

def compute(thresh,img):
    cv2.bitwise_not(thresh, thresh)#反相
    kernell = np.ones((5, 5), np.uint8)
    kernelb = np.ones((15, 15), np.uint8)
    ## c.图像的腐蚀，默认迭代次数
    erosion = cv2.erode(thresh, kernelb)#预处理
    dst = cv2.dilate(erosion, kernell)
    dst = cv2.dilate(dst, kernell)
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    res = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    return res, contours


def computeAres(contours):
    Contours_tuple = []
    for i in contours:
        cache = []
        for j in list(i.tolist()):
            # print(tuple(j[0]), end="")
            cache.append(tuple(j[0]))
        Contours_tuple.append(cache)
        # print("")
    include_All = 0
    for i in range(len(Contours_tuple)):
        for j in range(i):
            data1 = Contours_tuple[i]  # 带比较的第一个物体的顶点坐标
            data2 = Contours_tuple[j]  # 待比较的第二个物体的顶点坐标
            area = Cal_area_2poly(data1, data2)
            area1 = compute_polygon_area(data1)
            area2 = compute_polygon_area(data2)

            include = True if (min(area2, area1) / (area + 1)) > 0.9 and (
                        min(area2, area1) / (area + 1)) < 1.1 else False
            include_All = include_All + include
            # print(include, "编号{}和{}，面积{}和{}，相交{}".format(i, j, area1, area2, area))

    res = 1 - (include_All/len(Contours_tuple)) if len(Contours_tuple) else 1
    return res
path = "16.jpg"
threshold = 0
def scoarTrans(socar):
    if socar < 0.3:
        return 1 - ((0.3 - socar) / 0.3)
    elif socar > 0.5:
        return 1 - ((socar - 0.5) / 0.5)
    else:
        return 1

def celectTHore():
    for i in range(255):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        a = pd.DataFrame(thresh.reshape(-1, 1))
        result = a.apply(pd.value_counts)
        blackRadio = result.at[0, 0] / len(thresh.reshape(-1, 1))

        res, contours = compute(thresh)
        mps = computeAres(contours)
        threshold = threshold + 1
        blackRadio = scoarTrans(blackRadio)
        print("\r" + "阈值{}，B radio:{:.2f},scal:{:.2f}".format(threshold, blackRadio,mps), end="", flush=True)
        if mps + blackRadio == 2:
            print("over,the value is:", threshold)
            return threshold
        #     break
        # cv2.imshow("34", res)
        # cv2.imshow("12", thresh)
        # cv2.waitKey(1)
cv2.waitKey(0)