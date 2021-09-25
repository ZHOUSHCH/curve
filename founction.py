import cv2
import os
import numpy as np
import tips.rec_bd as rec_bd
import openpyxl
from PIL import Image, ImageDraw, ImageFont


def lines2line(pointsx, points1, barx, bary):
    min_xlines = []
    min_ylines = []
    for line in points1:
        x1, y1, x2, y2 = line
        if len(min_ylines) == 0:
            min_ylines.append((int((x1 + x2) / 2), min(y1, y2), int((x1 + x2) / 2), max(y1, y2)))
        elif abs(min_ylines[-1][0] - x1) < bary:
            min_ylines.append((int((x1 + x2) / 4 + (min_ylines[-1][0] + min_ylines[-1][2]) / 4),
                               min(y1, y2, min_ylines[-1][1], min_ylines[-1][3]),
                               int((x1 + x2) / 4 + (min_ylines[-1][0] + min_ylines[-1][2]) / 4),
                               max(y1, y2, min_ylines[-1][1], min_ylines[-1][3])))
            del min_ylines[0]
        elif abs(min_ylines[-1][0] - x1) > bary and (min_ylines[-1][0] - x1) > 0 and abs(y2 - y1) > 4 * bary:
            min_ylines.clear()
            min_ylines.append((int((x1 + x2) / 2), min(y1, y2), int((x1 + x2) / 2), max(y1, y2)))
    for line in pointsx:
        x1, y1, x2, y2 = line

        if len(min_xlines) == 0:
            if abs(x2 - x1) > barx:  # 没有初始线赋值
                min_xlines.append((min(x1, x2), int((y1 + y2) / 2), max(x1, x2), int((y1 + y2) / 2)))
            else:
                continue

        elif abs(min_xlines[-1][1] - y1) < barx:
            min_xlines.append((min(x1, x2, min_xlines[-1][0], min_xlines[-1][2]),
                               int((y1 + y2) / 4 + (min_xlines[-1][1] + min_xlines[-1][3]) / 4),
                               max(x1, x2, min_xlines[-1][0], min_xlines[-1][2]),
                               int((y1 + y2) / 4 + (min_xlines[-1][1] + min_xlines[-1][3]) / 4)))
            del min_xlines[0]
        elif abs(min_xlines[-1][1] - y1) > barx and (min_xlines[-1][1] - y1) < 0 and abs(x2 - x1) > 8 * barx:
            min_xlines.clear()
            min_xlines.append((min(x1, x2), int((y1 + y2) / 2), max(x1, x2), int((y1 + y2) / 2)))
    if len(min_ylines) > 0:
        return [min_ylines[0][0], min_xlines[0][1], min_xlines[0][2], min_xlines[0][3]], [
            min_ylines[0][0], min_ylines[0][1], min_ylines[0][2], min_xlines[0][3]]
    else:
        return None, None
def rec_axisposit(historm,img_ori):
    histormsum = [0] * len(historm)
    for i, ii in enumerate(historm):
        if i - 3 > 0:
            s = i - 3
        else:
            s = 0
        if i + 3 < img_ori.shape[1]:
            e = i + 3
        else:
            e = img_ori.shape[1]
        histormsum[i] = sum(historm[s:e]) + 5 * historm[i]
    len_histormsum = len(historm)
    halflen_histormsum = int(len_histormsum / 2)
    fhalf_histormsum = histormsum[:halflen_histormsum]
    bhalf_histormsum = histormsum[halflen_histormsum:]
    max_fhalf_hist = max(fhalf_histormsum)
    max_bhalf_hist = max(bhalf_histormsum)

    maxp_fhalf_hist = fhalf_histormsum.index(max(fhalf_histormsum))
    maxp_bhalf_hist = bhalf_histormsum.index(max(bhalf_histormsum)) + halflen_histormsum
    # print(max_fhalf_hist , max_bhalf_hist)
    if max_fhalf_hist + max_bhalf_hist != 0:
        deg_conf = 2 * min(max_fhalf_hist, max_bhalf_hist) / (max_fhalf_hist + max_bhalf_hist)
    else:
        deg_conf = None

    return (max_fhalf_hist, maxp_fhalf_hist), (max_bhalf_hist, maxp_bhalf_hist), deg_conf
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

def line_detect_possible_demo(image):
    debug = 0
    points = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    th = cv2.erode(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=2)
    # cv2.imshow("2", th)
    gray = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=2)
    # cv2.imshow("gray",gray)
    imgn = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3
    # edges = cv2.Canny(gray, 10, 250, apertureSize=3)  # apertureSize参数默认其实就是3
    # cv2.imshow(" edges",  edges)
    # cv2.waitKey(0)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=30, maxLineGap=5)
    # lines = cv2.HoughLinesP(gray, 1, np.pi / 360, 90, minLineLength=60, maxLineGap=5)
    if type(lines)!= type(None):
        for line in lines:
            # print(line[0])
            x1, y1, x2, y2 = line[0]
            if (image.shape[0] - y1) > 10:
                if debug:
                    cv2.line(imgn, (x1, y1), (x2, y2), (0, 255, 255), 2)
                points.append((x1, y1,x2, y2))
    if debug:
        cv2.imshow("imgn",imgn)
        cv2.waitKey(0)
    return points

def dropShadow(image):
    debug = 0
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, img1 = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    ret2, img2 = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    xStart = []
    yStart = []
    xEnd = []
    yEnd = []
    (h, w) = img1.shape
    # 初始化一个跟图像宽一样长度的数组，用于记录每一列的黑点个数
    a = [0 for z in range(0, w)]
    for i in range(0, w):  # 遍历每一列
        for j in range(0, h):  # 遍历每一行
            if img1[j, i] == 0:  # 判断该点是否为黑点，0代表是黑点
                a[i] += 1  # 该列的计数器加1
                img1[j, i] = 255  # 记录完后将其变为白色，即等于255
    for i in range(0, w):  # 遍历每一列
        if 2*i< w and 2*a[i]>h:
            xStart.append(i)
        elif 2*i > w and 2*a[i]>h:
            xEnd.append(i)
        # print(a[i], h)
        # for j in range(h - a[i], h):  # 从该列应该变黑的最顶部的开始向最底部设为黑点
        #     img1[j, i] = 0  # 设为黑点
    # cv2.imshow("img1", img1)
    a = [0 for z in range(0, h)]
    for i in range(0, h):  # 遍历每一行
        for j in range(0, w):  # 遍历每一列
            if img2[i, j] == 0:  # 判断该点是否为黑点，0代表黑点
                a[i] += 1  # 该行的计数器加一
                img2[i, j] = 255  # 将其改为白点，即等于255
    for i in range(0, h):  # 遍历每一行
        if 2*i < h and 2*a[i]>w:
            yStart.append(i)
        elif 2*i > h and 2*a[i]>w:
            yEnd.append(i)
    # print(yStart)
    pointStatX = int(sum(xStart)/len(xStart)) if len(xStart) else 0
    pointStatY = int(sum(yStart) / len(yStart))if len(yStart) else 0
    pointEndX = int(sum(xEnd) / len(xEnd)) if len(xEnd) else 0
    pointEndY = int(sum(yEnd) / len(yEnd)) if len(yEnd) else 0
    cv2.line(img1, (pointStatX,pointStatY), (pointEndX,pointEndY), (0, 255, 255), 2)
    min_xlines, min_ylines = [pointStatX, pointEndY, pointEndX, pointEndY], [pointStatX, pointStatY, pointStatX, pointEndY]
    if debug:
        if min_xlines != None:
            x1, y1, x2, y2 = min_xlines
            cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 155), 4)
        if min_ylines != None:
            x1, y1, x2, y2 = min_ylines
            cv2.line(img1, (x1, y1), (x2, y2), (0, 155, 0), 4)
        cv2.imshow("img1",img1)
        print(min_xlines, min_ylines,)
        cv2.waitKey(0)
    return min_xlines, min_ylines

def img_redraw(size, x_gap, y_gap, x_stanub, x_endnub, y_stanub, y_endnub,xaxis_gap,yaxis_gap,x_axvalue,txtx_str,txty_str):

    def imgdrawtxt(img,txt,colour,posetion):
        cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
        pilimg = Image.fromarray(cv2img)
        # PIL图片上打印汉字
        draw = ImageDraw.Draw(pilimg)  # 图片上打印
        font = ImageFont.truetype("simhei.ttf", 30, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
        draw.text(posetion, txt, colour, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
        # PIL图片转cv2 图片
        return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    # 使用Numpy创建一张A4(2105×1487)纸
    img = np.zeros((size[0], size[1], 3), np.uint8)
    # 使用白色填充图片区域,默认为黑色
    img.fill(255)
    img = cv2.rectangle(img, (int(xaxis_gap[2]), int(xaxis_gap[3])), (int(yaxis_gap[0]), int(yaxis_gap[1])), (150, 0, 0), 2)
    txt = (x_endnub - x_stanub) / 5
    txty = (y_endnub - y_stanub) / 5
    for i in range(6):

        cv2.line(img, (int(x_gap[0]), int(y_gap[0]) + int(i * (y_gap[1] - y_gap[0]) / 5)),
                 (int(x_gap[0]) + 10, int(y_gap[0]) + int(i * (y_gap[1] - y_gap[0]) / 5)), (0, 15, 150), 2)
        cv2.line(img, (int(x_gap[0]), int(y_gap[0]) + int((i + 0.5) * (y_gap[1] - y_gap[0]) / 5)),
                 (int(x_gap[0]) + 5,  int(y_gap[0]) + int((i + 0.5) * (y_gap[1] - y_gap[0]) / 5)), (0, 15, 150), 1)


        if y_gap[0] >= int(yaxis_gap[0]):
            saux_x = y_gap[0]
        else:
            saux_x = int(yaxis_gap[0])

        cv2.line(img, (saux_x + int((i - 0.5) * (x_gap[1] - saux_x) / 5), int(xaxis_gap[1])),
                 (saux_x + int((i - 0.5) * (x_gap[1] -saux_x) / 5), int(xaxis_gap[1]) - 5), (0, 15, 150),1)
        cv2.line(img, (saux_x + i * int((x_gap[1] - saux_x) / 5), int(xaxis_gap[1])),
                 (saux_x + i * int((x_gap[1] - saux_x) / 5), int(xaxis_gap[1]) - 10), (0, 15, 150), 2)
        if len(x_axvalue):
            img = cv2.putText(img, str(float(i * txt) + x_stanub),
                          (int(list(x_axvalue.keys())[0]) - 10 + i * int((x_gap[1] - list(x_axvalue.keys())[0]) / 5), int(xaxis_gap[1]) + 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)

        if txty != 0:
            if txty > 10:
                x_axnum_lab = str(int((5 - i) * txty + y_stanub))
            elif len(str(float((5 - i) * txty + y_stanub))) > 5 and (float((5 - i) * txty + y_stanub) >10000 and float((5 - i) * txty + y_stanub) < 0.0001):
                x_axnum_lab = str("%e"%float((5 - i) * txty + y_stanub))#判断是否用科学计数法
            elif len(str(float((5 - i) * txty + y_stanub))) > 5 and abs(float("{:.2f}".format(float((5 - i) * txty + y_stanub)))-float((5 - i) * txty + y_stanub))/float((5 - i) * txty + y_stanub) < 0.01:
                x_axnum_lab = str("{:.2f}".format(float((5 - i) * txty + y_stanub)))#判断是否精确小数后两位
            else:
                x_axnum_lab = str(float((5 - i) * txty + y_stanub))

            img = cv2.putText(img, x_axnum_lab, (int(x_gap[0]/2) , int(y_gap[0]) + int(i * (y_gap[1] - y_gap[0]) / 5)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)

    if len(txtx_str) > 0:
        # img = cv2.putText(img, txtx_str,(int((x_gap[1] + x_gap[0])/2), int(y_gap[1]) + 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 25, 0), 3)
        txtx_str = txtx_str.replace("", "").replace("/n", "").replace("nf", "")
        print(yaxis_gap[3],size[0])
        img = imgdrawtxt(img, txtx_str , (0, 25, 0), (int((x_gap[1] + x_gap[0])/2), (int(yaxis_gap[3]) + size[0])/2))
    if txty_str and len(txty_str) > 0:
        txty_str = txty_str.replace("", "").replace("/n", "").replace("nf", "")
        # img = cv2.putText(img, txty_str, (int(x_gap[0]/2), int(y_gap[1]/2)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 25, 0), 3)
        img = imgdrawtxt(img, txty_str, (0, 25, 0), (int(x_gap[0]/2)-10, (int(yaxis_gap[1])+int(yaxis_gap[-1]))/2-20))

    return img

def redress_arithmetic(prog):
    def is_arithmetic_progression(prog):
        _len = len(prog)
        for i in range(0, _len - 2):
            if prog[i + 1] - prog[i] != prog[i + 2] - prog[i + 1]:
                return False
        return True
    if len(prog) > 3:
        positive_l = 0
        negative_l = 0
        positive_r = 0
        negative_r = 0
        for i, ii in enumerate(prog):
            if ii == 0:
                zero_position = i

                for j in range(zero_position):
                    if prog[j] < 0:
                        negative_l += 1
                    else:
                        positive_l += 1
                for j in range(zero_position + 1, len(prog)):
                    if prog[j] < 0:
                        negative_r += 1
                    else:
                        positive_r += 1
                if (negative_l + positive_l) != 0 and (negative_r + positive_r) != 0:
                    p_l = positive_l / (negative_l + positive_l)
                    p_r = positive_r / (negative_r + positive_r)
                    if p_l > 0.8 and p_l > p_r:
                        for j in range(zero_position):
                            if prog[j] < 0:
                                prog[j] = -prog[j]
                        for j in range(zero_position + 1, len(prog)):
                            if prog[j] > 0:
                                prog[j] = -prog[j]
                    elif p_r > 0.8 and p_r > p_l:
                        for j in range(zero_position):
                            if prog[j] > 0:
                                prog[j] = -prog[j]
                        for j in range(zero_position + 1, len(prog)):
                            if prog[j] < 0:
                                prog[j] = -prog[j]
                    break
                else:
                    break


        for i in range(len(prog)-2):
            if is_arithmetic_progression(prog[i:i+3]):#子序列为等差序列
                othersright_numb = 0

                for j in range(i):
                    if prog[j] == prog[i] - (i-j) * (prog[i+1]-prog[i]):
                        othersright_numb += 1

                for k in range(i+3, len(prog)):

                    if prog[k] == prog[i+2] + (k-i-2) * (prog[i + 1] - prog[i]):
                        othersright_numb += 1

                if othersright_numb/(len(prog)-3) > 0.5:

                    for j in range(i):
                        prog[j] = prog[i] - (i-j) * (prog[i+1]-prog[i])
                    for k in range(i + 3, len(prog)):
                        prog[k] = prog[i + 2] + (k - i - 2) * (prog[i + 1] - prog[i])


                    break


        return prog

    else:
        return prog

def points2points(points, pix,img_ori):
    debug = 0
    points1 = []
    pointsx = []
    for point in points:
        if point[2] - point[0] != 0:
            tan_angle = abs((point[3] - point[1]) / (point[2] - point[0]))
            if tan_angle < 1 / pix:
                pointsx.append(point)
            elif tan_angle > pix:
                points1.append(point)
        else:
            points1.append(point)
    img_ori2 = np.zeros((img_ori.shape[0], img_ori.shape[1], 3), np.uint8)
    img_ori2[:, :, 0] = 255
    img_ori2[:, :, 1] = 255
    img_ori2[:, :, 2] = 255
    Xhistorm = [0] * img_ori.shape[1]
    Yhistorm = [0] * img_ori.shape[0]
    #################################
    bary = img_ori.shape[0] / 20
    barx = img_ori.shape[1] / 20

    for line in points1:
        x1, y1, x2, y2 = line
        cv2.line(img_ori2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for j in range(min(x1, x2), max(x1, x2)+1):
            Xhistorm[j] = Xhistorm[j] + int((abs(y1 - y2)+1)/(abs(x1 - x2)+1))


    for line in pointsx:
        x1, y1, x2, y2 = line
        cv2.line(img_ori2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for j in range(min(y1, y2), max(y1, y2) + 1):
            Yhistorm[j] = Yhistorm[j] + int((abs(x1 - x2)+1)/(abs(y1 - y2)+1))

    ###############################################
    (_, xo), (_, xe), deg_conf_x = rec_axisposit(Xhistorm,img_ori)
    (_, yo), (_, ye), deg_conf_y = rec_axisposit(Yhistorm,img_ori)



    if (deg_conf_x != None ) and deg_conf_x > 0.5 and deg_conf_y > 0.5:
        min_xlines, min_ylines = (xo, ye, xe, ye), (xo, yo, xo, ye)
    else:

        min_xlines, min_ylines = lines2line(pointsx, points1, barx, bary)
        if type(min_ylines)!=type(None):
            min_ylines[1] = int(8*img_ori.shape[0]/10) if min_ylines[1] < int(img_ori.shape[0]/2) else min_ylines[1]
    #--------------debug------------
    if debug:
        if min_xlines != None:
            x1, y1, x2, y2 = min_xlines
            # cv2.line(img_ori2, (x1, y1), (x2, y2), (0, 0, 155), 4)
        if min_ylines != None:
            x1, y1, x2, y2 = min_ylines
            cv2.line(img_ori2, (x1, y1), (x2, y2), (0, 155, 250), 4)
        cv2.imshow("img_ori2",img_ori2)
        print(min_xlines, min_ylines,)
        cv2.waitKey(0)
    return min_xlines, min_ylines

def verticalx(frame):
    h = len(frame)
    w = len(frame[0])
    pixdata = frame
    x_array = []
    startX = 0
    endX = 0
    for y in range(h):
        b_count = 0
        for x in range(w):
            if pixdata[y, x] < 150:
                b_count += 1

        if b_count > 0:
            if startX == 0:
                startX = y
        elif b_count == 0:
            if startX != 0:
                if y - startX > 1:
                    endX = y
                    x_array.append({'startX': startX, 'endX': endX})
                startX = 0
                endX = 0

    return x_array

def verticaly(frame):
    h = len(frame)
    w = len(frame[0])

    pixdata = frame
    x_array = []
    startX = 0
    endX = 0
    allVal= 0#竖直投影的总和
    aveVal = 0#竖直投影的平均值
    pixdataMean = pixdata.mean()
    for y in range(0,w, 1):
        b_count = 0
        for x in range(0, h, 1):
            if pixdata[x, y] < pixdataMean:
                b_count += 1

        allVal = allVal + b_count    #计算投影总和
        aveVal = allVal / (y+1)     #计算投影平均值
        if b_count > h/2 and h > w: #投影值过大可能是坐标轴
            b_count = 0
        if b_count > aveVal/3:#aveVal/3
            if startX == 0:
                startX = y
        elif b_count < aveVal/4:#aveVal/4
            if startX != 0:
                if y - startX > 1:
                    endX = y
                    x_array.append({'startX': startX, 'endX': endX})
                startX = 0
                endX = 0

    return x_array

def cutimg_x(y_img):

    gray = cv2.cvtColor(y_img, cv2.COLOR_BGR2GRAY)
    y_array = verticalx(gray)
    kernel = np.ones((15, 10), np.uint8)
    if len(y_array):
        if y_array[0]['startX'] < 2 and len(y_array) >= 2:
            if len(y_array) > 2:
                pixdata = cv2.erode(gray[y_array[2]['startX']:y_array[2]['endX'], :], kernel)
                remark = (y_array[2]['startX'], y_array[2]['endX'], verticaly(pixdata))
            else:
                remark = None

            y0 = y_array[1]['startX']
            y1 = y_array[1]['endX']
        else:
            if len(y_array) > 1:
                pixdata = cv2.erode(gray[y_array[1]['startX']:y_array[1]['endX'], :], kernel)
                remark = (y_array[1]['startX'], y_array[1]['endX'], verticaly(pixdata))
            else:
                remark = None
            y0 = y_array[0]['startX']
            y1 = y_array[0]['endX']

        pixdata = cv2.erode(gray[y0:y1, :], kernel)
        x_array = verticaly(pixdata[int(len(pixdata)/4): int(3 * len(pixdata)/4)])
        for poin_y in x_array:
            cv2.line(y_img, (poin_y["startX"], 10), (poin_y["endX"], 10), (0, 255, 0), 1)  # 绿色，3个像素宽度
        return y0, y1, x_array,remark
    else:
        return 0, 0, 0, 0
def cutimg_y(y_img):
    # cv2.imshow("y_img",y_img)
    # cv2.waitKey(0)
    debug = 0
    #=======图像预处理，膨胀降噪============
    gray = cv2.cvtColor(y_img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((8, 8), np.uint8)
    pixdata = cv2.erode(gray[int(gray.shape[0] / 5):4 * int(gray.shape[0] /5), :], kernel)
    # =======end============
    # =======确定竖直切割的起始点============
    y_array = verticaly(pixdata)
    if  len(y_array) > 1 and gray.shape[1] - y_array[-1]['endX'] < 1:#最后第一个靠近边缘为噪点
       del y_array[-1]
    if len(y_array) > 1:        #大于一个包含备注名称
        remarky0 = y_array[-2]['startX']  #备注的x轴起始位置
        remarky1 = y_array[-2]['endX']  # 备注的x轴起始位置
        y0 = y_array[-1]['startX'] #刻度值的x轴起始位置
        y1 = y_array[-1]['endX']
    else:
        remarky0 = 0  # 备注的x轴起始位置
        remarky1 = 0  # 备注的x轴起始位置
        y0 = 0
        y1 = gray.shape[1]
    imgYNum = gray[:, y0:y1]
    imgYLab = gray[:, remarky0:remarky1]

    x_array = verticalx(imgYNum)                 #切割刻度值
    x_array2 = verticalx(imgYLab)    # 切割备注竖直位置
    # _____________debug____________
    # 1.绿色线条为竖直切割起始点
    # 2.红色线条为水平切割起始点 刻度
    # 3.蓝色线条为水平切割起始点 备注
    if debug:
        for poin_y in y_array:
            cv2.line(y_img, (poin_y["startX"], 100), (poin_y["endX"], 100), (0, 255, 0), 1)  # 绿色，1个像素宽度
        for poin_x in x_array:
            cv2.line(y_img, (10, poin_x["startX"]), (10, poin_x["endX"]), (0, 0, 255), 1)  # 红色，1个像素宽度
        for poin_x in x_array2:
            cv2.line(y_img, (40, poin_x["startX"]), (40, poin_x["endX"]), (255, 0, 0), 1)  # 蓝色色，1个像素宽度
        print(x_array2)
        print(remarky0, remarky1)
        cv2.imshow("y_img", y_img)
        cv2.imshow("imgYNum", imgYNum)
        cv2.imshow("imgYLab", imgYLab)

        cv2.waitKey(0)
    # _____________debug____________
    if x_array2:
        return y0, y1, x_array, (remarky0, remarky1, x_array2[0]["startX"], x_array2[-1]["endX"])
    else:
        return y0, y1, x_array, (remarky0, remarky1, 0, 0)

def tracker_line(img, x_star, x_end, y_star, y_end):
    def distanse(p1,p2):
        #dis = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def getdiff(img):
        # 定义边长
        Sidelength = 30
        # 缩放图像
        img = cv2.resize(img, (Sidelength, Sidelength), interpolation=cv2.INTER_CUBIC)
        # 灰度处理
        gray = img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # avglist列表保存每行像素平均值
        avglist = []
        # 计算每行均值，保存到avglist列表
        for i in range(Sidelength):
            avg = sum(gray[i]) / len(gray[i])
            avglist.append(avg)
        # 返回avglist平均值
        return sum(avglist) / Sidelength
    debug = 0
    bar = 4
    line_wide = 0
    splashes = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean = int(getdiff(gray))
    # print("----mean0-----", mean)
    # print("----mean1-----", mean)
    # print("----mean2-----", 2*mean- 256)
    # print("----mean3-----", 3*mean- 512)
    # print("----mean4-----", 4*mean- 768)
    mean = mean- 60 if mean > 250 else 3*mean- 512  #compute the image mean  pix value
    ret, gray = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    tran_list = np.transpose(gray)
    # ------------debug-------------
    if debug:
        cv2.imshow("gray", gray)
        print("---------",mean)
        cv2.waitKey(0)

    # ------------debug-------------

    lines = []#[[(),()],[]...]
    for i in range(x_star, x_end):
        xs = []
        start_l = 0
        end_l = 0
        for j in range(y_star+2, y_end-2):
            if sum(tran_list[i][j - 2:j]) > sum(tran_list[i][j:j + 2]) and start_l == 0 and end_l == 0:
                start_l = j

            elif sum(tran_list[i][j - 2:j]) < sum(tran_list[i][j:j + 2]) and start_l != 0 and end_l == 0:
                end_l = j

            if start_l != 0 and end_l != 0:
                line_wide = (abs(end_l-start_l)+line_wide)/2
                bar = 2 * line_wide
                xs.append(int((start_l+end_l)/2))
                start_l = 0
                end_l = 0
        if len(xs) > 0:#检测到 点
            for point in xs:
                if len(lines) == 0:#no lines point
                    lines.append([(i, point)])
                else:
                    distanses = []
                    for num, line_point in enumerate(lines):
                        distanses.append(distanse(line_point[-1], (i, point)))
                    if min(distanses) <= 2*bar:
                        if min(distanses) < bar:
                            lines[distanses.index(min(distanses))].append((i, point))
                        else:
                            splashes.append((i, point))#
                    elif min(distanses) > 2*bar: #new line
                        lines.append([(i, point)])

                    else:
                        splashes.append((i, point))
                    for splash_id,splash in enumerate(splashes):
                        if splash_id == 0:
                            continue
                        elif distanse(splashes[splash_id-1], splash) < bar:
                            lines.append([splashes[splash_id-1], splash])

                            del splashes[splash_id], splashes[splash_id-1]
                            break
    imaginary_line = []
    delima_linenum = []
    lenps_ave = None
    for i, points in enumerate(lines):#虚线连接
        if len(points) < abs(x_end-x_star) / 20:
            if lenps_ave == None:
                lenps_ave = len(points)
            else:
                lenps_ave = (lenps_ave + len(points)) / 2
            if abs(len(points)/lenps_ave) > 0.8:
                imaginary_line += points
                delima_linenum.append(i)
    for i in delima_linenum[::-1]:
        del lines[i]
    lines.append(imaginary_line)
    #########################################
    linesnubs=[]
    tureline = []
    turestline = []
    for line in lines:
        linesnubs.append(len(line))

    for i, line in enumerate(lines):
        if len(line) > max(linesnubs) / 10:
            tureline.append(line)
            if len(line) > max(linesnubs) / 4:
                turestline.append(line)
        else:
            for j in line:
                splashes.append(j)


    #######################################~

    return splashes, tureline,turestline

def read_xy(y0, y1, x_array, y_array,img_ori,min_xlines,y_x0,y_x1):
    def custom_threshold(image):  # 平均值二值化
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
        h, w = gray.shape[:2]
        m = np.reshape(gray, [1, w * h])
        mean = m.sum() / (2 * w * h)
        ret, binary = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
        return binary
    mkdir("cacheimg")
    i = 0
    x_axvalue = {}
    txt_xstr = ""
    for c in x_array:
        i = i + 1
        x0, x1 = c["startX"], c["endX"]
        XIMG = img_ori[min_xlines[1] + y0 - 3:min_xlines[1] + y1 + 3, x0:x1]
        if x1 - x0 < min_xlines[1] + y1 + 3 - min_xlines[1] + y0 - 3:
            XIMG = cv2.resize(XIMG, (x1 - x0, x1 - x0))
        # XIMGkc = cv2.copyMakeBorder(XIMG, 50, 50, 50, 50, cv2.BORDER_REPLICATE)#图像扩充
        XIMGkc = cv2.copyMakeBorder(XIMG, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])  # 图像扩充
        cv2.imwrite("cacheimg/X" + str(i) + ".jpg", XIMGkc)
        txt = rec_bd.ocr_bd2txt("cacheimg/X" + str(i) + ".jpg").replace("","")
        # print("test bdx",i,txt)
        if txt != None:
            try:
                x_axvalue[int((x0 + x1) / 2)] = float(txt)
            except:
                txt_xstr += txt

    i = 0
    y_axvalue = {}
    for c in y_array:
        i = i + 1
        y0, y1 = c["startX"], c["endX"]
        if y_x0 - 3 > 0:#扩大显示区域
            y_x0 = y_x0 - 3
        else:
            y_x0 = 0
        if y0 - 3 > 0:  # 扩大显示区域
            y0 = y0 - 3
        else:
            y0 = 0
        if y0 < min_xlines[1]:#y轴数值高于x轴（y最低端）
            imgNum =img_ori[y0:y1 + 3, y_x0:y_x1 + 0] #y轴数字图片
            # XIMGkc = cv2.copyMakeBorder(imgNum, 50, 50, 50, 50, cv2.BORDER_REPLICATE)#图像扩充
            XIMGkc = cv2.copyMakeBorder(imgNum, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])  # 图像扩充
            cv2.imwrite("cacheimg/Y" + str(i) + ".jpg", XIMGkc)
            if rec_bd.detsci_not("cacheimg/Y" + str(i) + ".jpg"):
                txt_rea = rec_bd.ocr_bd2txt("cacheimg/Y" + str(i) + ".jpg")
                txt_exp = rec_bd.ocr_bd2txt("cacheimg/Y" + str(i) + ".jpg")
                if txt_rea == None:
                    txt_rea = 1
                if txt_exp == None:
                    txt_exp = 1
                try:
                    txt = pow(float(txt_rea),float(txt_exp))
                except:
                    print("ocr sci num err")
            else:
                txt = rec_bd.ocr_bd2txt("cacheimg/Y" + str(i) + ".jpg").replace("","").replace("—","").replace("-","")
                # print( c,"test :bdy", txt)


            if txt != None:
                try:
                    y_axvalue[int((y0 + y1) / 2)] = float(txt)
                except:
                    pass


    return x_axvalue, y_axvalue,txt_xstr

def read_ytxt(img_ori):
    mkdir("cacheimg")
    img = cv2.transpose(img_ori)
    img = cv2.flip(img, 1)
    # img = np.rot90(img)
    # XIMGkc = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_REPLICATE)  # 图像扩充
    XIMGkc = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255]) # 图像扩充
    cv2.imwrite("cacheimg/ylab.jpg", XIMGkc)
    txt = rec_bd.ocr_bd2txt("cacheimg/ylab.jpg")
    return txt

def pix2txt (points, xs, ys, savename="cache.xlsx"):#(x start value,x end value,x_startpix,x_endpix)
    if savename.split(".")[-1] == "xlsx":
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet['A1'] = "X-axis"
        sheet['B1'] = "Y-axis"
    else:
        # savename_n = savename.replace(savename.split(".")[-1], 'txt')

        f = open(savename, 'w')
        y = ""
        for i in range(len(points)-1):
            y += "     |   y{:}   |   ".format(i+1)
        f.write("x  "+y+"  \n")
    xyps = {}
    xps = []
    yps = []
    for i, point in enumerate(points):
        xps = []
        yps = []
        if type(point) != type(1):
            for xy in point:

                if (float(xs[3]) - float(xs[2]))!= 0 and (float(ys[3]) - float(ys[2]))!= 0:
                    x = xy[0] / (float(xs[3]) - float(xs[2])) * (float(xs[1]) - float(xs[0])) + float(xs[0])
                    y = xy[1] / (float(ys[3]) - float(ys[2])) * (float(ys[1]) - float(ys[0])) + float(ys[0])
                    xps.append(x)
                    if y == ys[0]:
                        yps.append(None)
                    else:
                        yps.append(y)

                    if savename.split(".")[-1] == "xlsx":
                        sheet['A' + str(i + 2)] = x
                        sheet['B' + str(i + 2)] = y
                    else:
                        if x in xyps.keys():
                            xyps[x] += [y,]
                        else:
                            spaces = []
                            for space in range(i-1):
                                spaces += [None, ]
                            xyps[x] =spaces + [y, ]
    if savename.split(".")[-1] == "xlsx":
        workbook.save(savename)
    else:


        f.write("————————————\n")
        for i, v in enumerate(xyps):
            stri = ""
            for j, jj in enumerate(xyps[v]):
                if jj != None:
                    stri = stri + "{:.4f}  ".format(jj)
                else:
                    stri = stri + "NONE  "
            f.write("{:.2f}  ".format(v) + stri + "\n")

        f.close()
        ##############################################
    return xps, yps
def pix2txt2 (points, xs, ys, savename="cache.xlsx"):#(x start value,x end value,x_startpix,x_endpix)



    if savename.split(".")[-1] == "xlsx":
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet['A1'] = "X-axis"
        sheet['B1'] = "Y-axis"
    else:
        # savename_n = savename.replace(savename.split(".")[-1], 'txt')
        savename_n = savename+".txt" if savename[-4:]!= ".txt" else savename
        f = open(savename_n, 'w')
        y = ""
        for i in range(len(points)-1):
            y += "     |   y{:}   |   ".format(i+1)
        f.write("x  "+y+"  \n")
    xyps = {}
    xps = []
    yps = []
    for i, point in enumerate(points):
        xps = []
        yps = []
        if type(point) != type(1):
            for xy in point:
                if (float(xs[3]) - float(xs[2])) != 0 and (float(ys[3]) - float(ys[2])) != 0:
                    x = xy[0] / (float(xs[3]) - float(xs[2])) * (float(xs[1]) - float(xs[0])) + float(xs[0])
                    y = xy[1] / (float(ys[3]) - float(ys[2])) * (float(ys[1]) - float(ys[0])) + float(ys[0])
                    xps.append(x)
                    if y == ys[0]:
                        yps.append(None)
                    else:
                        yps.append(y)

                    if savename.split(".")[-1] == "xlsx":
                        sheet['A' + str(i + 2)] = x
                        sheet['B' + str(i + 2)] = y
                    else:
                        if x in xyps.keys():
                            xyps[x] += [y,]
                        else:
                            spaces = []
                            for space in range(i-1):
                                spaces += [None, ]
                            xyps[x] =spaces + [y, ]
    if savename.split(".")[-1] == "xlsx":
        workbook.save(savename)
    else:


        f.write("————————————\n")
        for i, v in enumerate(xyps):
            stri = ""
            for j, jj in enumerate(xyps[v]):
                if jj != None:
                    stri = stri + "{:.4f}  ".format(jj)

            f.write("{:.2f}  ".format(v) + stri + "\n")

        f.close()
        ##############################################

    return xps, yps
if __name__ =="__main__":
    imgpath = r"E:\zhou\scienceai\g40.png"
    img = cv2.imread(imgpath)
    line_detect_possible_demo2(img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    def threshold_demo(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 要二值化图像，要先进行灰度化处理
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary
    gray = threshold_demo(img)

    cv2.imshow("img",gray)

    cv2.waitKey(0)
