import win32con
import cv2
from founction import *
colours = [(250, 0, 0), (0, 250, 0), (0, 0, 250), (0, 250, 250), (250, 0, 250), (250, 250, 0)]

def seekaxi(img_ori,imgMess):
    points = line_detect_possible_demo(img_ori)  # 检测直线，返回图和直线数组

    if points:
        min_xlines, min_ylines = points2points(points, 50, img_ori)  # 根据返回直线，选择坐标轴直线
        if type(min_xlines) == type(None):
            min_xlines, min_ylines = dropShadow(img_ori)  # 检测直线，返回图和直线数组
    else:
        min_xlines, min_ylines = dropShadow(img_ori)  # 检测直线，返回图和直线数组
    imgMess['Xaxis'] = min_xlines
    imgMess['Yaxis'] = min_ylines
    print(imgMess)
def rec_graph(imgpath,Subject):
    imgMess = {'imgSize': (100,  100),   #画幅大小(高度，长度)
               'Xaxis': [10, 90, 90, 10],#横轴位置(起点x，起点y，终点x,终点y) cv坐标在第四象限
               'Yaxis': [90, 10, 10, 10] #纵轴位置(起点x，起点y，终点x,终点y) cv坐标在第四象限
               }
    try:
        print("read pdf img",imgpath)
        img_ori = cv2.imread(imgpath)
        origin = cv2.imread(imgpath)
        imgMess['imgSize'] = origin.shape[:2]
    except Exception as e:
        print("错误代码：001，没有图片或图片格式不正确。"+str(e), "错误！！！")

    try:
        seekaxi(img_ori, imgMess)
    except Exception as e:
        print("错误代码：002，未检测到直线。"+str(e), "错误！！！")


    try:
        imgx = img_ori[imgMess['Xaxis'][1]:, :]  # 包含x刻度，备注等信息的图片
        imgy = img_ori[:, :imgMess['Yaxis'][0]]  # 包含y刻度，备注等信息的图片
        y0, y1, x_array, remark = cutimg_x(imgx)  # x轴刻度切除
        y_x0, y_x1, y_array, remarky = cutimg_y(imgy)  # y轴刻度切除
        pass
    except Exception as e:
        print("错误代码：004，坐标轴文字检测失败。"+str(e), "错误！！！")

    try:
        if imgMess['Xaxis'] != None:
            x_axvalue, y_axvalue, _ = read_xy(y0, y1, x_array, y_array, origin, imgMess['Xaxis'], y_x0, y_x1)  # 返回xy坐标轴数值
    except Exception as e:
        print("错误代码：005.1，坐标轴文字识别失败。"+str(e), "错误！！！")
    try:
        if imgMess['Xaxis'] != None:
            for i, ii in enumerate(redress_arithmetic(list(y_axvalue.values()))):
                y_axvalue[list(y_axvalue.keys())[i]] = ii
    except Exception as e:
        print("错误代码：005.2，y坐标轴文字重置失败。"+str(e), "错误！！！")

    try:
        if imgMess['Xaxis'] != None:
            _, _, txtx_str = read_xy(remark[0], remark[1], remark[2], y_array, img_ori, imgMess['Xaxis'], y_x0, y_x1)  # 返回xy坐标轴数值
            print("X备注文字识别。")
    except Exception as e:
        txtx_str =""
        print( "错误代码：005.3，X备注文字识别失败。" + str(e), "错误！！！")

    try:
        txty_str = read_ytxt(origin[remarky[2]-10:remarky[3]+10,remarky[0]:remarky[1]])
    except Exception as e:
        print("错误代码：005.4，Y备注文字识别失败。" + str(e), "错误！！！")
        txty_str = ""

    try:

        xaxis_gap = imgMess['Xaxis']
        yaxis_gap = imgMess['Yaxis']
        if len(y_axvalue) > 1 and len(x_axvalue) > 0:
            x_gap = (imgMess['Xaxis'][0], list(x_axvalue)[-1])
            y_gap = (list(y_axvalue)[0], list(y_axvalue)[-1])
            x_stanub = x_axvalue[list(x_axvalue)[0]]
            x_endnub = x_axvalue[list(x_axvalue)[-1]]
            y_stanub = y_axvalue[list(y_axvalue)[-1]]
            y_endnub = y_axvalue[list(y_axvalue)[0]]
        elif len(y_axvalue) > 1:
            x_gap = (imgMess['Xaxis'][0], 0)
            y_gap = (list(y_axvalue)[0], list(y_axvalue)[-1])
            x_stanub = 0
            x_endnub = 1
            y_stanub = y_axvalue[list(y_axvalue)[-1]]
            y_endnub = y_axvalue[list(y_axvalue)[0]]
        elif len(x_axvalue) > 0:

            x_gap = (imgMess['Xaxis'][0], list(x_axvalue)[-1])
            y_gap = (imgMess['Yaxis'][1], imgMess['Yaxis'][3])
            x_stanub = x_axvalue[list(x_axvalue)[0]]
            x_endnub = x_axvalue[list(x_axvalue)[-1]]
            y_stanub = 0
            y_endnub = 1
        else:
            x_gap = (imgMess['Xaxis'][0], imgMess['Xaxis'][2])
            y_gap = (imgMess['Yaxis'][1], imgMess['Yaxis'][3])
            x_stanub = 0
            x_endnub = 1
            y_stanub = 0
            y_endnub = 1
        img = img_redraw(img_ori.shape[:2],  x_gap, y_gap, x_stanub, x_endnub, y_stanub, y_endnub, xaxis_gap, yaxis_gap, x_axvalue, txtx_str, txty_str)  # 坐标轴重绘
    except Exception as e:
        print( "错误代码：006.0，图表坐标轴重绘失败。"+str(e), "错误！！！", win32con.MB_OK)
    try:
        print("tracker line...")
        if imgMess['Xaxis'] != None:
            splashes, lines,turestline = tracker_line(img_ori, imgMess['Xaxis'][0], imgMess['Xaxis'][2], imgMess['Yaxis'][1],imgMess['Yaxis'][3])
            y_vals = []
            y_valest = []
            for i, points in enumerate(lines):
                y_val = []
                y_vals.append(i)
                for point in points:
                    y_val.append(point)
                    cv2.circle(img, point, 1, colours[i % 6])  # 紫色
                y_vals.append(y_val)

            for i, points in enumerate(turestline):
                y_val = []
                y_valest.append(i)
                for point in points:
                    y_val.append(point)

                y_valest.append(y_val)

    except Exception as e:
        print( "错误代码：006.1，图表折线重绘失败。"+str(e), "错误！！！", win32con.MB_OK)

    try:
        if len(x_axvalue) == 0:
            xs = (0, 1,  imgMess['Xaxis'][0], imgMess['Xaxis'][2])
        else:
            xs = (x_axvalue[list(x_axvalue)[0]], x_axvalue[list(x_axvalue)[-1]],list(x_axvalue)[0], list(x_axvalue)[-1])
        if len(y_axvalue) == 0 or y_axvalue[list(y_axvalue)[0]]== y_axvalue[list(y_axvalue)[-1]]:
            ys = (0, 1, imgMess['Yaxis'][1], imgMess['Yaxis'][3])
        else:
            ys = (y_axvalue[list(y_axvalue)[0]], y_axvalue[list(y_axvalue)[-1]],list(y_axvalue)[0], list(y_axvalue)[-1])
    except Exception as e:
        print("错误代码：007.0，识别结果转换失败。"+str(e), "错误！！！", win32con.MB_OK)

    try:
        resultTxtPath = imgpath.replace(imgpath.split("/")[-1], Subject.replace(".com", "").replace("-Seeked", "-Redraw_raw.txt"))
        resultTxtPath2 = imgpath.replace(imgpath.split("/")[-1], Subject.replace(".com", "").replace("-Seeked", "-Redraw_fine.txt"))
        print("-----------------------------------------------")
        print(resultTxtPath [-14:])
        print(resultTxtPath2[-15:])
        resultTxtPath  = resultTxtPath  + "-Redraw_raw.txt" if resultTxtPath [-14:] != "Redraw_raw.txt" else resultTxtPath
        resultTxtPath2 = resultTxtPath2 + "-Redraw_fine.txt" if resultTxtPath2[-15:] != "Redraw_fine.txt" else resultTxtPath2
        pix2txt(y_vals, xs, ys, resultTxtPath)
        pix2txt2(y_valest, xs, ys, resultTxtPath2)
    except Exception as e:
        print("错误代码：007.1，识别结果保存失败。"+str(e), "错误！！！")
    print("test imgpath",imgpath)
    try:
        if imgpath[-4:] == ".jpg":
            resultpath = imgpath.replace("-Seeked.jpg", "-RedrawPic.jpg")
        else:
            resultpath = imgpath.replace(imgpath.split("/")[-1], Subject.replace("-Seeked","-RedrawPic.jpg"))

        resultpath = imgpath.replace(".jpg", "-RedrawPic.jpg")  if resultpath[-14:] != "-RedrawPic.jpg" else resultpath
        # cv2.imshow("img",img)
        # print(resultpath,Subject)
        cv2.imwrite(resultpath, img)
        print("save result img:",resultpath)
        return img
    except Exception as e:
        print("错误代码：007.2，tupian保存失败。"+str(e), "错误！！！")
        return 0
if __name__ =="__main__":
    imgpath = r"100.png"
    img = rec_graph(imgpath, "Subject-Seeked")
    cv2.imwrite("rg40.jpg", img)
    cv2.waitKey(0)