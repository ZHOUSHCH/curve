import win32api,win32con
import cv2
from founction import *
colours = [(250, 0, 0), (0, 250, 0), (0, 0, 250), (0, 250, 250), (250, 0, 250), (250, 250, 0)]
def rec_graph(imgpath,Subject):
    go = True
    try:
        img_ori = cv2.imread(imgpath)
    except Exception as e:
        go = False
        win32api.MessageBox(0, "错误代码：001，没有图片或图片格式不正确。"+str(e), "错误！！！", win32con.MB_OK)
    if go:
        try:
            img, points = line_detect_possible_demo(img_ori)#检测直线，返回图和直线数组

        except Exception as e:
            go = False
            win32api.MessageBox(0, "错误代码：002，未检测到直线。"+str(e), "错误！！！", win32con.MB_OK)
        if go:
            # try:
            if go:
                min_xlines, min_ylines,img_ori2 = points2points(points, 50, img_ori)#根据返回直线，选择坐标轴直线

            # except Exception as e:
            #     go = False
            #     win32api.MessageBox(0, "错误代码：003，直线筛选失败。"+str(e), "错误！！！", win32con.MB_OK)
            if go:
                try:
                    y0, y1, x_array, remark = cutimg_x(img_ori[min_xlines[1]:, :])#x轴刻度切除
                    y_x0, y_x1, y_array, remarky = cutimg_y(img_ori[:, :min_ylines[0]])#y轴刻度切除
                except Exception as e:
                    go = False
                    win32api.MessageBox(0, "错误代码：004，坐标轴文字检测失败。"+str(e), "错误！！！", win32con.MB_OK)
                if go:
                    try:
                        x_axvalue, y_axvalue, _ = read_xy(y0, y1, x_array, y_array, img_ori, min_xlines, y_x0, y_x1)  # 返回xy坐标轴数值
                    except Exception as e:
                        go = False
                        win32api.MessageBox(0, "错误代码：005.1，坐标轴文字识别失败。"+str(e), "错误！！！", win32con.MB_OK)
                    try:
                        for i, ii in enumerate(redress_arithmetic(list(y_axvalue.values()))):
                            y_axvalue[list(y_axvalue.keys())[i]] = ii
                    except Exception as e:
                        go = False
                        win32api.MessageBox(0, "错误代码：005.2，y坐标轴文字重置失败。"+str(e), "错误！！！", win32con.MB_OK)
                    if remark and len(remark) > 0:
                        try:
                            _, _, txtx_str = read_xy(remark[0], remark[1], remark[2], y_array, img_ori, min_xlines, y_x0, y_x1)  # 返回xy坐标轴数值
                            print("X备注文字识别。")
                        except Exception as e:
                            go = False
                            win32api.MessageBox(0, "错误代码：005.3，X备注文字识别失败。" + str(e), "错误！！！", win32con.MB_OK)
                    else:
                        txtx_str = ""
                    if remarky and len(remarky) > 0:
                        txty_str = read_ytxt(img_ori[:,remarky[0]:remarky[1]])
                    else:
                        txty_str = ""

                    if go:
                        try:
                            xaxis_gap = min_xlines
                            yaxis_gap = min_ylines
                            if len(y_axvalue) > 1 and len(x_axvalue) > 0:
                                x_gap = (min_xlines[0], list(x_axvalue)[-1])
                                y_gap = (list(y_axvalue)[0], list(y_axvalue)[-1])
                                x_stanub = x_axvalue[list(x_axvalue)[0]]
                                x_endnub = x_axvalue[list(x_axvalue)[-1]]
                                y_stanub = y_axvalue[list(y_axvalue)[-1]]
                                y_endnub = y_axvalue[list(y_axvalue)[0]]
                            elif len(y_axvalue) > 1:
                                x_gap = (min_xlines[0], 0)
                                y_gap = (list(y_axvalue)[0], list(y_axvalue)[-1])
                                x_stanub = 0
                                x_endnub = 1
                                y_stanub = y_axvalue[list(y_axvalue)[-1]]
                                y_endnub = y_axvalue[list(y_axvalue)[0]]
                            elif len(x_axvalue) > 0:

                                x_gap = (min_xlines[0], list(x_axvalue)[-1])
                                y_gap = (min_ylines[1], min_ylines[3])
                                x_stanub = x_axvalue[list(x_axvalue)[0]]
                                x_endnub = x_axvalue[list(x_axvalue)[-1]]
                                y_stanub = 0
                                y_endnub = 1
                            else:

                                x_gap = (min_xlines[0], min_xlines[2])
                                y_gap = (min_ylines[1], min_ylines[3])
                                x_stanub = 0
                                x_endnub = 1
                                y_stanub = 0
                                y_endnub = 1
                            img = img_redraw(img_ori.shape[:2],  x_gap, y_gap, x_stanub, x_endnub, y_stanub, y_endnub, xaxis_gap, yaxis_gap, x_axvalue, txtx_str, txty_str)  # 坐标轴重绘
                        except Exception as e:
                            go = False
                            win32api.MessageBox(0, "错误代码：006.0，图表坐标轴重绘失败。"+str(e), "错误！！！", win32con.MB_OK)
                        if go:
                            try:
                                print("tracker_line")
                                splashes, lines,turestline = tracker_line(img_ori, min_xlines[0], min_xlines[2], min_ylines[1],min_xlines[3])
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
                                go = False
                                win32api.MessageBox(0, "错误代码：006.1，图表折线重绘失败。"+str(e), "错误！！！", win32con.MB_OK)
                            if go:
                                try:
                                    go_txt = True
                                    if len(x_axvalue) == 0:
                                        xs = (0, 1,  min_xlines[0], min_xlines[2])
                                    else:
                                        xs = (x_axvalue[list(x_axvalue)[0]], x_axvalue[list(x_axvalue)[-1]],list(x_axvalue)[0], list(x_axvalue)[-1])
                                    if len(y_axvalue) == 0 or y_axvalue[list(y_axvalue)[0]]== y_axvalue[list(y_axvalue)[-1]]:
                                        ys = (0, 1, min_ylines[1], min_ylines[3])
                                    else:
                                        ys = (y_axvalue[list(y_axvalue)[0]], y_axvalue[list(y_axvalue)[-1]],list(y_axvalue)[0], list(y_axvalue)[-1])
                                except Exception as e:
                                    go_txt = False
                                    win32api.MessageBox(0, "错误代码：007.0，识别结果转换失败。"+str(e), "错误！！！", win32con.MB_OK)
                                if go_txt:
                                    try:
                                        pix2txt(y_vals, xs, ys, "data_"+Subject+".txt")
                                        pix2txt2(y_valest, xs, ys, "result_" + Subject + ".txt")
                                    except Exception as e:
                                        win32api.MessageBox(0, "错误代码：007.1，识别结果保存失败。"+str(e), "错误！！！", win32con.MB_OK)
    if go:
        cv2.imwrite("pic_"+Subject+".jpg", img)
        return img
    else:
        return [0]

if __name__ =="__main__":
    # imgpath = r"zhshichao@qq.com20200511190303.jpg"
    # imgpath = r"zhshichao@qq.com20200511191805.jpg"
    # imgpath = r"Screen Shot 2020-05-16 at 9.54.10 PM.png"
    # imgpath = r"A highly processable metallic glass.png"
    imgpath = r"curve.png"
    img = rec_graph(imgpath, "Subject")
    cv2.imshow('image', img)
    cv2.waitKey(0)