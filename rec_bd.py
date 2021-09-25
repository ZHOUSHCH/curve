from aip import AipOcr
import cv2
import numpy as np
import pytesseract
from PIL import Image

APP_ID = 'xxxxxxxxx'
API_KEY = 'xxxxxxxxxx'
SECRET_KEY = 'xxxxxxxxxxx'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def detsci_not(path):
    def projection_h(frame, threshold):
            h = len(frame)
            w = len(frame[0])

            projection_img = np.zeros((h, w, 1), np.uint8)

            projection_img.fill(255)
            pixdata = frame
            h_array = []
            for y in range(h):
                b_count = 0
                for x in range(w):
                    if pixdata[y, x] < threshold:
                        b_count += 1
                h_array.append(b_count)
            for i, ii in enumerate(h_array):
                for j in range(ii):
                    projection_img[i][j] = 0

            return h_array, projection_img

    def projection_w(frame, threshold):
            h = len(frame)
            w = len(frame[0])
            # 使用Numpy创建一张A4(2105×1487)纸
            projection_img = np.zeros((h, w, 1), np.uint8)
            # 使用白色填充图片区域,默认为黑色
            projection_img.fill(255)
            pixdata = frame
            w_array = []
            for y in range(w):
                b_count = 0
                for x in range(h):
                    if pixdata[x, y] < threshold:
                        b_count += 1
                w_array.append(b_count)
            for i, ii in enumerate(w_array):
                for j in range(ii):
                    projection_img[j][i] = 0
            return w_array, projection_img

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((int(len(gray)/3), int(len(gray)/3)), np.uint8)
    pixdata = cv2.erode(gray, kernel)
    h_array, projection_himg = projection_h(pixdata, 150)
    w_array, projection_wimg = projection_w(pixdata, 150)
    score = 0
    for i, ii in enumerate(h_array):
        if (i < len(h_array)/2 and ii > sum(h_array)/len(h_array)) or (i >= len(h_array)/2 and ii < sum(h_array)/len(h_array)):
            score += 1
    confidence = score / len(h_array)#是否为科学计数法
    delice_y = 0
    if confidence > 0.7:
        # print("confidence",confidence)
        for i in range(len(w_array)-1):
            if w_array[i] > sum(w_array) / len(w_array) and w_array[i+1]  < sum(w_array) / len(w_array):
                delice_y = i+1
                # print("position", i, len(w_array), w_array[i-2:i+2])

        cv2.imwrite(path.split(path.split('\\')[-1])[0]+"rea.jpg", img[:, :(delice_y - int(len(gray) / 6))])     #
        cv2.imwrite(path.split(path.split('\\')[-1])[0]+"exp.jpg", img[:int(len(gray) / 2), (delice_y - int(len(gray) / 6)):]) #指数次幂
        return True
    else:
        return False

def ocr_bd2txt(path):


    image = get_file_content(path)
    txt_all = client.basicGeneral(image)
    # print(txt_all)
    if txt_all.get('words_result', 0):
        ts = txt_all.get('words_result', 0)
        txt = ""
        for t1 in ts:
            t0 = t1['words']
            txt += t0
    else:
        txt = None
    if  txt == None:
        pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR/tesseract.exe'
        txt = pytesseract.image_to_string(Image.open(path))

    return txt

if __name__== "__main__":

    ocr_bd2txt("123.jpg")

    cv2.waitKey(0)