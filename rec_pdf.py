import fitz
from detect_mini import *
import rec_graph
import writeWord
import zipfile
from shutil import copyfile
def pdf2png(path):#name is must unsame with all
    zoom_x = 2# 每个尺寸的缩放系数为2，这将为我们生成分辨率提高四倍的图像。
    zoom_y = 2
    pdfName = path.split("//")[-1].replace(".pdf", "")
    dir_1 = './' + pdfName+'/'
    doc = fitz.open(path)
    imgPaths = []
    for pg in range(doc.pageCount):
        page = doc[pg]
        rotate = int(0)
        trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pm = page.getPixmap(matrix=trans, alpha=False)
        mkdirlambda = lambda x: os.makedirs(x) if not os.path.exists(x) else True  # 目录是否存在,不存在则创建
        mkdirlambda(dir_1)
        imgPath = '%s%s.jpg' % (dir_1, pg+1)
        pm.writePNG( imgPath)
        imgPaths.append(imgPath)
    return imgPaths

def img2GraphImg(paths):
    with torch.no_grad():
        colors, img_size, device, model, names = initiModel()
    RemarkImgPaths = []
    resultPaths = []
    confs = []
    for i, path in enumerate(paths):
        GraphImgName = path.replace(path.split("/")[-1], "")
        mkdirlambda = lambda x: os.makedirs(x) if not os.path.exists(x) else True  # 目录是否存在,不存在则创建
        mkdirlambda(GraphImgName + 'result')
        RemarkedImgPath = GraphImgName + 'result/' + path.split("/")[-1].replace(".jpg", "-Remarked.jpg")
        RemarkImgPaths.append(RemarkedImgPath)
        resultPath,conf = detect(path, RemarkedImgPath, colors, img_size, device, model, names)
        resultPaths.append(resultPath)
        confs.append(conf)
        for j in resultPath:
            print("rec graph ",j)
            img = rec_graph.rec_graph(j, j.split("/")[-1].replace(".jpg",""))
    return RemarkImgPaths,resultPaths,confs

def ziptxt(paths,zipPath = 'test/result.zip'):
    txts = []

    for i in paths:
        for j in i:
            txta = j.replace("-Seeked.jpg", "-Redraw_raw.txt")
            txtb = j.replace("-Seeked.jpg", "-Redraw_fine.txt")
            if os.path.exists(txta):
                cacheTxta = txta.split("/")[-1]
                copyfile(txta, cacheTxta)
                txts.append(cacheTxta)
            if os.path.exists(txtb):
                cacheTxtb = txtb.split("/")[-1]
                copyfile(txtb, cacheTxtb)

                txts.append(cacheTxtb)
    with zipfile.ZipFile(zipPath, "w", zipfile.ZIP_DEFLATED) as zf:
        print(txts)
        for txt in txts:
            if os.path.exists(txt):
                # shutil.copyfile(txt, txt)
                zf.write(txt)
                os.remove(txt)
    return zipPath
def pdf2ResFile(pdfpath):
    print("PDF is in,path:", pdfpath)
    paths = pdf2png(pdfpath)
    RemarkImgPaths,resultPaths,confs = img2GraphImg(paths)
    wordPath = writeWord.WriteWord(pdfpath.replace(".pdf", ""), RemarkImgPaths, resultPaths,confs)

    zipPath = ziptxt(resultPaths,pdfpath.replace(".pdf", "/result.zip"))
    return wordPath,zipPath

if __name__=='__main__':
    pdfpath = r'test.pdf'
    pdf2ResFile(pdfpath)

