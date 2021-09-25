from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import copy
def initiModel():
    img_size = 512
    weights = "utils/best_1122.pt"
    device = torch_utils.select_device(device='')  # cpu = "cpu"  gpu  =""
    model = Darknet('utils/yolov4-relu-hat.cfg', img_size)
    attempt_download(weights)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()
    names = load_classes('utils/hat.names')
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
    _ = model(img.float())
    return colors, img_size, device, model, names
def detect(path,savepath,colors,img_size,device,model,names):

    im0s = cv2.imread(path)
    imgOri = copy.deepcopy(im0s)
    img = letterbox(im0s, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.3, 0.6, multi_label=False)
    imgs = []
    confs = []
    for i, det in enumerate(pred):
        s, im0 ='', im0s
        s += '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            for j,(*xyxy, conf, cls) in enumerate(det):
                confs.append(conf)
                label = '%s %.2f' % (names[int(cls)], conf)
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                # print(c1, c2,savepath.replace(".jpg", "_%s.jpg" % j),len(im0[c1[1]:c2[1]]))
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                toleranceY = int((max(c2[1], c1[1]) - min( c2[1], c1[1])) * 0.1)
                toleranceX = int((max(c2[0], c1[0]) - min(c2[0], c1[0])) * 0.1)
                resXLU, resXRD = max(0, c1[0] - toleranceX),  c2[0] + toleranceX
                resYLU, resYRD = max(0, c1[1] - toleranceY),  c2[1] + toleranceY
                # print( resXLU, resXRD, resYLU, resYRD)
                singleImgPath = savepath.replace("-Remarked.jpg", "_%s-Seeked.jpg" % j)
                imgs.append(singleImgPath)
                cv2.imwrite(singleImgPath, imgOri[resYLU:resYRD, resXLU:resXRD])
        cv2.imwrite(savepath, im0)
        # print(i)
    conf = "%.2f" % (sum(confs)/len(confs)) if len(confs) else 0
    return imgs, conf
if __name__ == '__main__':

    with torch.no_grad():
        colors,img_size,device,model,names = initiModel()
        result = detect( r"006.png","res.jpg",colors,img_size,device,model,names)