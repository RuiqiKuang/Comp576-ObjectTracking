import cv2
import numpy as np
import time

from utils import xywh_to_xyxy

input_path = r'./data/road.MP4'
# output_path = 'out-' + input_path.split('/')[-1]
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
with open('coco.names', 'r') as f:
    labels = f.read().splitlines()
CONF_THRES = 0.5
NMS_THRES = 0.4
layersName = net.getLayerNames()
output_layers_name = [layersName[i - 1] for i in net.getUnconnectedOutLayers()]


def process(img, cnt):
    blobImg = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (416, 416), None, True,
                                    False)  # # net需要的输入是blob格式的，用blobFromImage这个函数来转格式
    net.setInput(blobImg)  # # 调用setInput函数将图片送入输入层

    # 获取网络输出层信息（所有输出层的名字），设定并前向传播
    outInfo = net.getUnconnectedOutLayersNames()
    start = time.time()
    layerOutputs = net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构。
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))  # # 可以打印下信息

    # 拿到图片尺寸
    H, W = img.shape[0], img.shape[1]
    boxes = []  # 所有框
    confidences = []  # 所有置信度
    classIDs = []  # 所有分类ID

    # 过滤
    for out in layerOutputs:
        for bbox in out:
            # 拿到置信度
            scores = bbox[5:]  # 各个类别的置信度
            classID = np.argmax(scores)  # 最高置信度的id即为分类id
            confidence = scores[classID]  # 拿到置信度

            # 根据置信度筛查
            if confidence > CONF_THRES:
                box = bbox[0:4] * np.array([W, H, W, H])  # 将边界框放会图片尺寸
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width // 2))
                y = int(centerY - (height // 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRES, NMS_THRES)

    # 应用检测结果
    np.random.seed(42)
    filename = "./data/labels/" + "road_" + str(cnt) + ".txt"
    print(filename)
    f = open(filename, 'w')
    if len(indexes) > 0:
        for i in indexes.flatten():
            x_, y_, w, h = boxes[i]
            x1, y1, x2, y2 = x_, y_, x_ + w, y_ + h
            line = "0 %d %d %d %d\n" % (x1, y1, x2, y2)
            f.write(line)


capture = cv2.VideoCapture(input_path)
cnt = 1
if capture.isOpened():
    while True:
        ret, img_src = capture.read()
        if not ret:
            break
        process(img_src, cnt)
        cnt += 1
else:
    print('Cannot open the video')
capture.release()
