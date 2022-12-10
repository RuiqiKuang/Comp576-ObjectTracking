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
                                    False)  # # The input required by net is in blob format, use the function blobFromImage to convert the format
    net.setInput(blobImg)  # # Call the setInput function to send the image to the input layer

    # Get network output layer information (names of all output layers), set and forward propagate
    outInfo = net.getUnconnectedOutLayersNames()
    start = time.time()
    layerOutputs = net.forward(outInfo)  # The information of each output layer, each detection box, etc. is obtained, and is a two-dimensional structure.
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))  # # Print the following information

    # Get the picture size
    H, W = img.shape[0], img.shape[1]
    boxes = []  # All boxes
    confidences = []  # All confidence levels
    classIDs = []  # All Category IDs

    # Filtering
    for out in layerOutputs:
        for bbox in out:
            # Get the confidence level
            scores = bbox[5:]  # Confidence level of each category
            classID = np.argmax(scores)  # The id with the highest confidence is the classification id
            confidence = scores[classID]  # Get the confidence level

            # Screening by confidence level
            if confidence > CONF_THRES:
                box = bbox[0:4] * np.array([W, H, W, H])  # Put the bounding box to the image size
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width // 2))
                y = int(centerY - (height // 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # # 2) Apply non-maxima suppression (non-maxima suppression, nms) to further screen out
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRES, NMS_THRES)

    # Application test results
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
