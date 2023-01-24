import cv2, base64, io
import numpy as np
from PIL import Image

def get_yolo_net(cfg_path: str, weight_path: str):
    return cv2.dnn.readNetFromDarknet(cfg_path, weight_path)

LABELS = open("static/yolo/yolo.names").read().strip().split("\n")
net = get_yolo_net("static/yolo/yolo.cfg", "static/yolo/yolo.weights")

def yolo_forward(net, LABELS, image, confidence_level, save_image=False):
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_outputs = net.forward(ln)

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_level:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_level, confidence_level)

    if len(idxs) > 0: filtered_idxs = idxs.flatten()
    else: filtered_idxs = []
    

    nms_class_ids = [int(class_ids[i]) for i in filtered_idxs]
    nms_boxes = [[int(elem) for elem in boxes[i]] for i in filtered_idxs]
    nms_confidences = [float(confidences[i]) for i in filtered_idxs]

    labels = [LABELS[i] for i in nms_class_ids]

    return nms_class_ids, labels, nms_boxes, nms_confidences


def bboxes(image, class_ids, boxes, labels, colors = None):
    if colors is None:
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(10000, 3), dtype='uint8')

    for i, box in enumerate(boxes):
        (x, y, w, h) = (box[0], box[1], box[2], box[3])

        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        text = '{}'.format(labels[i])

        font_scale = 0.9
        rectangle_bgr = color

        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=1)[0]
        text_offset_x = x
        text_offset_y = y - 3 

        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 10, text_offset_y - text_height - 10))
        cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(image, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(255, 255, 255), thickness=2)

    _, im_arr = cv2.imencode('.jpg', image)
    bytes = im_arr.tobytes()
    b64 = base64.b64encode(bytes)
    return b64.decode("utf-8", "strict")

def load_image(b64: str):
    bin_image = base64.b64decode(b64)
    image = Image.open(io.BytesIO(bin_image))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def detect(image):
    ids, labels, boxes, confs = yolo_forward(net, LABELS, image, 0.3, True)
    image = bboxes(image, ids, boxes, labels)
    return {"ids": ids, "labels": labels, "boxes": boxes, "confs": confs, "image": image}