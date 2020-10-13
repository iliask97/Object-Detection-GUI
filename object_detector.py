from tkinter import *
import cv2
import numpy as np
from tkinter import filedialog

root = Tk()
root.title("Object Detector")
root.iconbitmap("favicon.ico")
root.resizable(False, False)

def imagedetection():
    filename = filedialog.askopenfilename(initialdir="C:/", title="Select an image", filetypes=(("jpg files", "*.jpg"),))
    net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
    classes = []
    with open('coco.names', 'r') as f:
        classes = f.read().splitlines()
    image = cv2.imread(filename)
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    network_out = net.forward(output_layers_names)
    boxes = []
    confidences = []
    class_ids = []
    for output in network_out:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
    cv2.imshow('IMAGE', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def videodetection():
    filename = filedialog.askopenfilename(initialdir="C:/", title="Select a video", filetypes=(("mp4 files", "*.mp4"),))
    net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
    classes = []
    with open('coco.names', 'r') as f:
        classes = f.read().splitlines()
    cap = cv2.VideoCapture(filename)
    while True:
        _, image = cap.read()
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        network_out = net.forward(output_layers_names)
        boxes = []
        confidences = []
        class_ids = []
        for output in network_out:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
        cv2.imshow('VIDEO', image)
        key = cv2.waitKey(12)
        if key & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

def realtime():
    net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
    classes = []
    with open('coco.names', 'r') as f:
        classes = f.read().splitlines()
    cap = cv2.VideoCapture(0)
    while True:
        _, image = cap.read()
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        network_out = net.forward(output_layers_names)
        boxes = []
        confidences = []
        class_ids = []
        for output in network_out:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
        cv2.imshow('REAL TIME', image)
        key = cv2.waitKey(12)
        if key & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


header = Label(root, text="Welcome to Object Detector", font=("Gabriola", 40))
header.grid(row=0, column=0, columnspan=3)
frame1 = LabelFrame(root, text="Perform object detection in a single image", padx=5, pady=5)
frame1.grid(row=1, column=0, padx=5, pady=5)
label1_1 = Label(frame1, text="To get detections on a single image follow the instructions bellow!!!")
label2_1 = Label(frame1, text="1. Click the button: Image Detection.")
label3_1 = Label(frame1, text="2. Choose an image from the file chooser that will appear.")
label4_1 = Label(frame1, text="3. Wait a few seconds for the detections!!")
b_1 = Button(frame1, text="Image Detection", relief=GROOVE, command=imagedetection)
label1_1.grid(row=0, column=0, sticky=W)
label2_1.grid(row=1, column=0, sticky=W)
label3_1.grid(row=2, column=0, sticky=W)
label4_1.grid(row=3, column=0, sticky=W)
b_1.grid(row=4, column=0, sticky=W)
frame2 = LabelFrame(root, text="Perform object detection in a video", padx=5, pady=5)
frame2.grid(row=1, column=1, padx=5, pady=5)
label1_2 = Label(frame2, text="To get detections on a video follow the instructions bellow!!!")
label2_2 = Label(frame2, text="1. Click the button: Video Detection.")
label3_2 = Label(frame2, text="2. Choose a video from the file chooser that will appear.")
label4_2 = Label(frame2, text="3. Wait a few seconds for the detections!!")
b_2 = Button(frame2, text="Video Detection", relief=GROOVE, command=videodetection)
label1_2.grid(row=0, column=0, sticky=W)
label2_2.grid(row=1, column=0, sticky=W)
label3_2.grid(row=2, column=0, sticky=W)
label4_2.grid(row=3, column=0, sticky=W)
b_2.grid(row=4, column=0, sticky=W)
frame3 = LabelFrame(root, text="Perform object detection in real time", padx=5, pady=5)
frame3.grid(row=1, column=2, padx=5, pady=5)
label1_3 = Label(frame3, text="To get detections in real time follow the instructions bellow!!!")
label2_3 = Label(frame3, text="1. Click the button: Real Time Detection.")
label3_3 = Label(frame3, text="2. Wait a few seconds for the detections!!")
label4_3 = Label(frame3, text="Attention: For this button to work you need a web camera!!!")
b_3 = Button(frame3, text="Real Time Detection", relief=GROOVE, command=realtime)
label1_3.grid(row=0, column=0, sticky=W)
label2_3.grid(row=1, column=0, sticky=W)
label3_3.grid(row=2, column=0, sticky=W)
label4_3.grid(row=3, column=0, sticky=W)
b_3.grid(row=4, column=0, sticky=W)
button = Button(root, text="Exit", relief=GROOVE, command=root.destroy)
button.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky=E)
root.mainloop()