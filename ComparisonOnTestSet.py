from ultralytics import YOLO
import cv2
import os

# face_cascade = cv2.CascadeClassifier('default_frontal_face.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# smile_cascade = cv2.CascadeClassifier('default_smile_cascade.xml')
# smile_cascade = cv2.CascadeClassifier('smile_cascade.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

yolo_face = 0
cascade_face = 0
cascade_smile = 0

def detect_smiles(gray, frame):
    global frame_cop
    frame_cop = frame.copy()
    global a
    global b
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    a = len(faces)
    b = 0
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.9, 20)
        if len(smiles) != 0:
            b += 1
        cv2.rectangle(frame_cop, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(frame_cop, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 0, 255), 2)

    return frame_cop, a, b

CONFIDENCE_THRESHOLD = 0.2
GREEN = (0, 255, 0)
THICKNESS = -1

model = YOLO("yolov8n-face.pt")

input_folder = 'class0'
output_folder = 'res0'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = os.listdir(input_folder)
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)

    print(image_path)
    frame = cv2.imread(image_path)
    detections = model(frame)[0]
    img = frame.copy()

    yolo_face = len(detections.boxes.data.tolist())
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), GREEN, 4)

    cv2.imwrite(output_path, img)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_with_boxes, cascade_face, cascade_smile = detect_smiles(gray, frame.copy())
    print(yolo_face, cascade_face, cascade_smile)
    if (yolo_face == cascade_smile):
        output_class_folder = os.path.join(output_folder, 'class1')
    else:
        output_class_folder = os.path.join(output_folder, 'class0')

    if not os.path.exists(output_class_folder):
        os.makedirs(output_class_folder)

    output_path = os.path.join(output_class_folder, image_file)

    cv2.imwrite(output_path, frame_with_boxes)
    print(f"Processed {image_file} and saved to {output_class_folder}")

input_folder = 'class1'
output_folder = 'res1'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = os.listdir(input_folder)
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)

    print(image_path)
    frame = cv2.imread(image_path)
    detections = model(frame)[0]
    img = frame.copy()

    yolo_face = len(detections.boxes.data.tolist())
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), GREEN, 4)

    cv2.imwrite(output_path, img)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_with_boxes, cascade_face, cascade_smile = detect_smiles(gray, frame.copy())
    print(yolo_face, cascade_face, cascade_smile)
    if (yolo_face == cascade_smile):
        output_class_folder = os.path.join(output_folder, 'class1')
    else:
        output_class_folder = os.path.join(output_folder, 'class0')

    if not os.path.exists(output_class_folder):
        os.makedirs(output_class_folder)

    output_path = os.path.join(output_class_folder, image_file)

    cv2.imwrite(output_path, frame_with_boxes)
    print(f"Processed {image_file} and saved to {output_class_folder}")

TN = len(os.listdir('res0/class0'))
FP = len(os.listdir('res0/class1'))
FN = len(os.listdir('res1/class0'))
TP = len(os.listdir('res1/class1'))

print('----------------------------------------------')
# print(TN, FP, FN, TP)

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1s = (2 * precision * recall) / (precision + recall)
print('TP: '+str(TP)+' FN: '+str(FN)+' FP: '+str(FP)+' TN: '+str(TN))

print('Accuracy = '+str(accuracy))
print('Precision = '+str(precision))
print('Recall = '+str(recall))
print('F1 = '+str(F1s))