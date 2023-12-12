from ultralytics import YOLO
import cv2

# face_cascade = cv2.CascadeClassifier('default_frontal_face.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# smile_cascade = cv2.CascadeClassifier('default_smile_cascade.xml')
# smile_cascade = cv2.CascadeClassifier('smile_cascade.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

CONFIDENCE_THRESHOLD = 0.4
GREEN = (0, 255, 0)
THICKNESS = -1
video_cap = cv2.VideoCapture(0)
model = YOLO("yolov8n-face.pt")

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


while True:
    ret, frame = video_cap.read()
    if not ret:
        break
    detections = model(frame)[0]

    img = frame.copy()

    yolo_face = len(detections.boxes.data.tolist())
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), GREEN, 4)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_with_boxes, cascade_face, cascade_smile = detect_smiles(gray, frame.copy())
    cv2.imshow('Video', frame_with_boxes)
    print(yolo_face, cascade_face, cascade_smile)
    if (yolo_face == cascade_smile):
        cv2.imwrite('smiling_faces_image.jpg', frame)
        break
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
cv2.destroyAllWindows()