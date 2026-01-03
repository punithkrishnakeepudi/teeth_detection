import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("best (2).pt")

# Open webcam (use 0 or video file path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, conf=0.4)

    teeth_count = 0

    for r in results:
        if r.boxes is None:
            continue

        # Number of detected objects = number of teeth
        teeth_count = len(r.boxes)

        # Draw bounding boxes
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = f"Tooth {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Display teeth count
    cv2.putText(
        frame,
        f"Teeth Count: {teeth_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    cv2.imshow("YOLOv8 Teeth Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
