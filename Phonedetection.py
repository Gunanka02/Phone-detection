import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet(r"C:\Users\MyFolder\Downloads\yolov3.weights", r"C:\Users\MyFolder\Downloads\yolov3.cfg")

# Load class labels (COCO dataset)
with open(r"C:\Users\MyFolder\Downloads\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO output layer names
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()

# Ensure we correctly handle scalar and array cases
if isinstance(unconnected_layers, np.ndarray):
    unconnected_layers = unconnected_layers.flatten().astype(int)
output_layers = [layer_names[i - 1] for i in unconnected_layers]

# Start video capture (webcam)
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Get the index of the "phone" class (make sure to check the index in coco.names)
phone_class_id = classes.index("cell phone")  # or "phone" depending on your class names

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialization of lists for detection info
    class_ids = []
    confidences = []
    boxes = []

    # Process YOLO output
    for out in outs:
        for detection in out:
            scores = detection[5:]  # Scores of the classes
            class_id = np.argmax(scores)  # Class with the highest confidence score
            confidence = scores[class_id]  # Confidence of that class

            if confidence > 0.5:  # Confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                if class_id == phone_class_id:  # Check if the detected class is "phone"
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Apply Non-Maximum Suppression to filter overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Flag to check if a phone was detected
    phone_detected = False

    # Draw labels and boxes on detected phones
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])  # Get class label (e.g., "cell phone")
            confidence = confidences[i]

            # Draw a rectangle around the object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Set flag to indicate that a phone was detected
            phone_detected = True

    # Display notification if a phone was detected
    if phone_detected:
        cv2.putText(frame, "Phone Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the video frame with detections
    cv2.imshow('Phone Detection', frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
