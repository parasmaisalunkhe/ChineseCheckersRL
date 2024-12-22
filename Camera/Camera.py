import cv2
import depthai as dai
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("./Camera/Yolo11sModel.pt")

# Function to run YOLO inference on the captured frame
def run_yolo_inference(frame):
    results = model(frame)  # Run YOLO inference
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates (x_min, y_min, x_max, y_max)
        confs = result.boxes.conf  # Confidence scores
        classes = result.boxes.cls  # Class IDs

        # Draw the bounding boxes and labels
        for i in range(len(boxes)):
            x_min, y_min, x_max, y_max = map(int, boxes[i])
            label = f"{model.names[int(classes[i])]}: {confs[i]:.2f}"
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Initialize DepthAI pipeline for RGB camera
pipeline = dai.Pipeline()
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(640, 480)
camRgb.setInterleaved(False)
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
camRgb.preview.link(xout.input)

# Start the pipeline
with dai.Device(pipeline) as device:
    while True:
        videoQueue = device.getOutputQueue("video", maxSize=4, blocking=False)
        print("Press 's' to capture and run YOLO inference. Press 'q' to quit.")
        inFrame = videoQueue.get()
        frame = inFrame.getCvFrame()
            # cv2.imshow("DepthAI Camera Feed", frame)
        frame_with_detections = run_yolo_inference(frame)
        cv2.waitKey(1)
        cv2.imshow("YOLO Inference", frame_with_detections)
            # cv2.imwrite("yolo_inference_result.jpg", frame_with_detections)
            # print("Inference result saved as 'yolo_inference_result.jpg'.")

        # cv2.destroyAllWindows()
