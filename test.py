import cv2
import numpy as np
import pytesseract
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime
import threading

from flask import Flask

app = Flask(__name__)

input_width = 224
input_height = 224

cred = credentials.Certificate(r"C:\Users\User\Documents\GitHub\my-electron-app\RedLight\fyptraffic-92fba-firebase-adminsdk-c0yk7-3b039ef6b8.json")  # Replace with the path to your service account JSON file
firebase_admin.initialize_app(cred, {
    'storageBucket': 'fyptraffic-92fba.appspot.com',
    'databaseURL': "https://fyptraffic-92fba-default-rtdb.asia-southeast1.firebasedatabase.app"
})
bucket = storage.bucket()
db_ref = db.reference('Saman')

# Load YOLOv4 model
yolov4_config_path = r'C:\Users\User\Documents\GitHub\my-electron-app\RedLight\yolov4-tiny.cfg'
yolov4_weights_path = r'C:\Users\User\Documents\GitHub\my-electron-app\RedLight\yolov4-tiny.weights'
net = cv2.dnn.readNet(yolov4_weights_path, yolov4_config_path)

# Set up Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to check if a car is crossing the line
def is_crossing_line(car_bbox, line_start, line_end):
    car_x_min, car_y_min, car_x_max, car_y_max = car_bbox
    line_x1, line_y1 = line_start
    line_x2, line_y2 = line_end

    # Check if the bounding box intersects with the line
    if car_x_max >= line_x1 and car_x_min <= line_x2:
        if line_y1 <= car_y_min <= line_y2 or line_y1 <= car_y_max <= line_y2:
            return True

    return False

# Function for preprocessing the car plate region
def preprocess_image(car_region):
    if car_region is None or car_region.size == 0:
        return None
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(car_region, cv2.COLOR_BGR2GRAY)

    # Apply image enhancement or other preprocessing techniques as required

    return grayscale_image

# Define the line position
line_length = 700  # Length of the line
line_start = None
line_end = None

# Create a list of video paths
video_paths = [
    r'C:\Users\User\Documents\GitHub\my-electron-app\RedLight\video.mp4',
    r'C:\Users\User\Documents\GitHub\my-electron-app\RedLight\v2.mp4',
    # Add more video paths for additional videos if needed
]

def process_video(video_path, line_y):
    # Create a VideoCapture object to read video frames
    cap = cv2.VideoCapture(video_path)

    # Check if the video capture is successfully opened
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Read the first frame to determine the line position
    ret, frame = cap.read()
    if not ret:
        print(f"Error reading video frame: {video_path}")
        return

    # Define the line position if not defined already
    if line_y is None:
        line_y = frame.shape[0] // 2 + 100  # y-coordinate for the line (horizontal position)
        line_start = (frame.shape[1] // 2 - line_length // 2, line_y)  # Starting point of the line (x-coordinate, y-coordinate)
        line_end = (frame.shape[1] // 2 + line_length // 2, line_y)  # Ending point of the line (x-coordinate, y-coordinate)

    image_counter = 0

    # Process video frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if a frame was successfully read
        if not ret:
            break

        # Perform car detection on the frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(output_layers)

        # Filter for car detections on the line
        cars_on_line = []
        class_ids = []
        confidences = []
        boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1 and class_id == 2:  # Adjust the confidence threshold and class ID for cars
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])
                     # Check if the width and height values are valid and finite
                    if not (np.isfinite(width) and np.isfinite(height)) or width == 0 or height == 0:
                        # Handle the exceptional case where width or height is zero or infinity
                        # Assign appropriate default values or skip further processing
                        width = 20  # Assign a default width value
                        height = 20  # Assign a default height value
                        continue
                    if not np.isfinite(height):
                        # Handle the exceptional case where height is zero or infinity
                        # Assign an appropriate default height value or skip further processing
                        continue
                    width = int(width)
                    height = int(height)
                    x_min = int(center_x - width / 2)
                    y_min = int(center_y - height / 2)
                    x_max = int(center_x + width / 2)
                    y_max = int(center_y + height / 2)
                    if is_crossing_line((x_min, y_min, x_max, y_max), line_start, line_end):
                        cars_on_line.append((x_min, y_min, x_max, y_max))
                        class_ids.append(class_id)
                        confidences.append(confidence)
                        boxes.append([x_min, y_min, x_max, y_max])

        # Check if cars are detected
        if len(cars_on_line) > 0:
            print(f"Cars detected in video: {video_path}")

        # Apply non-maxima suppression to eliminate overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and annotations on the cars passing the line
        if len(indices) > 0:
            for i in indices.flatten():
                x_min, y_min, x_max, y_max = boxes[i]
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # Additional annotations or processing can be performed here

        for car_bbox in cars_on_line:
            x_min, y_min, x_max, y_max = car_bbox

            # Extract the car plate region from the frame
            car_plate_region = frame[y_min:y_max, x_min:x_max]

            # Preprocess the car plate region
            processed_plate = preprocess_image(car_plate_region)

            # Perform OCR on the preprocessed plate region
            plate_number = pytesseract.image_to_string(processed_plate)

            # Display the detected plate number
            print(f"License Plate Number: {plate_number}")

              # Calculate the height of the car bounding box
            if detection[3] == float('inf'):
                height = 20 # Replace default_height_value with the desired value
            else:
                height = int(detection[3] * frame.shape[0])
                   # Calculate the height of the car bounding box



        # Draw the line on the frame
        cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

        # Save image when a car is detected
        if len(cars_on_line) > 0:
            # Get the number of child nodes under the "Saman" node
            saman_ref = firebase_admin.db.reference('Saman')
            num_child_nodes = len(saman_ref.get().keys())

            # Generate the new node key as the number of child nodes + 1
            new_node_key = f"s{num_child_nodes + 1}"
            image_name = f"{new_node_key}.jpg"
            cv2.imwrite(image_name, frame)
            print(f"Image saved for video {video_path}: {image_name}")

            # Upload image to Firebase Storage
            blob = bucket.blob(image_name)
            blob.upload_from_filename(image_name)

            # Get the public URL of the uploaded image
            image_url = blob.public_url

            # Save image details to Firebase Realtime Database
            new_image_ref = db_ref.child(new_node_key)
            new_image_ref.set({
                'CarPlate': plate_number,
                'Time': datetime.now().strftime("%m/%d %H:%M:%S"),
                'Location': image_url
            })

            car_plate = "ABC123"  # Replace with the detected car plate
            location = "unknown"
            time = datetime.now().strftime("%m/%d %H:%M:%S")

            new_node_data = {
                'CarPlate': car_plate,
                'Location': location,
                'Picture': image_name,
                'Time': time
            }
            new_image_ref.set(new_node_data)

            print(f"Image details saved to Firebase Realtime Database for video {video_path}")
            image_counter += 1

        # Display the frame
        cv2.imshow('Frame', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close windows
    cap.release()
    cv2.destroyAllWindows()

# Create and start a thread for each video
threads = []
line_y = None
for video_path in video_paths:
    thread = threading.Thread(target=process_video, args=(video_path, line_y))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()

if __name__ == '__main__':
    app.run(port=8000)  # Run on port 8000
