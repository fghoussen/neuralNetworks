# Import library
import cv2
import sys
import argparse
import os
import wget
import numpy as np

def cmdLineArgs():
    # Create parser.
    parser = argparse.ArgumentParser(description='Publisher parser.')
    parser.add_argument('--videoID', type=int, default=0)
    parser.add_argument('--yoloConfidence', type=float, default=0.5)
    parser.add_argument('--nmsThreshold', type=float, default=0.3)

    return parser.parse_args()

def getYOLOCfg():
    # Download YOLO files.
    if not os.path.isfile('yolov3.weights'):
        wget.download('https://pjreddie.com/media/files/yolov3.weights')
    else:
        print('yolov3.weights has already been downloaded.')
    if not os.path.isfile('yolov3.cfg'):
        wget.download('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg')
    else:
        print('yolov3.cfg has already been downloaded.')
    if not os.path.isfile('coco.names'):
        wget.download('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names')
    else:
        print('coco.names has already been downloaded.')

    # Load the COCO class labels our YOLO model was trained on.
    labels = open('coco.names').read().strip().split("\n")

    # Initialize a list of colors to represent each possible class label.
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # load our YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    return colors, labels, net, ln

def runYOLODetection(frame, colors, labels, net, ln, args):
    # Construct a blob from the input frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Perform a forward pass of the YOLO object detector.
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # Initialize our lists of detected bounding boxes, confidences, and class IDs.
    boxes, confidences, classIDs = [], [], []

    # Loop over each of the layer outputs.
    (H, W) = frame.shape[:2]
    for output in layerOutputs:
        # Loop over each of the detections.
        for detection in output:
            # Extract the class ID and confidence (i.e., probability) of the current detection.
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter out weak predictions by ensuring the detected probability is greater than the minimum probability.
            if confidence > args.yoloConfidence:
                # Scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO
                # returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height.
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y)-coordinates to derive the top and and left corner of the bounding box.
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update our list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes.
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.yoloConfidence, args.nmsThreshold)

    # Ensure at least one detection exists.
    if len(idxs) > 0:
        # Loop over the indexes we are keeping.
        for i in idxs.flatten():
            # Extract the bounding box coordinates.
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw a bounding box rectangle and label on the frame.
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    # Get command line arguments.
    args = cmdLineArgs()

    # Get YOLO configuration files.
    colors, labels, net, ln = getYOLOCfg()

    # Capture video.
    vid = cv2.VideoCapture(args.videoID)
    while(True):
        # Capture the video frame by frame.
        ret, frame = vid.read()
        if not ret:
            continue

        # run YOLO detection.
        runYOLODetection(frame, colors, labels, net, ln, args)

        # Display the resulting frame.
        cv2.imshow('frame', frame)

        # Press 'q' to quit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the capture object.
    vid.release()
    # Destroy all the windows.
    cv2.destroyAllWindows()

# Main program.
if __name__ == '__main__':
    sys.exit(main())
