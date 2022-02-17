# Import library.
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

    return parser.parse_args()

def getENetCfg():
    # Download ENet files.
    if not os.path.isfile('enet-classes.txt'):
        wget.download('https://raw.githubusercontent.com/ishacusp/ARGO_Labs/master/opencv-semantic-segmentation/enet-cityscapes/enet-classes.txt')
    else:
        print('enet-classes.txt has already been downloaded.')
    if not os.path.isfile('enet-colors.txt'):
        wget.download('https://raw.githubusercontent.com/ishacusp/ARGO_Labs/master/opencv-semantic-segmentation/enet-cityscapes/enet-colors.txt')
    else:
        print('enet-colors.txt has already been downloaded.')
    if not os.path.isfile('enet-model.net'):
        wget.download('https://raw.githubusercontent.com/ishacusp/ARGO_Labs/master/opencv-semantic-segmentation/enet-cityscapes/enet-model.net')
    else:
        print('enet-model.net has already been downloaded.')

    # Load the cityscapes classes our ENet model was trained on.
    classes = open('enet-classes.txt').read().strip().split("\n")

    # Initialize a list of colors to represent each possible class label.
    colors = open('enet-colors.txt').read().strip().split("\n")
    colors = [np.array(c.split(",")).astype("int") for c in colors]
    colors = np.array(colors, dtype="uint8")

    # Load our ENet neural network trained on cityscapes dataset.
    net = cv2.dnn.readNet('enet-model.net')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return classes, colors, net

def runENetSegmentation(frame, net, colors, legend):
    # Construct a blob from the input frame.
    # The original ENet input image dimensions was trained on was 1024x512, so, this shape is imposed.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (1024, 512), swapRB=True, crop=False)

    # Perform a forward pass of the ENet neural network.
    net.setInput(blob)
    output = net.forward()

    # Infer number of classes and the spatial dimensions of the mask image from the output array.
    (numClasses, height, width) = output.shape[1:4]

    # Our output class ID map will be num_classes x height x width in size, so we take the argmax to find
    # the class label with the largest probability for each and every (x, y)-coordinate in the image.
    classMap = np.argmax(output[0], axis=0)

    # Given the class ID map, we can map each of the class IDs to its corresponding color.
    mask = colors[classMap]

    # Resize the mask and class map such that its dimensions match the original size of the input image.
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    classMap = cv2.resize(classMap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Perform a weighted combination of the input image with the mask to form an output visualization
    segFrame = ((0.4 * frame) + (0.6 * mask)).astype("uint8")

    # Concatenate legend and segmented frame to get them side-by-side.
    segFrame = np.concatenate((legend, segFrame), axis=1)

    return segFrame

def main():
    # Get command line arguments.
    args = cmdLineArgs()

    # Get ENet configuration files.
    classes, colors, net = getENetCfg()

    # Capture video.
    legend = None
    vid = cv2.VideoCapture(args.videoID)
    while(True):
        # Capture the video frame by frame.
        ret, frame = vid.read()
        if not ret:
            continue

        # Create legend.
        if legend is None:
            # Initialize the legend visualization.
            legend = np.zeros((frame.shape[0], frame.shape[1]//3, 3), dtype="uint8")

            # Loop over the class names and colors.
            for (i, (name, color)) in enumerate(zip(classes, colors)):
                # Draw the class name and color on the legend.
                color = [int(c) for c in color]
                cv2.putText(legend, name, (5, (i * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(legend, (100, (i * 25)), (200, (i * 25) + 25), tuple(color), -1)

        # Run ENet segmentation.
        segFrame = runENetSegmentation(frame, net, colors, legend)

        # Display the resulting frames.
        cv2.imshow('Raw frame', frame)
        cv2.imshow('Segmented frame', segFrame)

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
