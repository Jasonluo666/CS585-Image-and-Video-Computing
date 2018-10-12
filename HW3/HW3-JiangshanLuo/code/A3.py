import cv2
from template_matching import *

# Fine-tune following params for best performance
# blurring kernel size
BLURRING_KERNEL = 20

# absolute thresholding threshold
THRESH = 80

# NCC and SSD threshold
NCC_THRESH = 0.5
SSD_THRESH = 2000

# Threshold for area of skin
AREA_THRESH = 8000

# centroid queue size
QUEUE_SIZE = 100
DISTANCE_THRESH = 10000

# waving upper bound
wavingUpperBound = 60

# waving upper bound
wavingLowerBound = 25

# drawing threshold
drawThreshold = 61

def imgMax(img):
    maxMat = np.max(img.reshape(img.shape[0]*img.shape[1], 3), axis=1).reshape(img.shape[0], img.shape[1])
    return maxMat

def imgMin(img):
    minMat = np.min(img.reshape(img.shape[0]*img.shape[1], 3), axis=1).reshape(img.shape[0], img.shape[1])
    return minMat

# returns the @param img with only skin color (the rest of the pixels are black)
def skinDetect(img):
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    skin = np.zeros_like(img)
    maxMat = imgMax(img)
    minMat = imgMin(img)
    cond = (R > 95) & (G > 40) & (B > 20) &((maxMat - minMat) > 15) & (np.abs(R - G) > 15) & (R > G) & (R > B)
    skin[cond] = img[cond]
    return skin


def gesture_identifier(mirror=False):

    # the video stream 
    cam = cv2.VideoCapture(0)

    # window that displays the live video stream
    cv2.namedWindow('objects')

    # read template images from directory 
    data_path = abspath('./templates')
    data_list = [join(data_path, file) for file in listdir(data_path) if isfile(join(data_path, file)) and '.jpg' in file]
    
    # list of grayscale template image file names
    image_grayscale = [x for x in data_list if 'grayscale' in x]

    # list of the hand shape names 
    template_name_grayscale = [x for x in listdir(data_path) if 'grayscale' in x]
    
    # extract names of hand shapes
    for i in range(len(template_name_grayscale)):
        substrings = template_name_grayscale[i].split()
        name = ""
        for j in substrings[:-1]:
            name = name + " " + j
        template_name_grayscale[i] = name[1:]

    # for each grayscale template image in the directory
    # store the image in an list
    grayscale_templates = []
    for template in image_grayscale:
        img = cv2.imread(template)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        grayscale_templates.append(img.copy())

        # remove noise in image using largest contour
        grayscale_templates = remove_padding(grayscale_templates)

    ###########################################################################
    
    traces = []

    # create a structuring element for opening
    elipticalKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:

        # read the current frame
        ret_val, img = cam.read()

        # if not successful, exit program
        if ret_val == 0:
            raise Exception("Frame not found")

        # if the video stream is flipped, fix the image
        if mirror: 
            img = cv2.flip(img, 1)

        # remove any pixels in the frame that are not skin color
        skin = skinDetect(img)

        # morphology technique opening to remove noise 
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, elipticalKernel)

        # convert to grayscale
        grayscale = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY);

        # smoothing grayscale image
        grayscaleBlurred = cv2.blur(grayscale, (BLURRING_KERNEL, BLURRING_KERNEL))

        # thresholding blurred grayscale image
        retval, threshOutput = cv2.threshold(grayscaleBlurred, THRESH, 1, cv2.THRESH_BINARY)

        # find the contours in the image 
        im2, contours, hierarchy = cv2.findContours(threshOutput, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # initailize a list of bounding recs for each blob in the frame
        imgBounding = img.copy()
        boundingRecs = []

        # a list of a grayscale image for each blob found
        subimgGrayscaleBlurred =[]

        # for each blob in the frame 
        # determine if the blob is big enough to be a hand 
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > AREA_THRESH:
                rec = cv2.boundingRect(contour)

                # save the bounding box in the boundingRecs list
                boundingRecs.append(rec)
                x, y, w, h = rec

                # save the grayscale image 
                subimgGrayscaleBlurred.append(grayscaleBlurred[y:y+h, x:x+w])

        # if hand blobs exist in this frame 
        if len(subimgGrayscaleBlurred) > 0:
            centroids = []

            # use template matching
            shape_recognition_ncc, shape_value_ncc = template_matching(subimgGrayscaleBlurred, grayscale_templates, method='grayscale_ncc')
            
            # for each NNC value of a hand blob
            for index_ncc in range(len(subimgGrayscaleBlurred)):

                # determine if the blob is a hand shape 
                if shape_value_ncc[index_ncc] > NCC_THRESH:

                    # display hand shape name and bounding box
                    x, y, w, h = boundingRecs[index_ncc]
                    textY = y - 10
                    if textY < 0:
                        textY = y + h + 25
                    cv2.putText(imgBounding, template_name_grayscale[shape_recognition_ncc[index_ncc]], (x, textY),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0), 2)
                    cv2.rectangle(imgBounding, (x, y), (x + w, y + h), (0, 255, 0), 5, 8, 0)
                    
                    # save the centroid for tracking
                    centroids.append(np.array([x + int(w/2), y + int(h/2)]))

            # center evaluations that decides whether it is a center from previously tracked objects or a new object
            trace_update = [False for x in traces]
            
            for centroid in centroids:
                if len(traces) == 0:
                    traces.append([centroid])
                else:
                    dist = [np.sum((x[-1] - centroid) ** 2) for x in traces]
                    if np.min(dist) < DISTANCE_THRESH:
                        traces[np.argmin(dist)].append(centroid)
                        trace_update[np.argmin(dist)] = True
                    else:
                        traces.append([centroid])

            # if trace longer than threshold, pop first.
            # if not updated(tracked),drop first 1/4 of the traces
            for index in range(len(trace_update)):
                if len(traces[index]) > QUEUE_SIZE:
                    traces[index].pop(0)
                elif trace_update[index] == False:
                    if len(traces[index]) > int(QUEUE_SIZE / 2):
                        del traces[index][:int(QUEUE_SIZE / 2)]
                    else:
                        traces[index] = []
            traces = [x for x in traces if len(x) != 0]

            # set up filtering to decide if current tracked object is waving or drawing
            for trace in traces:
                trace = np.array(trace)
                avg = np.mean(np.array(trace), axis = 0)#.reshape()
                avgDelta = np.sum(np.abs(trace - avg))/len(trace)
                if wavingLowerBound < avgDelta < wavingUpperBound:
                    cv2.putText(imgBounding, 'waving', tuple(trace[-1]) ,cv2.FONT_HERSHEY_PLAIN,2,(0,255,0), 2)
                elif avgDelta > drawThreshold:
                    cv2.putText(imgBounding, 'drawing', tuple(trace[-1]) ,cv2.FONT_HERSHEY_PLAIN,2,(0,255,0), 2)
                    for index in range(1, len(trace)):
                       cv2.line(imgBounding,tuple(trace[index - 1]),tuple(trace[index]),(255,0,0),5)

        cv2.imshow('objects', imgBounding)

        if cv2.waitKey(1) == 27 :#
            break  # esc to quit
    cv2.destroyAllWindows()

def main():
    gesture_identifier(mirror=True)


if __name__ == '__main__':
    main()