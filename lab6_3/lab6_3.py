import cv2
import numpy as np

# set video path
video = cv2.VideoCapture('vid.mp4')
# set video size
width, height = 600, 400


# func for read frame from video and resize it
def read_frame(video_input):
    pixels = video_input.read()[1]
    resize = cv2.resize(pixels, (width, height))
    return resize


# set number of frame
k = 1
while True:
    # read frame
    frame_original_cur = read_frame(video)
    # save original frame
    show_frame = read_frame(video)

    # convert to hsv
    frame_original_cur = cv2.cvtColor(frame_original_cur, cv2.COLOR_RGB2HSV)

    # threshold dolphin
    frame_original_cur = cv2.threshold(frame_original_cur, 190, 255, cv2.THRESH_BINARY)[1]

    # set color for dolphin and make mask, which will be used for bitwise operation for set color for dolphin
    lower_dolphin = np.array([0, 255, 0])
    upper_dolphin = np.array([255, 255, 255])
    mask = cv2.inRange(frame_original_cur, lower_dolphin, upper_dolphin)
    frame_original_cur = cv2.bitwise_and(frame_original_cur, frame_original_cur, mask=mask)

    # if it is first frame, select roi
    if k == 1:
        # show frame for select region of interest (roi)
        bbox = cv2.selectROI("Object Tracking", show_frame, False)
        # initialize tracker
        roi = frame_original_cur[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        # convert roi to hsv
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # calculate histogram for roi
        roi_hist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # normalize histogram
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        # set termination criteria for tracker
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # set hsv for current frame
    frame_hsv = cv2.cvtColor(frame_original_cur, cv2.COLOR_BGR2HSV)
    # calculate back projection of the frame, which means that we calculate probability of each pixel to be in roi
    frame_backproj = cv2.calcBackProject([frame_hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
    # apply camshift to get the new bounding box, find new position of roi
    ret, bbox = cv2.CamShift(frame_backproj, bbox, term_crit)
    # draw the new bounding box on the frame
    pts = cv2.boxPoints(ret)
    # convert float to int
    pts = np.int0(pts)
    # draw rectangle
    cv2.polylines(show_frame, [pts], True, (0, 255, 0), 2)

    # update number of frame
    k += 1

    # show original with rectangle and edited frames
    cv2.imshow('original', show_frame)
    cv2.imshow('edited', frame_original_cur)

    # exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
