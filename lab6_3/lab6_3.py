import cv2
import numpy as np

video = cv2.VideoCapture('vid.mp4')
width, height = 600, 400


def read_frame(video_input):
    pixels = video_input.read()[1]
    resize = cv2.resize(pixels, (width, height))
    return resize


def analyze_frames(prev_frame, cur_frame, next_frame):
    diff_frames_1 = cv2.absdiff(next_frame, cur_frame)
    diff_frames_2 = cv2.absdiff(cur_frame, prev_frame)
    return cv2.bitwise_and(diff_frames_1, diff_frames_2)


frame_original_prev = read_frame(video)
frame_original_cur = read_frame(video)
frame_original_next = read_frame(video)
k = 1
contours_prev = []
contours_cur = []
hierarchy_prev = []
hierarchy_cur = []
while True:
    show_frame = read_frame(video)

    frame_original_prev = cv2.cvtColor(frame_original_prev, cv2.COLOR_RGB2HSV)
    frame_original_cur = cv2.cvtColor(frame_original_cur, cv2.COLOR_RGB2HSV)
    frame_original_next = cv2.cvtColor(frame_original_next, cv2.COLOR_RGB2HSV)

    frame_original_prev = cv2.threshold(frame_original_prev, 190, 255, cv2.THRESH_BINARY)[1]
    frame_original_cur = cv2.threshold(frame_original_cur, 190, 255, cv2.THRESH_BINARY)[1]
    frame_original_next = cv2.threshold(frame_original_next, 190, 255, cv2.THRESH_BINARY)[1]

    # lower_dolphin = np.array([0, 0, 0])
    # upper_dolphin = np.array([255, 255, 255])
    # mask = cv2.inRange(frame_original_prev, lower_dolphin, upper_dolphin)
    # frame_original_prev = cv2.bitwise_and(frame_original_prev, frame_original_prev, mask=mask)
    # mask = cv2.inRange(frame_original_cur, lower_dolphin, upper_dolphin)
    # frame_original_cur = cv2.bitwise_and(frame_original_cur, frame_original_cur, mask=mask)
    # mask = cv2.inRange(frame_original_next, lower_dolphin, upper_dolphin)
    # frame_original_next = cv2.bitwise_and(frame_original_next, frame_original_next, mask=mask)

    frame_diff = analyze_frames(frame_original_prev, frame_original_cur, frame_original_next)

    frame_original_prev = frame_original_cur
    frame_original_cur = frame_original_next
    frame_original_next = read_frame(video)

    frame_diff_save = frame_diff.copy()

    frame_diff = cv2.Canny(frame_diff, 80, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    frame_diff = cv2.morphologyEx(frame_diff, cv2.MORPH_CLOSE, kernel)

    if k == 1:
        contours_prev, hierarchy_prev = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_cur, hierarchy_cur = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours_prev = contours_cur
        hierarchy_prev = hierarchy_cur
        contours_cur, hierarchy_cur = cv2.findContours(frame_diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours_cur)):
        print(len(contours_cur))
        if cv2.contourArea(contours_cur[i]) < 1000 or cv2.contourArea(contours_cur[i]) > 10000:
            continue
        cv2.drawContours(show_frame, contours_cur, i, (0, 255, 0), 3)

    # rectangles_prev = []
    # rectangles_cur = []
    #
    # for contour in contours_prev:
    #     approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    #     rect = cv2.boundingRect(approx)
    #     rectangles_prev.append(rect)
    #     # if cv2.contourArea(contour) < 1000 or cv2.contourArea(contour) > 10000:
    #     #     continue
    #     # cv2.rectangle(show_frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 255, 2)
    #
    # for contour in contours_cur:
    #     approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    #     rect = cv2.boundingRect(approx)
    #     rectangles_cur.append(rect)
    #     # if cv2.contourArea(contour) < 1000 or cv2.contourArea(contour) > 10000:
    #     #     continue
    #     # cv2.rectangle(show_frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 255, 2)

    for i in range(len(contours_prev)):
        pass



    k += 1

    cv2.imshow('original', show_frame)
    cv2.imshow('edited', frame_original_cur)
    cv2.imshow('frame_diff', frame_diff_save)
    cv2.imshow('frame_diff with filters', frame_diff)

    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
