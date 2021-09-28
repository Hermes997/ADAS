import copy
import math

import cv2
import numpy as np


cap = cv2.VideoCapture('calibratedSource5.mp4')
net = cv2.dnn.readNetFromDarknet("yolov4-tiny_custom.cfg", "yolov4-tiny_custom_final.weights")
classes = ['', 'bus', 'rickshaw', 'motorbike', 'car', 'three wheelers (CNG)', 'pickup', 'minivan', 'suv', 'van',
           'truck', 'bicycle', 'policecar', 'ambulance', 'human hauler', 'wheelbarrow', 'minibus', 'auto rickshaw',
           'army vehicle', 'scooter', 'garbagevan']

_, sample_img = cap.read()
cap_image_height, cap_image_width, _ = sample_img.shape
handle_image_source = cv2.imread('handle.png', cv2.IMREAD_GRAYSCALE)
handle_image = cv2.resize(handle_image_source, dsize=(int(cap_image_height * 0.4), int(cap_image_height * 0.4)),
                          interpolation=cv2.INTER_CUBIC)
mask_image_inverted = cv2.bitwise_not(handle_image)
_, mask_image = cv2.threshold(handle_image[:, :], 1, 50, cv2.THRESH_BINARY)
handle_image_height, handle_image_width = mask_image_inverted.shape


CONFIDENCE = 0.1  # 차이냐 아니냐
THRESHOLD = 0.3  # ㅇ
CAR_WIDTH_TRESHOLD = 500


chaser_flag = 0
p2_width = 0
p1_width = 0
p2_height = 0
p1_height = 0
count = 0
x = 0
y = 0
w = 0
h = 0
x1 = 0
y1 = 0
distance = 0
final_speed = 0
distance_str = ''
final_speed_str = ''
font = cv2.FONT_HERSHEY_PLAIN
relative_speed = 0



dark_image_side_height = int(cap_image_height * 0.4)
dark_image_side_width = int(cap_image_width * 0.35)
dark_image_side = np.zeros((dark_image_side_height, dark_image_side_width, 3))
dark_image_side_left_offset_y = 0
dark_image_side_left_offset_x = 0
dark_image_side_right_offset_y = 0
dark_image_side_right_offset_x = cap_image_width - dark_image_side_width

roi_image_offset_y = int(cap_image_height * 0.666667)
roi_image_offset_x = int(cap_image_width * 0.210417)

roi_image_height = int(cap_image_height * 0.177778)
roi_image_width = int(cap_image_width * 0.514583)

warp_ratio_left = 0.285
warp_ratio_right = 0.722

threshold_value = 80000
threshold_value_shadow = 200

bird_view_height = 75
bird_view_width = 112
dark_image = np.zeros((bird_view_height, int(bird_view_width * 0.5), 3))

corners_original = np.array([[int(roi_image_width * warp_ratio_left), 0],
                             [int(roi_image_width * warp_ratio_right), 0],
                             [0, roi_image_height - 1],
                             [roi_image_width - 1, roi_image_height - 1]], np.float32)

corners_warp = np.array([[0, 0], [bird_view_width - 1, 0], [0, bird_view_height - 1],
                         [bird_view_width - 1, bird_view_height - 1]], np.float32)


def Detect_distance(car_w, car_x, img_cenx):  ##추가

    f_x = 13613.69404427889 * 0.375
    f_y = 14639.52311652638 * 0.375
    f = (f_x + f_y) / 2 / 3 / 2
    if car_x < img_cenx:
        car_x += car_w
    # 차길이 1900mm, 점선한개 8m, 차길이 4m, 중앙과 차사이 거리: w
    weig = 4 / 1.9  # 가중치
    car_h = car_w * weig  # 차 길이 픽셀
    k = abs(img_cenx - car_x)
    e = abs(car_h * k / (f - car_h))
    # x축으로 중심과 거리
    real_w = car_w - e  # 실제 차 범퍼 길이, car_w 인식되는 차길이
    car_distance = 1.9 * f / real_w

    return car_distance


def Detect_object(img, net, CONFIDENCE, THRESHOLD):
    H, W, _ = img.shape
    global p2_width
    global p1_width
    global p2_height
    global p1_height
    global count
    global x, y, w, h
    global distance
    global final_speed
    global font
    global final_speed_str
    global distance_str
    global relative_speed
    global x1
    global y1
    blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255., size=(416, 416), swapRB=True)
    net.setInput(blob)
    output = net.forward()

    boxes, confidences, class_ids = [], [], []

    for det in output:
        box = det[:4]
        scores = det[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONFIDENCE:
            cx, cy, w, h = box * np.array([W, H, W, H])
            x = cx - (w / 2)
            y = cy - (h / 2)

            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
            distance = round(Detect_distance(w, x, W / 2))
            distance_str = str(distance)

            if cap_image_width * 0.4 - y * cap_image_width / cap_image_height * 0.05 < x < cap_image_width * 0.5:
                relative_speed = ((p1_width - p2_width) + (p1_height - p2_height)) * y / cap_image_height
                final_speed_str = str(round(relative_speed, 1))
                p2_width = p1_width
                p1_width = w
                p2_height = p1_height
                p1_height = h
                x1, y1 = [x, y]
                count = 20

    if count > 0:
        if relative_speed >= 0:
            cv2.putText(img, 'Rel_speed :   ' + final_speed_str + 'km/h', (x1 - 50, y1 - 10), font, 1, (50, 255, 50), 1)
            cv2.putText(img, 'Level of dangerous', (x1 - 30, y1 - 40), font, 1, (170, 170, 255), 1)
        else:
            cv2.putText(img, 'Rel_speed : ' + final_speed_str + 'km/h', (x1 - 50, y1 - 10), font, 1, (50, 255, 50), 1)
            cv2.putText(img, 'Level of dangerous', (x1 - 30, y1 - 40), font, 1, (0, 0, 255), 1)
        cv2.putText(img, 'Distance :    ' + distance_str + 'm', (x1 - 30, y1 - 25), font, 1, (50, 255, 50), 1)

        count = count - 1

    return img


def Equalize_histogram_in_hsl(image):
    image[dark_image_side_left_offset_y:dark_image_side_left_offset_y + dark_image_side_height,
    dark_image_side_left_offset_x:dark_image_side_left_offset_x + dark_image_side_width] = dark_image_side

    image[dark_image_side_right_offset_y:dark_image_side_right_offset_y + dark_image_side_height,
    dark_image_side_right_offset_x:dark_image_side_right_offset_x + dark_image_side_width] = dark_image_side

    image_hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    image_hsl_hue, image_hsl_luminance, image_hsl_saturation = cv2.split(image_hsl)
    image_hsl_luminance_equalized = cv2.equalizeHist(image_hsl_luminance)
    image_hsl_saturation_equalized = cv2.equalizeHist(image_hsl_saturation)
    image_equalized = cv2.merge([image_hsl_hue, image_hsl_luminance_equalized, image_hsl_saturation_equalized])

    return image_equalized


def Region_of_interest(image):
    roi_image = image[roi_image_offset_y:roi_image_offset_y + roi_image_height,
                roi_image_offset_x:roi_image_offset_x + roi_image_width]
    return roi_image


def Warp_image(roi_image):
    global corners_original
    global corners_warp

    warp_trans_matrix = cv2.getPerspectiveTransform(corners_original, corners_warp)
    bird_view_size = (bird_view_width, bird_view_height)
    warp_image = cv2.warpPerspective(roi_image, warp_trans_matrix, bird_view_size)

    return warp_image


def Draw_Histogram(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_histogram = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    return image_gray_histogram


def Count_bright(image_gray_histogram):
    sum_lange_100to255 = 0
    for i in range(100, 256):
        sum_lange_100to255 += image_gray_histogram[i]
    return sum_lange_100to255


def Filter_lane_by_hsl(warp_image_hsl, sum_lange_100to255):
    warp_image_height = warp_image_hsl.shape[0]
    warp_image_width = warp_image_hsl.shape[1]
    threshold_ratio = sum_lange_100to255 / threshold_value
    if threshold_ratio > 1:
        threshold_ratio = 1
    shadow = 0
    warp_image_hsl_hue, warp_image_hsl_luminance, warp_image_hsl_saturation = cv2.split(warp_image_hsl)

    for i in range(0, warp_image_width):
        for j in range(0, int(warp_image_height * 0.25)):
            if warp_image_hsl_luminance[j, i] < 100 - 85 * threshold_ratio:
                shadow = shadow + 1
            if not ((10 + 10 * threshold_ratio < warp_image_hsl_hue[j, i] < 35
                     and 210 - 85 * threshold_ratio < warp_image_hsl_luminance[j, i] < 255
                     and warp_image_hsl_saturation[j, i] > 115 * threshold_ratio)
                    or (215 - 25 * threshold_ratio < warp_image_hsl_luminance[j, i] < 255)):
                warp_image_hsl[j, i] = 0
        # f shadow > threshold_value_shadow:
        # warning1.copyTo(AR_image.rowRange(0, warning1.rows).colRange(0, warning1.cols))
        for j in range(int(warp_image_height * 0.25), warp_image_height):
            if not ((10 + 10 * threshold_ratio < warp_image_hsl_hue[j, i] < 35
                     and 210 - 85 * threshold_ratio < warp_image_hsl_luminance[j, i] < 255
                     and warp_image_hsl_saturation[j, i] > 115 * threshold_ratio)
                    or (215 - 25 * threshold_ratio < warp_image_hsl_luminance[j, i] < 255)):
                warp_image_hsl[j, i] = 0
    warp_image_hsl[0:bird_view_height, int(bird_view_width * 0.25):int(bird_view_width * 0.75)] = dark_image

    return warp_image_hsl


def Remove_blemishes(img):
    kernel = np.ones((1, 1))
    clear_image1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    clear_image2 = cv2.morphologyEx(clear_image1, cv2.MORPH_CLOSE, kernel)

    return clear_image2


def Decide_lane(lane_image):
    lane_image_gray_height = lane_image.shape[0]
    lane_image_gray_width = lane_image.shape[1]
    lane_image_bgr = cv2.cvtColor(lane_image.copy(), cv2.COLOR_HLS2BGR)
    lane_image_gray = cv2.cvtColor(lane_image_bgr, cv2.COLOR_BGR2GRAY)
    return_image = lane_image_bgr.copy()
    c1 = 0
    for j in range(10, lane_image_gray_height - 10, 10):
        c = 0
        i1 = 0
        for i in range(int(lane_image_gray_width * 0.3), int(lane_image_gray_width * 0), -1):
            if int(lane_image_gray[j, i]) >= 50:
                c = c + 1

        for i in range(int(lane_image_gray_width * 0.3), int(lane_image_gray_width * 0.1), -1):
            i1 = i
            if int(lane_image_gray[j, i]) >= 50 and 2 < c < 10:
                cv2.rectangle(return_image, (i - 7, j - 7), (i + 3, j + 3), (0, 0, 255), 1)
                break

        c = 0
        k1 = 0
        for k in range(int(lane_image_gray_width * 0.7), lane_image_gray_width):
            if int(lane_image_gray[j, k]) >= 50:
                c = c + 1

        for k in range(int(lane_image_gray_width * 0.7), int(lane_image_gray_width * 0.9)):
            k1 = k
            if int(lane_image_gray[j, k]) >= 50 and 2 < c < 10:
                cv2.rectangle(return_image, (k + 2, j + 2), (k - 8, j - 8), (0, 0, 255), 1)
                break

        if i1 != int(lane_image_gray_width * 0.1) + 1 and k1 != int(lane_image_gray_width * 0.9) - 1:
            cv2.rectangle(return_image, (i1, j), (k1, j + 2), (0, 100, 0), 2)
        elif i1 != int(lane_image_gray_width * 0.1) + 1:
            cv2.rectangle(return_image, (i1, j), (i1 + int(lane_image_gray_width * 0.67), j + 2), (0, 100, 0), 2)
        elif k1 != int(lane_image_gray_width * 0.9) - 1:
            cv2.rectangle(return_image, (k1 - int(lane_image_gray_width * 0.67), j), (k1, j + 2), (0, 100, 0), 2)

        if lane_image_gray_height * 0.5 < j < lane_image_gray_height * 0.7:
            if i1 > lane_image_gray_width * 0.2 or k1 < lane_image_gray_width * 0.8:
                c1 = c1 + 1

        if lane_image_gray_height * 0.7 < j < lane_image_gray_height - 1:
            if i1 > lane_image_gray_width * 0.2 or k1 < lane_image_gray_width * 0.8:
                c1 = c1 + 2

    if c1 >= 3:
        ...

    return return_image


def To_ar_image_with_lane(image_source, lane_image):
    global corners_original
    global corners_warp

    ar_component_matrix = cv2.getPerspectiveTransform(corners_warp, corners_original)
    ar_image_size = (roi_image_width, roi_image_height)
    ar_component = cv2.warpPerspective(lane_image, ar_component_matrix, ar_image_size)
    set_box_image = np.zeros((cap_image_height, cap_image_width, 3), dtype=image_source.dtype)
    set_box_image[roi_image_offset_y:roi_image_offset_y + roi_image_height,
    roi_image_offset_x:roi_image_offset_x + roi_image_width] = ar_component
    ar_image = image_source + set_box_image
    return ar_image


def Draw_hough_line_image(image, source):
    global mask_image_inverted
    global font
    global cap_image_width
    global cap_image_height
    image_height = image.shape[0]
    image_width = image.shape[1]
    line_image = np.zeros((image_height, image_width, 3), image.dtype)
    canny_image = cv2.Canny(image, 100, 200)
    lines = cv2.HoughLines(canny_image, 1, 3.141592 / 180, 40)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1_x = int(x0 + 1000 * (-b))
            pt1_y = int(y0 + 1000 * a)
            pt2_x = int(x0 - 1000 * (-b))
            pt2_y = int(y0 - 1000 * a)
            cv2.line(line_image, (pt1_x, pt1_y), (pt2_x, pt2_y), (255, 0, 0), 1, 8)

        M = cv2.getRotationMatrix2D((handle_image_height / 2, handle_image_width / 2),
                                    - (1 / lines[0][0][0] * 400), 1)
        dst = cv2.warpAffine(mask_image_inverted, M, (handle_image_height, handle_image_width))

        roi = source[int(cap_image_height * 0.6):int(cap_image_height * 0.6) + handle_image_height,
                    int(cap_image_width * 0.1):int(cap_image_width * 0.1) + handle_image_width]

        bg = cv2.bitwise_and(roi, roi, mask=mask_image)

        dst_a = cv2.bitwise_and(dst, dst, mask=mask_image_inverted)
        dst_a = cv2.cvtColor(dst_a, cv2.COLOR_GRAY2BGR)
        added = dst_a + bg
        source[int(cap_image_height * 0.6):int(cap_image_height * 0.6) + handle_image_height,
            int(cap_image_width * 0.1):int(cap_image_width * 0.1) + handle_image_width] = added
    else:
        cv2.putText(source, 'Warning : Cannot find line',
                    (int(cap_image_width * 0.25), int(cap_image_height * 0.9)), font, 1, (50, 50, 255), 1)


def Main():
    while cap.isOpened():
        ret, img = cap.read()
        ar_image = copy.deepcopy(img)
        equalized_image = Equalize_histogram_in_hsl(img)
        equalized_roi_image = Region_of_interest(equalized_image)
        warp_equalized_roi_image = Warp_image(equalized_roi_image)
        cap_image_gray_histogram = Draw_Histogram(img)
        count_bright = Count_bright(cap_image_gray_histogram)
        lane_image = Filter_lane_by_hsl(warp_equalized_roi_image, count_bright)
        not_blemishes_image = Remove_blemishes(lane_image)
        Draw_hough_line_image(not_blemishes_image, ar_image)
        decided_lane = Decide_lane(not_blemishes_image)
        detected_image = Detect_object(ar_image, net, CONFIDENCE, THRESHOLD)
        To_ar_image_with_lane(ar_image, decided_lane)

        cv2.imshow('detected_image', detected_image)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    Main()
