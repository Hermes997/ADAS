import cv2
import numpy as np

# from dynamikontrol import Module

cap = cv2.VideoCapture('calibratedSource1.mp4')
net = cv2.dnn.readNetFromDarknet("yolov4-tiny_custom.cfg", "yolov4-tiny_custom_final.weights")

CONFIDENCE = 0.1  # 차이냐 아니냐
THRESHOLD = 0.3  # ㅇ
CAR_WIDTH_TRESHOLD = 500

_, sample_img = cap.read()
cap_image_height, cap_image_width, _ = sample_img.shape
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


def Detect_object(img, net, CONFIDENCE, THRESHOLD):
    H, W, _ = img.shape

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

    # 390, 240, 90, 70 -> x,y,w,h



    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]

            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
            # cv2.putText(img,label + " " + confidence, (x,y+20),font,1,(0,0,255),1)
            # cv2.putText(img, text='%s %.2f %d' % (LABELS[class_ids[i]], confidences[i], w), org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            checkx1 = x + w / 3
            checkx2 = x + w - w / 3

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
    corners_original = np.array([[int(roi_image_width * warp_ratio_left), 0],
                                 [int(roi_image_width * warp_ratio_right), 0],
                                 [0, roi_image_height - 1],
                                 [roi_image_width - 1, roi_image_height - 1]], np.float32)

    corners_warp = np.array([[0, 0], [bird_view_width - 1, 0], [0, bird_view_height - 1],
                             [bird_view_width - 1, bird_view_height - 1]], np.float32)

    warp_trans_matrix = cv2.getPerspectiveTransform(corners_original, corners_warp)
    bird_view_size = (bird_view_width, bird_view_height)
    warp_image = cv2.warpPerspective(roi_image, warp_trans_matrix, bird_view_size)
    return warp_image


def Filter_lane(warp_image):
    # warp_image_hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    warp_image_hsl_hue, warp_image_hsl_luminance, warp_image_hsl_saturation = cv2.split(warp_image)


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


def Main():
    while cap.isOpened():
        ret, img = cap.read()
        detected_image = Detect_object(img, net, CONFIDENCE, THRESHOLD)
        equalized_image = Equalize_histogram_in_hsl(img)
        equalized_roi_image = Region_of_interest(equalized_image)
        warp_equalized_roi_image = Warp_image(equalized_roi_image)
        cap_image_gray_histogram = Draw_Histogram(img)
        count_bright = Count_bright(cap_image_gray_histogram)
        lane_image = Filter_lane_by_hsl(warp_equalized_roi_image, count_bright)
        cv2.imshow('img', detected_image)
        cv2.imshow('result', lane_image)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    Main()
