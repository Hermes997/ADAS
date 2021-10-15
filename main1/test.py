import copy
import math

import cv2
import numpy as np

# 비디오 소스
cap = cv2.VideoCapture('calibratedSource1.mp4')
# 다크넷 기반 프레임워크로 딥러닝을 구동하기위한 cfg파일과 weight파일 불러오기
net = cv2.dnn.readNetFromDarknet("yolov4-tiny_custom.cfg", "yolov4-tiny_custom_final.weights")
# 탐지하고자 하는 오브젝트 종류들
classes = ['', 'bus', 'rickshaw', 'motorbike', 'car', 'three wheelers (CNG)', 'pickup', 'minivan', 'suv', 'van',
           'truck', 'bicycle', 'policecar', 'ambulance', 'human hauler', 'wheelbarrow', 'minibus', 'auto rickshaw',
           'army vehicle', 'scooter', 'garbagevan']

# 핸들이미지를 불러오고 AR영상에 띄울수 있도록 크기조정, 이미지 합성 사전작업
_, sample_img = cap.read()
cap_image_height, cap_image_width, _ = sample_img.shape
handle_image_source = cv2.imread('handle.png', cv2.IMREAD_GRAYSCALE)
handle_image = cv2.resize(handle_image_source, dsize=(int(cap_image_height * 0.4), int(cap_image_height * 0.4)),
                          interpolation=cv2.INTER_CUBIC)
handle_image_height, handle_image_width = handle_image.shape

# 딥러닝에서 판단한 신뢰율, 필터링할 임계값
CONFIDENCE = 0.1
THRESHOLD = 0.3

# 상대속도 측정을 위한 칸토어 길이정보 저장 변수선언
p2_width = 0
p1_width = 0
p2_height = 0
p1_height = 0

# 오브젝트가 일시적으로 탐지되지 않았을 때 탐지정보를 유지할 프레임 수 변수선언
count = 0

# 위험발생 탐지시 출력 유지 프레임 수 변수선언
count_warning = 0

# 실시간 오브젝트의 위치 및 크기 변수선언
x = 0
y = 0
w = 0
h = 0

# 탐지된 차량의 정보출력 텍스트의 위치 오프셋 변수선언
x1 = 0
y1 = 0

# 상대거리, 상대속도, ttc 측정 및 출력에 사용되는 변수 변수선언
distance = 0
distance_str = ''
relative_speed = 0
time_to_collision = 0

# 추후에 실제 차량의 속도를 불러올수 있게 될 경우 상대속도 + 현재 나의속도의 결과값 변수선언
final_speed_str = ''
final_speed = 0

# 출력 텍스트 폰트 설정
font = cv2.FONT_HERSHEY_PLAIN

# EqualHist를 할 때 주요 영역만을 적용시키기 위한 마스크이미지(검정색)적용
dark_image_side_height = int(cap_image_height * 0.4)
dark_image_side_width = int(cap_image_width * 0.35)
dark_image_side = np.zeros((dark_image_side_height, dark_image_side_width, 3))
dark_image_side_left_offset_y = 0
dark_image_side_left_offset_x = 0
dark_image_side_right_offset_y = 0
dark_image_side_right_offset_x = cap_image_width - dark_image_side_width

# 입력 이미지에 대한 roi 이미지 상대 위치 오프셋
roi_image_offset_y = int(cap_image_height * 0.666667)
roi_image_offset_x = int(cap_image_width * 0.210417)

# 입력 이미지에 대한 roi 이미지의 상대크기
roi_image_height = int(cap_image_height * 0.177778)
roi_image_width = int(cap_image_width * 0.514583)

# warping결과 이미지의 크기(단위 : 픽셀)설정
bird_view_height = 75
bird_view_width = 112

# roi 이미지를 birdview를 할때 warping 비율 (left은 0, right는 1에 가까워질수록 원본 영상과 가까워짐)
# 적절한 비율 설정 시 차선의 비스듬한 이미지를 상공에서 도로를 본 모습처럼 차선이 평행선으로 보임
warp_ratio_left = 0.285
warp_ratio_right = 0.722

# birdview를 위한 warping 위치 초기화
corners_original = np.array([[int(roi_image_width * warp_ratio_left), 0],
                             [int(roi_image_width * warp_ratio_right), 0],
                             [0, roi_image_height - 1],
                             [roi_image_width - 1, roi_image_height - 1]], np.float32)
corners_warp = np.array([[0, 0], [bird_view_width - 1, 0], [0, bird_view_height - 1],
                         [bird_view_width - 1, bird_view_height - 1]], np.float32)

# 이미지 밝기판단을 위한 기준 오프셋
threshold_value = 80000

# 차선 가운데는 관심구역이 아니므로 마스크이미지(검은색) 적용
dark_image = np.zeros((bird_view_height, int(bird_view_width * 0.5), 3))

# 핸들각도 조절
p1_handle_angle = 0
p2_handle_angle = 0

# 오브젝트 추적을 위한 변수, 배열
Prev = []
Pcheck = 0
Addbox = []
Tracker = []
DETECT_CAR = []
prevTime = 0


def Set_roi_image_offset_y(pos):
    pass


def Set_roi_image_offset_x(pos):
    pass


def Set_warp_ratio_left(pos):
    pass


def Set_warp_ratio_right(pos):
    pass


cv2.namedWindow('final_result')
cv2.createTrackbar('Set_roi_image_offset_y', 'final_result', 450, 750, Set_roi_image_offset_y)
cv2.createTrackbar('Set_roi_image_offset_x', 'final_result', 100, 300, Set_roi_image_offset_x)
cv2.createTrackbar('Set_warp_ratio_left', 'final_result', 100, 400, Set_warp_ratio_left)
cv2.createTrackbar('Set_warp_ratio_right', 'final_result', 600, 900, Set_warp_ratio_right)

cv2.setTrackbarPos('Set_roi_image_offset_y', 'final_result', 666)
cv2.setTrackbarPos('Set_roi_image_offset_x', 'final_result', 210)
cv2.setTrackbarPos('Set_warp_ratio_left', 'final_result', 285)
cv2.setTrackbarPos('Set_warp_ratio_right', 'final_result', 722)


# 검출된 차량의 칸토어의 크기를 기준으로 한 내 차와의 상대거리 측정
def Detect_distance(car_w, car_x, car_h, car_y, img_cenx):

    f=1162
    if car_x<img_cenx:
        car_x+=car_w
    #차길이 1900mm, 점선한개 8m, 차길이 4m, 중앙과 차사이 거리: w
    rw = 1.9
    weig = 4/1.9 #가중치
    if car_h + 5 >= car_w:
        weig = 3.5/1.5
        rw = 1.5
    car_h = car_w*weig #차 길이 픽셀
    k = abs(img_cenx - car_x)
    e = abs(car_h*k/(f-car_h))
     #x축으로 중심과 거리
    real_w = car_w - e  #실제 차 범퍼 길이, car_w 인식되는 차길이
    car_distance = rw * f / real_w
    #print("Error:", e, ", 이미지중심 ", img_cenx, ", 차중심", car_x, ", 차넓이", car_w, ", real_w:", real_w, "f: ", f, "D: ", car_distance)

    return car_distance


def aoi(a, b, c, d):
    x, y = 0, 1
    w = min(b[x], d[x]) - max(a[x], c[x])
    h = min(a[y], c[y]) - max(b[y], d[y])
    if min(w, h) > 0:
        return w * h
    else:
        return 0


def check(b1, b2, Threshold):
    a = [b1[0], b1[1] + b1[3]]
    b = [b1[0] + b1[2], b1[1]]
    c = [b2[0], b2[1] + b2[3]]
    d = [b2[0] + b2[2], b2[1]]
    x, y = 0, 1
    area1 = (b[x] - a[x]) * (a[y] - b[y])
    area2 = (d[x] - c[x]) * (c[y] - d[y])
    intersect = aoi(a, b, c, d)
    ratio1 = abs(intersect / area1)
    ratio2 = abs(intersect / area2)
    # print(ratio1, ratio2)
    if ratio1 > Threshold or ratio2 > Threshold:
        return True
    else:
        return False


def Change_rec(box1, box2):  # box1 는 이전, box2는 새로 검출 2-1
    ok = 0
    new_rec = []
    for b2 in box2:
        for b1 in box1:
            if check(b1, b2, 0.6):
                ok = 1
                break
        if ok != 1:
            new_rec.append(b2)
            ok = 0
        ok = 0
    return new_rec


def remove_rec(box1, box2):
    ok = 0
    for b1 in box1:
        for b2 in box2:
            if check(b1, b2, 0.8):
                if ok != 0 or b2[0] <= 20 or b2[1] <= 20:
                    box2.remove(b2)
                    ok += 1
        ok = 0

    temp = []
    for i in range(0, len(box2)):
        for j in range(i + 1, len(box2)):
            if check(box2[i], box2[j], 0.8):
                temp.append(j)

    for i in temp:
        box2.remove(box2[i])

    return box2


def remove_single(box2):
    temp = []
    for i in range(0, len(box2)):
        for j in range(i + 1, len(box2)):
            if check(box2[i], box2[j], 0.3):
                temp.append(j)

    temp = list(set(temp))
    tbox = copy.deepcopy(box2)
    for i in temp:
        tbox.remove(box2[i])
    return tbox


def Previous_check(pbox, lbox, H, W):
    # H, W, _ = img.shape
    box = Change_rec(pbox, lbox)
    for i in box:
        centx = i[0] + i[2]/2
        centy = i[1] + i[3]/2
        #일점 범위에서만 트레킹 되도록 설정 + 아래에 초록색은 트래킹 안함
        if  centx < 70 or centx > W-70 or centy > 230 or centy < 100:
            box.remove(i)
    return box


def traking_car(boxes, img):
    trackrec = []
    tracker = cv2.TrackerMIL_create()
    TrackingState = 0
    TrackingROI = (0, 0, 0, 0)
    for box in boxes:
        x, y, w, h = box
        TrackingROI = (x, y, w, h)
        ok = tracker.init(img, TrackingROI)
        ok, TrackingROI = tracker.update(img)
        TrackingROI = tuple([int(_) for _ in TrackingROI])
        if ok:
            trackrec.append(TrackingROI)
        else:
            break
    return trackrec


# yolo4를 이용한 오브젝트(타 차량) 검출
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
    global count_warning
    global time_to_collision
    global Ccheck
    global Prev
    global Pcheck
    global Addbox
    global Tracker
    global DETECT_CAR
    global prevTime


    # 입력 영상을 blop객체로 변환하여 추론 진행
    # 0~1 정규화된 입력영상으로 학습된 파일이므로 scalefactor=1 / 255.
    # 출력영상크기 설정(size=(416, 416)), 입력 영상의 R,B채널 서로 바꾸기(swapRB=True))
    blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255., size=(416, 416), swapRB=True)
    net.setInput(blob)
    output = net.forward()

    # 탐지된 오브젝트의 속성 배열
    boxes, confidences, class_ids = [], [], []

    # 추론 결과를 저장
    for det in output:
        box = det[:4]
        scores = det[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # 일정이상 신뢰율을 가져야 오브젝트의 칸토어 정보 저장
        if confidence > CONFIDENCE:
            cx, cy, w, h = box * np.array([W, H, W, H])
            x = cx - (w / 2)
            y = cy - (h / 2)

            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    Addbox = Change_rec(boxes, DETECT_CAR)
    if len(Tracker) == 0:
        Tracker = Addbox
    else:
        for i in Addbox:
            Tracker.append(i)

    # 트래킹하다 다시 검출하면 없앰
    Tracker = Previous_check(boxes, Tracker, H, W)
    # 트래킹 중복(겹침) 방지
    Tracker = remove_single(Tracker)
    # 트래킹 수행
    Tracker = traking_car(Tracker, img)

    # 저장된 오브젝트의 칸토어 정보를 통해 직사각형 모양으로 출력
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
            distance = round(Detect_distance(w, x, h, y, W / 2))
            distance_str = str(distance)

            # 상대속도 측정을 위해 실시간 칸토어 정보값을 전역변수에 저장
            if cap_image_width * 0.4 - y * cap_image_width / cap_image_height * 0.05 < x < cap_image_width * 0.5:
                relative_speed = ((p1_width - p2_width) + (p1_height - p2_height)) * y / cap_image_height
                final_speed_str = str(round(relative_speed, 1))
                p2_width = p1_width
                p1_width = w
                p2_height = p1_height
                p1_height = h
                x1, y1 = [x, y]

                # 오브젝트를 놓치더라도 20프레임만큼 오브젝트 정보를 출력유지
                count = 20

    for j in Tracker:
        x1, y1, w1, h1 = j
        distance = round(Detect_distance(w1, x1, h1, y1, W / 2))
        distance_str = str(distance)
        if distance < 30:
            cv2.rectangle(img, pt1=(x1, y1), pt2=(x1 + w1, y1 + h1), color=(0, 0, 255), thickness=2)
            cv2.putText(img, "tracking mode", (x1 - 30, y1 + h1), font, 1, (0, 0, 255), 1)
        elif distance < 80:
            cv2.rectangle(img, pt1=(x1, y1), pt2=(x1 + w1, y1 + h1), color=(0, 102, 255), thickness=2)
            cv2.putText(img, "tracking mode", (x1 - 30, y1 + h1), font, 1, (0, 102, 255), 1)
        else:
            pass  # 초록색은 검출 안되게함
            # cv2.rectangle(img, pt1=(x1, y1), pt2=(x1 + w1, y1 + h1), color=(0, 255, 0), thickness=2)
            # cv2.putText(img,distance + 'm '+ "tracking mode", (x1,y1+20),font,1,(0,255,0),1)

    DETECT_CAR = boxes

    if len(boxes) != 0:
        Pcheck += 1
    if Pcheck == 20:
        Prev = Tracker
        Pcheck = 0
        Tracker = Previous_check(Prev, Tracker,H,W)

    # 탐지된 차량의 상대거리, 상대속도, 위험도 출력
    if count > 0:
        if int(distance) < 30:
            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
            cv2.putText(img, 'Distance :    ' + distance_str + 'm', (x1 - 30, y1 - 25), font, 1, (0, 0, 255), 1)
        elif int(distance) < 80:
            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 102, 255), thickness=2)
            cv2.putText(img, 'Distance :    ' + distance_str + 'm', (x1 - 30, y1 - 25), font, 1, (0, 102, 255), 1)
        else:
            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
            cv2.putText(img, 'Distance :    ' + distance_str + 'm', (x1 - 30, y1 - 25), font, 1, (0, 255, 0), 1)

        if relative_speed >= 0:
            cv2.putText(img, 'Rel_speed :   ' + final_speed_str + 'km/h', (x1 - 50, y1 - 10), font, 1, (50, 255, 50), 1)
            if count_warning == 0:
                cv2.putText(img, 'TTC : far away', (x1 - 30, y1 - 40), font, 1, (50, 255, 50), 1)
            elif count_warning > 0:
                cv2.putText(img, 'TTC : ' + str(time_to_collision) + 'sec', (x1 - 30, y1 - 40), font, 1, (0, 0, 255), 1)
                count_warning = count_warning - 1
        else:
            count_warning = 10
            time_to_collision = int(distance / ((-relative_speed) * 3.6))
            cv2.putText(img, 'Rel_speed : ' + final_speed_str + 'km/h', (x1 - 50, y1 - 10), font, 1, (50, 255, 50), 1)
            cv2.putText(img, 'TTC : ' + str(time_to_collision) + 'sec', (x1 - 30, y1 - 40), font, 1, (0, 0, 255), 1)

        count = count - 1

    return img


# 열악한 환경에서도 차선 검출이 쉽게 될 수 있도록 이미지 히스토그램정보를 평탄화함으로써 이미지의 contrast를 높임
# 평탄화할 이미지의 관심구역을 설정하여 contrast를 높이는데 필요없는 구역은 검은이미지로 덧씌워 삭제
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


# 이미지의 roi를 설정함으로써 차선이 유력하게 존재할 구역만을 남김
def Region_of_interest(image):
    global roi_image_offset_y
    global roi_image_offset_x
    roi_image = image[roi_image_offset_y:roi_image_offset_y + roi_image_height,
                roi_image_offset_x:roi_image_offset_x + roi_image_width]
    roi_image_bgr = cv2.cvtColor(copy.deepcopy(roi_image), cv2.COLOR_HLS2BGR)
    cv2.imshow('roi_image_bgr', roi_image_bgr)
    return roi_image


# roi가 된 차선이미지를 birdview로 보기위해 이미지 warping처리
def Warp_image(roi_image):
    global corners_original
    global corners_warp
    warp_trans_matrix = cv2.getPerspectiveTransform(corners_original, corners_warp)
    bird_view_size = (bird_view_width, bird_view_height)
    warp_image = cv2.warpPerspective(roi_image, warp_trans_matrix, bird_view_size)
    warp_image_bgr = cv2.cvtColor(copy.deepcopy(warp_image), cv2.COLOR_HLS2BGR)
    cv2.imshow('warp_image_bgr', warp_image_bgr)
    return warp_image


# 밝기 변화에 대비하기 위한 이미지 히스토그램 분석
def Draw_Histogram(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_histogram = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    return image_gray_histogram


# 측정된 이미지의 히스토그램 정보를 토대로 이미지의 밝기를 판단
def Count_bright(image_gray_histogram):
    sum_lange_100to255 = 0
    for i in range(100, 256):
        sum_lange_100to255 += image_gray_histogram[i]
    return sum_lange_100to255


#  판단된 이미지의 밝기의 정보로 차선 필터링
def Filter_lane_by_hsl(warp_image_hsl, sum_lange_100to255):
    warp_image_height = warp_image_hsl.shape[0]
    warp_image_width = warp_image_hsl.shape[1]
    threshold_ratio = sum_lange_100to255 / threshold_value
    if threshold_ratio > 1:
        threshold_ratio = 1
    shadow = 0
    warp_image_hsl_hue, warp_image_hsl_luminance, warp_image_hsl_saturation = cv2.split(warp_image_hsl)

    # 평탄화된 이미지의 hls정보를 기준으로 주황색 차선의 특성, 흰색 차선의 특성 두가지 모두 고려한 필터 적용
    for i in range(0, warp_image_width):
        for j in range(0, int(warp_image_height * 0.25)):
            if warp_image_hsl_luminance[j, i] < 100 - 85 * threshold_ratio:
                shadow = shadow + 1
            if not ((10 + 10 * threshold_ratio < warp_image_hsl_hue[j, i] < 35
                     and 210 - 85 * threshold_ratio < warp_image_hsl_luminance[j, i] < 255
                     and warp_image_hsl_saturation[j, i] > 115 * threshold_ratio)
                    or (215 - 25 * threshold_ratio < warp_image_hsl_luminance[j, i] < 255)):
                warp_image_hsl[j, i] = 0
        for j in range(int(warp_image_height * 0.25), warp_image_height):
            if not ((10 + 10 * threshold_ratio < warp_image_hsl_hue[j, i] < 35
                     and 210 - 85 * threshold_ratio < warp_image_hsl_luminance[j, i] < 255
                     and warp_image_hsl_saturation[j, i] > 115 * threshold_ratio)
                    or (215 - 25 * threshold_ratio < warp_image_hsl_luminance[j, i] < 255)):
                warp_image_hsl[j, i] = 0
    warp_image_hsl[0:bird_view_height, int(bird_view_width * 0.25):int(bird_view_width * 0.75)] = dark_image

    return warp_image_hsl


# 이미지 잡티 제거
def Remove_blemishes(img):
    kernel = np.ones((1, 1))
    clear_image1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    clear_image2 = cv2.morphologyEx(clear_image1, cv2.MORPH_CLOSE, kernel)

    return clear_image2


# 검출된 차선 이미지로 차선의 영역 판단
def Decide_lane(lane_image, source):
    lane_image_gray_height = lane_image.shape[0]
    lane_image_gray_width = lane_image.shape[1]
    lane_image_bgr = cv2.cvtColor(lane_image.copy(), cv2.COLOR_HLS2BGR)
    lane_image_gray = cv2.cvtColor(lane_image_bgr, cv2.COLOR_BGR2GRAY)
    return_image = lane_image_bgr.copy()

    # 차선 이탈 판단에 이용되는 변수
    departure_count = 0

    # 핸들이미지 출력 조건 카운트 변수
    count_lane = 0

    for j in range(10, lane_image_gray_height - 10, 10):
        c = 0
        i1 = 0

        # 왼쪽 차선 검출
        for i in range(int(lane_image_gray_width * 0.3), int(lane_image_gray_width * 0), -1):
            if int(lane_image_gray[j, i]) >= 50:
                c = c + 1

        for i in range(int(lane_image_gray_width * 0.3), int(lane_image_gray_width * 0.1), -1):
            i1 = i
            if int(lane_image_gray[j, i]) >= 50 and 2 < c < 10:
                cv2.rectangle(return_image, (i - 7, j - 7), (i + 3, j + 3), (0, 0, 255), 1)
                count_lane = count_lane + 1
                break

        # 차선 이탈 판단에 이용되는 변수
        c = 0
        k1 = 0

        # 오른쪽 차선 검출
        for k in range(int(lane_image_gray_width * 0.7), lane_image_gray_width):
            if int(lane_image_gray[j, k]) >= 50:
                c = c + 1

        for k in range(int(lane_image_gray_width * 0.7), int(lane_image_gray_width * 0.9)):
            k1 = k
            if int(lane_image_gray[j, k]) >= 50 and 2 < c < 10:
                cv2.rectangle(return_image, (k + 2, j + 2), (k - 8, j - 8), (0, 0, 255), 1)
                count_lane = count_lane + 1
                break

        # 차선이 오른쪽, 왼쪽 한쪽차선만 검출될때, 두쪽 모두 검출될때의 경우로 나누어 동작후 영역 출력
        if i1 != int(lane_image_gray_width * 0.1) + 1 and k1 != int(lane_image_gray_width * 0.9) - 1:
            cv2.rectangle(return_image, (i1, j), (k1, j + 2), (0, 100, 0), 2)
        elif i1 != int(lane_image_gray_width * 0.1) + 1:
            cv2.rectangle(return_image, (i1, j), (i1 + int(lane_image_gray_width * 0.67), j + 2), (0, 100, 0), 2)
        elif k1 != int(lane_image_gray_width * 0.9) - 1:
            cv2.rectangle(return_image, (k1 - int(lane_image_gray_width * 0.67), j), (k1, j + 2), (0, 100, 0), 2)

        # 차선 이탈 판단, 차량과 가까운 지점일수록 이탈판단 가중치가 높음
        if lane_image_gray_height * 0.5 < j < lane_image_gray_height * 0.7:
            if i1 > lane_image_gray_width * 0.2 or k1 < lane_image_gray_width * 0.8:
                departure_count = departure_count + 1
        if lane_image_gray_height * 0.7 < j < lane_image_gray_height - 1:
            if i1 > lane_image_gray_width * 0.2 or k1 < lane_image_gray_width * 0.8:
                departure_count = departure_count + 2

    # 차선이탈판단 조건 만족시 경고 출력
    if departure_count >= 3:
        cv2.putText(lane_image, 'Lane departure are detected.',
                    (int(cap_image_width * 0.25), int(cap_image_height * 0.85)), font, 1, (50, 50, 255), 1)

    # 핸들이미지 조건 만족시 차선의 기울기에 따라 핸들의 회전된 이미지 출력
    if count_lane >= 4:
        Draw_hough_line_image(lane_image, source)
    else:
        # 차선추세를 파악할수 없는 경우 핸들 사진 대신 경고문구 출력
        cv2.putText(source, 'Warning : Cannot find line',
                    (int(cap_image_width * 0.25), int(cap_image_height * 0.9)), font, 1, (50, 50, 255), 1)

    cv2.imshow('return_image', return_image)

    return return_image


# 오브젝트 검출과 차선 검출결과를 동시에 출력
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


# birdview로 검출된 차선의 기울기를 토대로 커브 판단, 기울기정도의 판단은 허프라인이 사용됨
# 커브정보를 출력하는 방법은 핸들 이미지의 꺾인 모습을 출력함으로서 인터페이스 구현
def Draw_hough_line_image(image, source):
    global mask_image_inverted
    global font
    global cap_image_width
    global cap_image_height
    image_height = image.shape[0]
    image_width = image.shape[1]
    line_image = np.zeros((image_height, image_width, 3), image.dtype)
    canny_image = cv2.Canny(image, 150, 200)

    # 허프라인 생성후 차선의 직선추세를 그린 직선이미지 출력
    lines = cv2.HoughLines(canny_image, 1, 3.141592 / 180, 30)
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

        # 핸들 이미지를 주행 영상에 합성하기위한 사전작업
        if lines[0][0][0] < -0.07 or lines[len(lines) - 1][0][0] > 0.7:
            if lines[0][0][0] < -0.07:
                M = cv2.getRotationMatrix2D((handle_image_height / 2, handle_image_width / 2),
                                            - (1 / lines[0][0][0] * 400), 1)
            else:
                M = cv2.getRotationMatrix2D((handle_image_height / 2, handle_image_width / 2),
                                            - (1 / lines[len(lines) - 1][0][0] * 400), 1)
        else:
            M = cv2.getRotationMatrix2D((handle_image_height / 2, handle_image_width / 2), 0, 1)

        """dst = cv2.warpAffine(mask_image_inverted, M, (handle_image_height, handle_image_width))
        roi = source[int(cap_image_height * 0.6):int(cap_image_height * 0.6) + handle_image_height,
                    int(cap_image_width * 0.1):int(cap_image_width * 0.1) + handle_image_width]
        bg = cv2.bitwise_and(roi, roi, mask=mask_image)
        dst_a = cv2.bitwise_and(dst, dst, mask=mask_image_inverted)
        dst_a = cv2.cvtColor(dst_a, cv2.COLOR_GRAY2BGR)
        added = dst_a + bg

        # 커브정도의 정보를 받고 꺾인 핸들의 모습을 주행영상에 합성
        source[int(cap_image_height * 0.6):int(cap_image_height * 0.6) + handle_image_height,
            int(cap_image_width * 0.1):int(cap_image_width * 0.1) + handle_image_width] = added"""

        cv2.imshow('line_image', line_image)


def Main():
    global roi_image_offset_y
    global roi_image_offset_x
    global warp_ratio_left
    global warp_ratio_right
    global corners_original
    global corners_warp

    corners_original = np.array([[int(roi_image_width * warp_ratio_left), 0],
                                 [int(roi_image_width * warp_ratio_right), 0],
                                 [0, roi_image_height - 1],
                                 [roi_image_width - 1, roi_image_height - 1]], np.float32)
    corners_warp = np.array([[0, 0], [bird_view_width - 1, 0], [0, bird_view_height - 1],
                             [bird_view_width - 1, bird_view_height - 1]], np.float32)

    while cap.isOpened():
        roi_image_offset_y = int(cap_image_height * cv2.getTrackbarPos('Set_roi_image_offset_y', 'final_result') / 1000)
        roi_image_offset_x = int(cap_image_width * cv2.getTrackbarPos('Set_roi_image_offset_x', 'final_result') / 1000)
        warp_ratio_left = cv2.getTrackbarPos('Set_warp_ratio_left', 'final_result') / 1000
        warp_ratio_right = cv2.getTrackbarPos('Set_warp_ratio_right', 'final_result') / 1000

        corners_original = np.array([[int(roi_image_width * warp_ratio_left), 0],
                                     [int(roi_image_width * warp_ratio_right), 0],
                                     [0, roi_image_height - 1],
                                     [roi_image_width - 1, roi_image_height - 1]], np.float32)
        corners_warp = np.array([[0, 0], [bird_view_width - 1, 0], [0, bird_view_height - 1],
                                 [bird_view_width - 1, bird_view_height - 1]], np.float32)

        ret, img = cap.read()
        ar_image = copy.deepcopy(img)
        equalized_image = Equalize_histogram_in_hsl(img)
        equalized_roi_image = Region_of_interest(equalized_image)
        warp_equalized_roi_image = Warp_image(equalized_roi_image)
        cap_image_gray_histogram = Draw_Histogram(img)
        count_bright = Count_bright(cap_image_gray_histogram)
        lane_image = Filter_lane_by_hsl(warp_equalized_roi_image, count_bright)
        not_blemishes_image = Remove_blemishes(lane_image)
        decided_lane = Decide_lane(not_blemishes_image, ar_image)
        Detect_object(ar_image, net, CONFIDENCE, THRESHOLD)
        final_result = To_ar_image_with_lane(ar_image, decided_lane)

        cv2.imshow('final_result', final_result)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    Main()
