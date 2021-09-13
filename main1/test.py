import cv2
import numpy as np

def Day(img):
    return img
       
def Night(img):
   
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r_image, g_image, b_image = cv2.split(img)
    r_image_eq = cv2.equalizeHist(r_image) 
    g_image_eq = cv2.equalizeHist(g_image)
    b_image_eq = cv2.equalizeHist(b_image)
    img = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    im_blurred = cv2.GaussianBlur(img, (1,1), 1)
    img = cv2.addWeighted(img, 1 + 3.0, im_blurred, -3, 0) 
    return img

CONFIDENCE = 0.05 #차이냐 아니냐
THRESHOLD = 0.6 #ㅇ
classes = ['','bus','rickshaw','motorbike','car','three wheelers (CNG)','pickup','minivan','suv','van','truck','bicycle','policecar','ambulance','human hauler','wheelbarrow','minibus','auto rickshaw','army vehicle','scooter','garbagevan']
CAR_WIDTH_TRESHOLD = 500

cap = cv2.VideoCapture('testvideo.mp4')

net = cv2.dnn.readNetFromDarknet("yolov4-tiny_custom.cfg","yolov4-tiny_custom_final.weights")

while cap.isOpened():

    ret, img = cap.read()
    if not ret:
        break

    hight, W, _ = img.shape

    #blob = cv2.dnn.blobFromImage(img, scalefactor=1/255., size=(416, 416), swapRB=True)
    blob = cv2.dnn.blobFromImage(img, 1/255,(448,448),(0,0,0),swapRB = False,crop= False)
    net.setInput(blob)
    output = net.forward()

    boxes, confidences, class_ids = [], [], []

    for det in output:
        box = det[:4]
        scores = det[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONFIDENCE:
            cx, cy, w, h = box * np.array([W, hight, W, hight])
            x = cx - (w / 2)
            y = cy - (h / 2)

            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    font = cv2.FONT_HERSHEY_PLAIN
    if len(idxs) > 0:
        for i in idxs.flatten():
                label = str(classes[class_ids[i]])
                x,y,w,h = boxes[i]
                veh=y+h/2
                
                lineveh=hight-96
    

                confidence = str(round(confidences[i],2))
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
                cv2.putText(img,label + " " + confidence, (x,y+20),font,1,(0,0,255),1) 
    
    

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break
