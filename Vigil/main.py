#####################################################################
#Done by: Group NDSM 
#Members: Nicholas Yap, Chang Dao Zheng, Lim Sui Kiat, Chan Ming Han 
#Date: 17 Oct 2021 
#####################################################################

#Imports
import cv2
import time
import imutils
import numpy as np


#Inputs 
filename = ''#Insert Video Path 

#Controls Variables  
distLimit = 550
timeLimit= 2.0


#Functions 
def main(filename):
    global red_box_frame_counter, fps

    create = None
    frameno = 0
    filename = filename
    output_path = "Processed_" + filename

    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)

    red_box_frame_counter = 0
    time1 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_img = frame.copy()
        current_img = imutils.resize(current_img, width=1080)
        
        frameno += 1
        if(frameno%2 == 0 or frameno == 1):
            Setup()
            analyse(current_img,interactionLimit=1)
            Frame = processedImg
            if create is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                create = cv2.VideoWriter(output_path, fourcc, fps, (Frame.shape[1], Frame.shape[0]), True)
        create.write(Frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    time2 = time.time()
    print("Completed. Total Time Taken: {} minutes".format((time2-time1)/60))

    cap.release()
    cv2.destroyAllWindows()


def Setup():
    global net, layerNames, labels
    weights = "yolov3.weights"
    config = "yolov3.cfg"
    labelsPath = 'coco.names'
    with open(labelsPath,'r') as f: 
        labels = [label.strip() for label in f.readlines()]
    net = cv2.dnn.readNet(weights,config)
    layerNames = net.getUnconnectedOutLayersNames()  

#Gives warning if red box persists for longer than specified time limit
def red_box_checker(red_box):
    global red_box_frame_counter
    warning = False
    if red_box == True: 
        red_box_frame_counter += 1
    else: 
        red_box_frame_counter=0
    if red_box_frame_counter*2/fps >= timeLimit:
        warning = True
        print("WARNING")
    return warning
    
       
    
        
#Frame analysis to identify "person" class objects 
def analyse(image, interactionLimit):
    global processedImg
    (H, W) = (None, None)
    frame = image.copy()
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    

    #Request Network Response to Frame 
    net.setInput(blob)
    starttime = time.time()
    layerOutputs = net.forward(layerNames) #Network scores for the layers > Layer > outputs from network(4x bounding box param, 1x bounding box confidence score, 80x class confidence scores)
    stoptime = time.time()
    print("{:.4f} seconds per frame".format((stoptime-starttime))) 
    
    
    confidences = []
    outline = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            max_class = np.argmax(scores)
            confidence = scores[max_class] #class confidence of class with highest score 
            if labels[max_class] == "person" and confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    outline.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))

    box_line = cv2.dnn.NMSBoxes(outline, confidences, 0.5, 0.3)
    red_box = False
    if len(box_line) > 0:
        flat_box = box_line.flatten()
        pairs = []
        center = []
        status = [] 
        

        #create boxes for all 'person' objects
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(0)

        #Check for distance between 'person' objects
        for i in range(len(center)):
            for j in range(len(center)):
                close = Check(center[i], center[j], distLimit)
                if close:
                    pairs.append([center[i], center[j]])
                    status[i] +=1
       
        #Colour Coding 'person' objects based on number of 'close' interactions with others 
        index = 0
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            if status[index] < interactionLimit: 
                cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 150, 0), 2)
            elif status[index] == interactionLimit:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                red_box = True 
            index += 1
        for h in pairs:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (255 ,0, 255), 2)
    
    #Warning
    warning = red_box_checker(red_box)
    if warning: 
            cv2.putText(frame,"Warning", org=(10,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1 , color=(0, 0, 255), lineType=2, thickness=3)
    processedImg = frame.copy()


def Check(a, b, c):
    if not a==b:
        dist = ((a[0] - b[0]) ** 2 + c / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
        calibration = (a[1] + b[1]) / 2
        if 0 < dist < 0.25 * calibration:
            return True
        else:
            return False
    return False


main(filename)
