import cv2
import os

groundtruth_box = [[104,262,40,368],[162,370,55,244],[49,348,34,365],
                   [172,346,33,381],[204,307,48,264],[56,352,29,327],[140,270,67,285],
                   [86,198,87,253],[200,378,75,252],[54,311,93,324],
                   [107,279,54,376],[10,278,91,341],[53,322,44,403],
                   [97,351,43,362],[68,325,96,387],[118,301,58,316],
                   [70,332,30,319],[106,274,46,298],[127,281,41,247],
                   [148,347,68,317],[97,322,23,310],[1,296,17,249],[108,326,50,350],
                   [157,302,55,307],[154,257,32,276],[81,224,94,264],[140,273,39,300],
                   [95,251,103,347],[31,371,33,416],[21,370,12,385],
                   [113,351,26,346],[64,363,28,288],[21,286,13,276],
                   [119,318,28,264],[66,373,16,380],[2,241,23,211],
                   [131,340,18,317],[101,308,19,246],[63,416,15,301],
                   [48,323,26,366],[33,319,65,416],[112,266,71,365],
                   [29,309,13,263],[56,366,23,369],[67,364,19,386],
                   [56,360,20,383],[100,307,35,355],[54,271,58,291],
                   [88,294,59,416],[8,315,22,416],[9,377,23,416],
                   [23,366,44,411],[13,388,47,408],[38,338,37,416],
                   [47,339,23,384],[79,328,39,397],[42,330,37,416],
                   [26,361,34,376],[48,337,48,401],[42,370,55,390],
                   [24,394,16,415],[23,378,51,400],[73,355,42,397],
                   [50,365,55,408],[35,342,36,426],[11,367,39,403],
                   [40,398,48,403],[42,334,39,402],[28,352,44,412],
                   [47,345,6,375],[49,353,29,407],[12,358,19,410],
                   [47,386,17,417],[54,340,56,406],[30,345,38,399],
                   [27,335,65,400],[61,384,55,413],[49,357,45,406],
                   [68,342,48,403],[51,321,78,396],[43,333,38,416],
                   [19,335,38,416],[44,331,69,395],[43,391,62,402],
                   [51,364,30,387],[14,362,32,416],[34,351,39,404],
                   [37,341,34,391],[26,393,48,387],[29,388,34,408],
                   [44,328,31,389],[38,365,17,387],[37,364,37,400],
                   [61,361,29,388],[37,346,35,410],[30,346,51,393],
                   [40,368,28,381],[52,389,30,416],[26,340,39,416],
                   [9,416,28,417]]


CASCADE="haarcascade_frontalface_default.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

def compute_iou(g_xmin,g_xmax,g_ymin,g_ymax, d_xmin,d_xmax,d_ymin,d_ymax):
   
    
    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa) * max(0, yb - ya)

    boxAArea = (g_xmax - g_xmin) * (g_ymax - g_ymin)
    boxBArea = (d_xmax - d_xmin) * (d_ymax - d_ymin)

    return intersection / float(boxAArea + boxBArea - intersection)




iOUArr = []
DETECTION_THRESHOLD = 0.5
import os
TP = 0
FP = 0
FN = 0
i = 0
TN = 0

# Raccogli i dati iterando la cartella test
for filename in sorted(os.listdir('./test/')):
    #print(filename)
    path = './test/' + filename
    

    image=cv2.imread(path)
    image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.3,minNeighbors=5,minSize=(25,25),flags=0)
    if len(faces) == 0:
        FN = FN + 1
        
    #print(len(faces))
    for x,y,w,h in faces:
        iOU = compute_iou(groundtruth_box[i][0],groundtruth_box[i][1],groundtruth_box[i][2],groundtruth_box[i][3],x,x+w,y,y+h)
        iOUArr.append(iOU)
        if iOU < DETECTION_THRESHOLD:
            FP = FP + 1
        elif iOU > DETECTION_THRESHOLD:
            TP = TP +1
        i = i+1
        
print(TP,TN,FN,FP)
specificity = TN/(TN + FP)
sensitivity = TP/(TP + FN)
precision = TP/(TP + FP)
accuracy = (TP + TN)/(TP+FP+TN+FN)

print('La specificità è: ', specificity)
print('La precisione è: ', precision)
print('La sensitività è: ' , sensitivity)
print('La accuratezza è: ' , accuracy)
