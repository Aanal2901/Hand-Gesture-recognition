A better implementation of this code and [explanation](https://github.com/aakashjhawar/hand-gesture-recognition)
```python
import cv2
import numpy as np
from sklearn.metrics import pairwise 

rect = [0, 399, 0, 399]

cap = cv2.VideoCapture(0)
bg = None
num_frames = 0
while(cap.isOpened()):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    roi = frame[rect[0]:rect[1], rect[2]:rect[3]]
    
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_blur = cv2.GaussianBlur(roi_gray, (7,7), 0)
    
    if num_frames<60:
    #background weighted
        if bg is None:
            bg = roi_blur.copy().astype('float')
        else:
            cv2.accumulateWeighted(roi_blur, bg, 0.5)
        if num_frames <= 59:
            cv2.putText(roi_blur.copy(), "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Finger Count",roi_blur.copy())    
    else:
    #thresholding and segmentation
        diff = cv2.absdiff(bg.astype('uint8'), roi_blur)
        _, thresh = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)
    
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)!=0:
            hand_seg = max(contours, key = cv2.contourArea)
    
            if hand_seg is not None:
        
                conv_hull = cv2.convexHull(hand_seg)
    
                top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
                bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
                left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
                right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
    
                cy = (top[1]+bottom[1])//2
                cx = (right[0]+left[0])//2
    
                dist = pairwise.euclidean_distances([[cx, cy]], Y = [left, right, top, bottom])[0]
    
                max_dist = dist.max()
                radius = int(max_dist*0.7)
                circum = 2*np.pi*radius
    
                circular_roi = np.zeros(thresh.shape[:2], dtype = 'uint8')
                cv2.circle(circular_roi, (cx, cy), radius, 255, 10)
    
                fingers = cv2.bitwise_and(thresh, thresh, mask = circular_roi)
                contours, _ = cv2.findContours(fingers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                count = 0
    
                for cnt in contours:
                    (x,y,w,h) = cv2.boundingRect(cnt)
        
                    upwards = (cy + 0.25*cy>(y + h))
                    inside = (circum*0.25>cnt.shape[0])
        
                    if upwards and inside:
                        count += 1
                print(count)
    cv2.rectangle(frame, (rect[0], rect[2]), (rect[1], rect[3]), (255, 0, 0), 5)
    cv2.imshow('frame', frame)    
    num_frames +=1    
    k = cv2.waitKey(1)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()```
