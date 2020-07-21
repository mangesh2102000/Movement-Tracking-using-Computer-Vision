import cv2
import numpy as np

# Insert video file name here
cap = cv2.VideoCapture('S13.mp4')

window_name = 'Window'

fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if cap.isOpened() == False:
    print('WRONG CODEC USED OR ERROR FILE NOT FOUND!')
    
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    
    ret,frame = cap.read()
    if ret == True:     
        
        # Edge Boundaries
        up_y = 0
        low_y = 540
        left_x = 0
        right_x = 720
        left=0
        right=720
        up=0
        low=540
                
        frame = cv2.resize(frame,(720,540))
                        
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        # Range for pitch color
        lower_green = np.array([40,40, 40])
        upper_green = np.array([70, 255, 255])
        mask1 = cv2.inRange(hsv, lower_green, upper_green)
        edges = cv2.Canny(mask1,50,150,apertureSize = 3)
        
        # Detect the pitch using Hough Line Transform
        lines = cv2.HoughLines(edges,1,np.pi/180,100)       
        
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 100*(-b))
            y1 = int(y0 + 100*(a))
            x2 = int(x0 - 100*(-b))
            y2 = int(y0 - 100*(a))                              
                        
            if (x2-x1) != 0 and abs((y2-y1)/(x2-x1)) <= 1 :
                
                nx1=100
                nx2=620
                
                ny1 = int ( y1 + ((y2-y1)/(x2-x1) * (nx1-x1)) )
                ny2 = int ( y2 + ((y2-y1)/(x2-x1) * (nx2-x2)) )          
                
                if ny1 < (270) or ny2 < (270) :                    
                    tmp = min(ny1,ny2)
                    if up_y != 0:
                        up_y = min(up_y,tmp)
                    else:
                        up_y = tmp
                else :                    
                    tmp = max(ny1,ny2)
                    if low_y != 540:
                        low_y = max(low_y,tmp)     
                    else:
                        low_y = tmp

                
                #cv2.line(frame,(nx1,ny1),(nx2,ny2),(0,255,0),2)
              
            if (x2-x1) == 0 or abs((y2-y1)/(x2-x1)) > 1 :
                
                ny1=100
                ny2=440
                
                nx1 = int ( x1 + ((x2-x1)/(y2-y1) * (ny1-y1)) )
                nx2 = int ( x2 + ((x2-x1)/(y2-y1) * (ny2-y2)) )                
                
                if nx1 < (360) or nx2 < (360) :
                    tmp = min(nx1,nx2)
                    if left_x != 0:
                        left_x = min(left_x,tmp)
                    else:
                        left_x = tmp
                else :
                    tmp = max(nx1,nx2)
                    if right_x != 720:
                        right_x = max(right_x,tmp)     
                    else:
                        right_x = tmp  
                    
                #cv2.line(frame,(nx1,ny1),(nx2,ny2),(0,255,0),2)                
        
        #cv2.imshow(window_name,frame) 
        
        rect1 = mask1[up:low,0:left_x]
        
        if left_x != 0 :
            if rect1.mean()/255 < 0.5:
                left = left_x
        
        rect2 = mask1[up:low,right_x:720]
    
        if right_x != 720 :
            if rect2.mean()/255 < 0.5:
                right = right_x
                
        rect3 = mask1[0:up_y,0:720]
        
        if up_y != 0 :
            if rect3.mean()/255 < 0.5:
                up = up_y
        
        rect4 = mask1[low_y:540,0:720]
    
        if low_y != 540 :
            if rect4.mean()/255 < 0.5:
                low = low_y    

        # Detect players using Edge Detection and Contours                 
    
        frame2 = frame[up:low,left:right,0:3]
        hsv = cv2.cvtColor(frame2,cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_green, upper_green) 
        edges = cv2.Canny(mask1,50,150,apertureSize = 3) 
        
        ret, thresh1 = cv2.threshold(mask1,127,255,cv2.THRESH_BINARY_INV)
        
        kernel = np.ones((5,5),np.uint8)        
        dilation = cv2.dilate(thresh1,kernel,iterations = 1)        
        #cv2.imshow(window_name,dilation)           
        erosion = cv2.erode(dilation,kernel,iterations = 1)
        #cv2.imshow(window_name,erosion)
                
        median = cv2.medianBlur(mask1,5)
        #cv2.imshow(window_name,median)
        
        _, contours, hierarchy = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        lower_white = np.array([0,0,192])
        upper_white = np.array([0,0,255])
        hsv = cv2.cvtColor(frame2,cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, lower_white, upper_white)
        
        # Color For Team A 
        A = np.uint8([[[196, 166, 135]]]) 
        hsvA = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
        lower_A = np.array([hsvA[0][0][0] - 10, 80, 80])
        upper_A = np.array([hsvA[0][0][0] + 10, 255, 255])
        maskA = cv2.inRange(hsv, lower_A, upper_A)

        # Color For Team B
        B = np.uint8([[[69, 80, 155]]]) 
        hsvB = cv2.cvtColor(B, cv2.COLOR_BGR2HSV)
        lower_B = np.array([hsvB[0][0][0] - 10, 80, 80])
        upper_B = np.array([hsvB[0][0][0] + 10, 255, 255])
        maskB = cv2.inRange(hsv, lower_B, upper_B)
                
       # cv2.imshow(window_name,maskB)

        # Draw rectangles of different colors for both the teams
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            
            if h < (1.2)*w or h > (2.8)*w:
                continue
            
            rect1 = mask1[y:y+h,x:x+w]
            if rect1.mean()/255 > 0.7:
                continue
                
            rect2 = mask2[y:y+h,x:x+w]
            if rect2.mean()/255 > 0.1:
                continue
                
            if cv2.contourArea(contour) < 100 or cv2.contourArea(contour) > 400:
                continue
                
            
            rectA = maskA[y:y+h,x:x+w]
            if rectA.mean()/255 > 0.01:
                cv2.rectangle(frame2, (x, y), (x+w, y+h), (255,0,0), 2)
                continue
            
            rectB = maskB[y:y+h,x:x+w]
            if rectB.mean()/255 > 0.01:
                cv2.rectangle(frame2, (x, y), (x+w, y+h), (0,0,255), 2)
                continue        
                
            cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
        

        cv2.imshow(window_name,frame2)
                      
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    else:
        break
        
cap.release()
cv2.destroyAllWindows()
    
