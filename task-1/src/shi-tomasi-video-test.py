# corner detection

# Import necessary library
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the data
cap = cv2.VideoCapture("./resources/task-1-clipped.mp4")
ret, first_frame = cap.read()

# img = cv2.imread('./resources/chess.jpg')
# cv2.imshow("Image", first_frame)
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
# convert image to grayscale
prevcorn = []
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_img, 100000, 0.01, 10)
    corners = np.intp(corners)
    if(len(prevcorn) == 0):
        prevcorn = corners
    for i in corners:
        x, y = i.ravel()
        x, y = int(x), int(y)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
    for i in range(len(prevcorn)):
        a, b = prevcorn[i].ravel()
        a, b = int(a), int(b)
        cv2.circle(frame, (a, b), 2, (255, 0, 0), -1)
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    prevcorn = corners
    cv2.imshow("Image", frame)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()