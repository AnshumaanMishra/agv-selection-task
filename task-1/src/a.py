import cv2
import numpy as np

# Read the image
# image = cv2.imread('/home/anshumaan/Development/College/agv-selection-task/task-1/resources/chess.jpg')
# image = cv2.imread('/home/anshumaan/Development/College/agv-selection-task/task-1/resources/chess.webp')
cap = cv2.VideoCapture('/home/anshumaan/Development/College/agv-selection-task/task-1/resources/untitled.mp4')
# image = cv2.imread('/home/anshumaan/Development/College/agv-selection-task/task-1/resources/image.png')S
feature_params = dict(maxCorners = 500, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
k = 0.04

factor = 3
threshold = 20000

yKernel = np.array([[-1, -2, -1], \
                    [0, 0, 0], \
                    [1, 2, 1]])
xKernel = np.array([[-1, 0, 1], \
                    [-2, 0, 2], \
                    [-1, 0, 1]])

ret, prev_frame = cap.read()
prev_frame=cv2.resize(prev_frame, (800, 450))
gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_corners = cv2.goodFeaturesToTrack(gray_frame, mask=None, **feature_params)
def getLines(prev_frame, frame):
    velocities = np.array([[0], [0]])
    corners = prev_corners.copy()

    for i, ele in enumerate(corners):
        A = np.array([[0, 0], [0, 0]])
        b = np.array([[0], [0]])

        y, x = int(ele[0][0]), int(ele[0][1])
        if(x < 1 or y < 1):
            continue
        mat = prev_frame[x - 1 : x + 2, y - 1 : y + 2]
        mat2 = frame[x - 1 : x + 2, y - 1 : y + 2]
        print(x, y, mat, mat2, sep='\n')
        Ix = xKernel.dot(mat)
        Iy = yKernel.dot(mat)
        It = mat2 - mat
        for j in range(0, 9):
            x1 = j // 3
            y1 = j % 3
            Ixi = Ix[x1][y1]
            Iyi = Iy[x1][y1]
            Iti = It[x1][y1]
            A += np.array([[Ixi * Ixi, Ixi * Iyi], [Ixi * Iyi, Iyi * Iyi]])
            b += np.array([[-1 * Ixi * Iti], [-1 * Iyi * Iti]])
        
        velocities = np.linalg.pinv(A).dot(b)
        v2 = velocities.copy()
        # while((v2[0][0] < 1 and v2[0][0] >= -1) and (v2[1][0] < 1 and v2[1][0] >= -1)):
        #     if(v2[1][0] == 0 and v2[0][0] == 0):
        #         break
        #     v2 += velocities
        #     # print(v2)
        print("Velocities = ", v2)
        corners[i][0][0] -= v2[1][0]
        corners[i][0][1] -= v2[0][0]
    return corners


def plotCorners(image, corners, color):
    for i in corners:
        cv2.circle(image, (int(i[0][0]), int(i[0][1])), 3, color, -1)

while 1:
    ret, frame = cap.read()
    frame=cv2.resize(frame, (800, 450))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    m, n = gray_frame.shape
    print(m, n)
    corners = getLines(gray_prev_frame, gray_frame)
    print(corners - prev_corners)
    plotCorners(frame, corners, (255, 255, 0))
    # plotCorners(frame, prev_corners, (0, 255, 255))
    # output = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow("Cornered Image", frame)
    prev_frame = frame.copy()
    prev_corners = corners.copy()
    # cv2.waitKey(0)
    if (cv2.waitKey(10) & 0xFF == ord('q')):
        break


cap.release()
cv2.destroyAllWindows()