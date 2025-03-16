import cv2
import numpy as np

cap = cv2.VideoCapture('/home/anshumaan/Development/College/agv-selection-task/task-1/resources/task-1-clipped.mp4')

feature_params = dict(maxCorners = 10, qualityLevel = 0.2, minDistance = 20, blockSize = 7)

speed = 0.002
xKernel = np.array([[-1, 0, 1], \
                    [-2, 0, 2], \
                    [-1, 0, 1]])
yKernel = xKernel.T
dim = len(xKernel)
def min_max_normalization(image, I_min, I_max):
    if I_max - I_min == 0:  # Avoid division by zero
        return np.zeros_like(image)
    return (image - I_min) / (I_max - I_min)

def min_max_denormalization(image_norm, I_min, I_max):
    if(I_max - I_min == 0):
        return image_norm
    return image_norm * (I_max - I_min) + I_min


def normalize_zscore(value_denorm, mean, std):
    if(std == 0):
        return np.zeros_like(value_denorm)
    return (value_denorm - mean) / std
    
def denormalize_zscore(value_norm, mean, std):
    if(std == 0):
        return np.zeros_like(value_norm)
    return value_norm * std + mean

ret, prev_frame = cap.read()
# prev_frame = cv2.GaussianBlur(prev_frame, (3, 3), 0)
gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_corners = cv2.goodFeaturesToTrack(gray_frame, mask=None, 
**feature_params)
print("LENPREV", len(prev_corners))
# input()
history = prev_corners.copy()
def getLines(prev_frame, frame):
    print(len(prev_corners))
    corners = prev_corners.copy()
    for i, ele in enumerate(corners):
        y, x = int(ele[0][0]), int(ele[0][1])
        mat = prev_frame[x - (dim - 1) // 2: x + (dim + 1) // 2, y - (dim - 1) // 2 : y + (dim + 1) // 2]
        if(len(mat) != dim or len(mat[0]) != dim):
            continue
        mat2 = frame[x - (dim - 1) // 2: x + (dim + 1) // 2, y - (dim - 1) // 2 : y + (dim + 1) // 2]
        Ix = np.zeros((3, 3))
        Iy = np.zeros((3, 3))
        It = np.zeros((3, 3))
        tempmat2 = prev_frame[x - 2: x + 3, y - 2 : y + 3]
        # print("SobelX = ", np.int64(cv2.Sobel(tempmat2, cv2.CV_64F, 1, 0, ksize=3, scale=1))[1:-1, 1:-1])
        # print("SobelY = ", np.int64(cv2.Sobel(tempmat2, cv2.CV_64F, 0, 1, ksize=3, scale=1))[1:-1, 1:-1])
        # print("Filer2d = ", cv2.filter2D(tempmat2, -1, xKernel))
        for i in range(-1, 2):
            for j in range(-1, 2):
                tempmat1 = prev_frame[x + j - 1: x + j + 2, y + i - 1 : y + i + 2]
                # print("Tempmat = ", tempmat1)
                # print("Tempmat2 = ", tempmat2)
                # print("Filer2d = ", cv2.filter2D(tempmat1, -1, xKernel))
                # print(f"Product[{j + 1}, {i + 1}] = ", sum(sum(tempmat1 * xKernel)))
                Ix[j + 1][i + 1] = sum(sum(tempmat1 * xKernel))
                Iy[j + 1][i + 1] = sum(sum(tempmat1 * yKernel))
                It[j + 1][i + 1] = mat2[j + 1][i + 1] - mat[j + 1][i + 1]
        # Ix[0][0] = Ix[1][0] = Ix[2][0] = Ix[0][2] = Ix[1][2] = Ix[2][2] = 0
        # Iy[0][0] = Iy[0][1] = Iy[0][2] = Iy[2][0] = Iy[2][1] = Iy[2][2] = 0
        # Ix = xKernel * mat
        # Iy = yKernel * mat
        # It = (mat2 - mat)
        # Ix = np.int64(cv2.Sobel(mat, cv2.CV_64F, 1, 0, ksize=3, scale=1))
        # Iy = np.int64(cv2.Sobel(mat, cv2.CV_64F, 1, 0, ksize=3, scale=1))
        # It = mat2 - mat
        print(f"tempmat2: \n{tempmat2}\n, Ix : \n{Ix}\n, Iy: \n{Iy}\n, It: \n{It}\n, It2: \n{mat2-mat}\n")
        xMean, xStd = np.mean(Ix), np.std(Ix)
        yMean, yStd = np.mean(Iy), np.std(Iy)
        tMean, tStd = np.mean(It), np.std(It)
        Ix = normalize_zscore(Ix, xMean, xStd)
        Iy = normalize_zscore(Iy, yMean, yStd)
        It = normalize_zscore(It, tMean, tStd)

        Ix = np.reshape(Ix, (dim**2, 1))
        Iy = np.reshape(Iy, (dim**2, 1))
        It = np.reshape(It, (dim**2, 1))
        A = np.array([np.array([i[0], j[0]]) for (i, j) in zip(Ix, Iy)])
        b = -It
        print(Ix, Iy, It, A, b, sep="\n\n")
        t1 = np.matmul(A.T, A)
        t2 = np.matmul(A.T, b)
        # 
        velocities = np.matmul(np.linalg.pinv(t1), (t2))
        velocities[0] = denormalize_zscore(velocities[0], xMean, xStd)
        velocities[1] = denormalize_zscore(velocities[1], yMean, yStd)
        # if(velocities[0][0] == velocities[1][0] == 0):
        #     velocities[0][0] = velocities[1][0] = -5
        print("Velocities = ", velocities)
        # print("Corners[i] = ", corners[i])
        # if(velocities[0] > corners[i][0][0] or velocities[1] > corners[i][0][1]):
        #     continue
        corners[i] -= velocities.T * speed
        # print("Corners[i] = ", corners[i])
    return corners


# for _ in range(50):
#     ret, frame = cap.read()
mask = np.zeros_like(prev_frame)
frame_count = 1
while 1:
    frame_count += 1
    # if frame_count % 15 == 0:
    #     prev_corners = cv2.goodFeaturesToTrack(gray_frame, mask=None, **feature_params)
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    m, n = gray_frame.shape
    # print(m, n)
    corners = getLines(gray_prev_frame, gray_frame)

    for i in range(len(corners)):
        mask = cv2.line(mask, np.int64(prev_corners[i][0]), np.int64(corners[i][0]), (0, 255, 255),2)
        # mask = cv2.circle(mask, (int(prev_corners[i][0][0]), int(prev_corners[i][0][1])), 2, (0, 255, 255), -1)
        frame = cv2.circle(frame, (int(corners[i][0][0]), int(corners[i][0][1])), 2, (255, 255, 255), -1)
    # return mask
    output = cv2.add(frame, mask)
    output = cv2.resize(output, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow("Cornered Image", output)
    prev_frame = frame.copy()
    prev_corners = corners.copy()
    # history = prev_corners.copy()
    prev_corners = cv2.goodFeaturesToTrack(gray_frame, mask=None, **feature_params)
    # cv2.waitKey(0)
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        break


cap.release()
cv2.destroyAllWindows()
