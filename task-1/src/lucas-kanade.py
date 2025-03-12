import cv2
import numpy as np

# Read the image
# image = cv2.imread('/home/anshumaan/Development/College/agv-selection-task/task-1/resources/chess.jpg')
# image = cv2.imread('/home/anshumaan/Development/College/agv-selection-task/task-1/resources/chess.webp')
cap = cv2.VideoCapture('/home/anshumaan/Development/College/agv-selection-task/task-1/resources/task-1-clipped.mp4')
# image = cv2.imread('/home/anshumaan/Development/College/agv-selection-task/task-1/resources/image.png')S
feature_params = dict(maxCorners = 200, qualityLevel = 0.2, minDistance = 1, blockSize = 7)
k = 0.04

def lucas_kanade_optical_flow(prev_gray, curr_gray, window_size=5):
    # Compute image gradients
    Ix = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=3)
    It = curr_gray - prev_gray  # Temporal gradient

    # Precompute squared terms and products
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    Ixt = Ix * It
    Iyt = Iy * It

    # Sum over the window using boxFilter (efficient moving average)
    sum_Ixx = cv2.boxFilter(Ixx, -1, (window_size, window_size))
    sum_Iyy = cv2.boxFilter(Iyy, -1, (window_size, window_size))
    sum_Ixy = cv2.boxFilter(Ixy, -1, (window_size, window_size))
    sum_Ixt = cv2.boxFilter(Ixt, -1, (window_size, window_size))
    sum_Iyt = cv2.boxFilter(Iyt, -1, (window_size, window_size))

    # Compute determinant and inverse of A matrix
    det_A = sum_Ixx * sum_Iyy - sum_Ixy ** 2
    inv_A11 = sum_Iyy / det_A
    inv_A22 = sum_Ixx / det_A
    inv_A12 = -sum_Ixy / det_A

    # Compute optical flow components Vx, Vy
    Vx = inv_A11 * sum_Ixt + inv_A12 * sum_Iyt
    Vy = inv_A12 * sum_Ixt + inv_A22 * sum_Iyt

    return Vx, Vy

def compute_optical_flow(prev_gray, curr_gray, window_size=3):
    # Compute image gradients
    Ix = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=3)
    It = curr_gray - prev_gray  # Temporal gradient

    half_w = window_size // 2
    flow = np.zeros_like(prev_gray, dtype=np.float32)  # Store velocities

    for y in range(half_w, prev_gray.shape[0] - half_w):
        for x in range(half_w, prev_gray.shape[1] - half_w):
            print(x, y)
            # Get window
            Ix_window = Ix[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
            Iy_window = Iy[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
            It_window = It[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()

            A = np.vstack((Ix_window, Iy_window)).T
            b = -It_window

            # Solve for V = (Vx, Vy)
            if np.linalg.cond(A.T @ A) < 1e-2:  # Avoid singular matrix
                continue
            V = np.linalg.pinv(A.T @ A) @ (A.T @ b)

            flow[y, x] = np.linalg.norm(V)  # Store optical flow magnitude

    return flow

factor = 3
threshold = 20000

laplacian1 = np.array([[0, 1, 0], \
                        [1, -4, 1], \
                        [0, 1, 0]])
laplacian2 = np.array([ [1, 1, 1], \
                        [1, -8, 1], \
                        [1, 1, 1]])


speed = 0.01
# yKernel = np.array([[-3, -10, -3], \
#                     [0, 0, 0], \
#                     [3, 10, 3]], dtype=np.float32)
# xKernel = np.array([[-3, 0, 3], \
#                     [-10, 0, 10], \
#                     [-3, 0, 3]], dtype=np.float32)
speed = 0.07
yKernel = np.array([[-1, -2, -1], \
                    [0, 0, 0], \
                    [1, 2, 1]])
xKernel = np.array([[-1, 0, 1], \
                    [-2, 0, 2], \
                    [-1, 0, 1]])

# Define the input matrix
mat = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]], dtype=np.float32)

# Define Sobel X and Y kernels
xKernel = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

yKernel = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]], dtype=np.float32)

# Apply filter2D to compute Ix and Iy
Ix = cv2.filter2D(mat, -1, xKernel)
Iy = cv2.filter2D(mat, -1, yKernel)

print("Ix:\n", Ix)
print("Iy:\n", Iy)

def normalize_zscore(value_denorm, mean, std):
    if(std == 0):
        return value_denorm
    return (value_denorm - mean) / std
    
def denormalize_zscore(value_norm, mean, std):
    if(std == 0):
        return value_norm
    return value_norm * std + mean

ret, prev_frame = cap.read()
# prev_frame = cv2.GaussianBlur(prev_frame, (3, 3), 0)
gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_corners = cv2.goodFeaturesToTrack(gray_frame, mask=None, **feature_params)
def getLines(prev_frame, frame):
    velocities = np.array([[0], [0]])
    corners = prev_corners.copy()
    # print((corners))
    for i, ele in enumerate(corners):
        A = np.array([[0, 0], [0, 0]])
        b = np.array([[0], [0]])

        y, x = int(ele[0][0]), int(ele[0][1])
        mat = prev_frame[x - 1 : x + 2, y - 1 : y + 2]
        if(len(mat) != 3 or len(mat[0]) != 3):
            continue
        mat2 = frame[x - 1 : x + 2, y - 1 : y + 2]
        # mat = mat.astype(np.float32)
        # mat2 = mat2.astype(np.float32)
        # print(x, y, mat, mat2, sep='\n')
        # Ixa = np.matmul(xKernel, mat)
        # Iya = np.matmul(yKernel, mat)
        print("Sobel", (cv2.Sobel(mat, cv2.CV_64F, 1, 0, ksize=3, scale=1)), (cv2.Sobel(mat, cv2.CV_64F, 0, 1, ksize=3, scale=1)), sep='\n')
        # mat = laplacian1 * mat
        # mat2 = laplacian1 * mat2
        # mat = laplacian2 * mat
        # mat2 = laplacian2 * mat2
        mat = cv2.Laplacian(mat, cv2.CV_64F, ksize=3)
        mat2 = cv2.Laplacian(mat2, cv2.CV_64F, ksize=3)
        Ix = np.int64(cv2.Sobel(mat, cv2.CV_64F, 1, 0, ksize=3, scale=1))
        Iy = np.int64(cv2.Sobel(mat, cv2.CV_64F, 0, 1, ksize=3, scale=1))
        # print(mat, mat2, sep='\n')
        # Ix = xKernel * mat
        # Iy = yKernel * mat
        It = (mat2 - mat)
        # print(f"Ix : \n{Ix}")
        # Ix = cv2.filter2D(mat, -1, xKernel)
        # Iy = cv2.filter2D(mat, -1, yKernel)
        # print(f"Ix : \n{Ix}")

        xMean, xStd = np.mean(Ix), np.std(Ix)
        yMean, yStd = np.mean(Iy), np.std(Iy)
        tMean, tStd = np.mean(It), np.std(It)
        Ix = normalize_zscore(Ix, xMean, xStd)
        Iy = normalize_zscore(Iy, yMean, yStd)
        It = normalize_zscore(It, tMean, tStd)
        # print(f"Ix : \n{Ix}\n, Iy: \n{Iy}\n, It: \n{It}\n")
        # print(f"Ixa : \n{Ixa}\n, Iya: \n{Iya}\n")
        Ix = np.reshape(Ix, (1,9))
        Iy = np.reshape(Iy, (1,9))
        It = np.reshape(It, (1,9))
        I = np.concatenate((Ix, Iy)).T
        T = np.concatenate((It, It)).T
        A = np.int64(np.matmul(I.T, I))
        b = np.int64(np.matmul(I.T, T))
        b = -1 * (b.T)[0]
        # print(b)
        
        velocities = np.matmul(np.linalg.pinv(A), (b))
        velocities[0] = denormalize_zscore(velocities[0], xMean, xStd)
        velocities[1] = denormalize_zscore(velocities[1], yMean, yStd)
        print(lucas_kanade_optical_flow(prev_frame, frame))
        print("Velocities = ", velocities)
        # print("Corners[i] = ", corners[i])
        if(velocities[0] > corners[i][0][0] or velocities[1] > corners[i][0][1]):
            continue
        corners[i] += velocities * speed
        # print("Corners[i] = ", corners[i])
    return corners


def plotCorners(image, corners, color):
    for i in corners:
        cv2.circle(image, (int(i[0][0]), int(i[0][1])), 3, color, -1)

# for _ in range(50):
#     ret, frame = cap.read()
mask = np.zeros_like(prev_frame)
while 1:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    m, n = gray_frame.shape
    print(m, n)
    corners = getLines(gray_prev_frame, gray_frame)
    # print(np.concatenate((corners, prev_corners), axis=1))
    # plotCorners(frame, corners, (255, 255, 0))
    # plotCorners(frame, prev_corners, (0, 255, 255))
    # mask = plotLines(frame, prev_corners, corners)
    for i in range(len(corners)):
        mask = cv2.line(mask, np.int64(prev_corners[i][0]), np.int64(corners[i][0]), (255, 255, 255), 2)
        frame = cv2.circle(frame, (int(corners[i][0][0]), int(corners[i][0][1])), 5, (255, 255, 255), -1)
    # return mask
    output = cv2.add(frame, mask)
    output = cv2.resize(output, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow("Cornered Image", output)
    prev_frame = frame.copy()
    prev_corners = corners.copy()
    # cv2.waitKey(0)
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        break


cap.release()
cv2.destroyAllWindows()
