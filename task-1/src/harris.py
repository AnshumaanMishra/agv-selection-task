import cv2
import numpy as np

# Read the image
image = cv2.imread('/home/anshumaan/Development/College/agv-selection-task/task-1/resources/chess.webp')

k = 0.04

factor = 3
threshold = 2000000000

xKernel = np.array([[-1, 0, 1]])
yKernel = np.array([[-1], [0], [1]])

if image is None:
    print("Error: Could not load image.")
else:
    print("Image loaded successfully.")
    
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print(*gray_image)
def allPossibleMatrices(image, row, column):
    matrices = []
    for i in range(row - 1, row + factor - 1):
        for j in range(column - 1, column + factor - 1):
            # print(i, row, j, column)
            # print(image[i : i + factor][j : j + factor])
            matrices.append(image[i : i + factor, j : j + factor])
    return matrices

def getHarris(Ix, Iy):
    a11 = sum(sum(Ix.dot(Ix)).T)
    a12 = sum(sum(Ix.dot(Iy)).T)
    a21 = a12
    a22 = sum(sum(Iy.dot(Iy)).T)
    harrisMatrix = np.array([[a11, a12], [a21, a22]])
    return harrisMatrix


def getC(harrisMatrix):
    determinant = np.linalg.det(harrisMatrix)
    trace = np.linalg.trace(harrisMatrix)
    return determinant - k * (trace ** 2)

def getIxandIy(image, row, column):
    mainMat = image[row : row + factor, column : column + factor]
    Ix = np.array([[0 for _ in range(factor)] for _ in range(factor)])
    Iy = np.array([[0 for _ in range(factor)] for _ in range(factor)])
    for i in range(factor):
        for j in range(factor):
            currentX = image[row + i : row + i + 1, column + j - 1 : column + j + 2]
            currentY = image[row + i - 1 : row + i + 2, column + j : column + j + 1]
            # print(f"currentX: {currentX, xKernel}, \n currentY: {currentY, yKernel}")
            dx = sum((xKernel*currentX).T)
            dy = sum(yKernel*currentY)
            # print(dx, dy)
            Ix[i][j] = dx
            Iy[i][j] = dy
    return (Ix, Iy)
            
def HarrisCorners(image, row, column):
    Ix, Iy = getIxandIy(image, row, column)
    harris = getHarris(Ix, Iy)
    C = getC(harris)
    return C > threshold


m, n = gray_image.shape
print(m, n)

corners = []

for i in range(factor - 1, m - (factor + factor - 2)):
    for j in range(factor - 1, n - (factor + factor - 2)):
        print(f"current-mid : {(i, j)}")
        if(HarrisCorners(gray_image, i, j)):
            corners.append((i, j))

corners = np.array(corners)


# corners = cv2.goodFeaturesToTrack(gray_image, 128, 0.01, 10)
# print(*corners, sep="\n")
for i in corners:
    cv2.circle(image, (int(i[1]), int(i[0])), 1, (255, 255, 0), -1)
output = cv2.resize(image, (0, 0), fx=2.5, fy=2.5)

cv2.imshow("Cornered Image", output)
cv2.waitKey(0)

cv2.destroyAllWindows()
