import cv2
import numpy as np

# Read the image
image = cv2.imread('/home/anshumaan/Development/College/agv-selection-task/task-1/resources/chess.jpg')

factor = 2
threshold = 250# Check if the image was successfully loaded
if image is None:
    print("Error: Could not load image.")
else:
    print("Image loaded successfully.")
# print(image)
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.circle(image, (400, 20), 5, (0, 255, 0), -1)

# Display the grayscale image
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print(*gray_image)

def gettSSD(mat1, mat2):
    # print(mat1, mat2, sep="\n", end="\n\n")
    # print("MSUM", sum((mat1 - mat2) ** 2))
    return sum(sum((mat1 - mat2) ** 2))

def allPossibleMatrices(image, row, column):
    matrices = []
    for i in range(row - 1, row + factor - 1):
        for j in range(column - 1, column + factor - 1):
            # print(i, row, j, column)
            # print(image[i : i + 3][j : j + 3])
            matrices.append(image[i : i + factor, j : j + factor])
    return matrices

def allSSD(image, row, column):
    matrices = allPossibleMatrices(image, row, column)
    # print("Matrices = ", matrices)
    mid = matrices[len(matrices) // 2]
    ssd = []
    for i in range(0, factor):
        for j in range(0, factor):
            # print(f"gettSSD(matrices[{3 * i + j}], mid)", gettSSD(matrices[3 * i + j], mid))
            ssd.append(gettSSD(matrices[factor * i + j], mid))
    return ssd

def corner(ssd, thresh):
    corners = []
    for i in range(0, len(ssd)):
        # print("SSDi", ssd[i])
        if(ssd[i] > thresh):
            corners.append(np.array([i // factor, i % factor]))
    return corners 

m, n = gray_image.shape
print(m, n)

corners = []

for i in range(factor - 1, m - (factor + factor - 2)):
    for j in range(factor - 1, n - (factor + factor - 2)):
        print(f"current-mid : {(i, j)}")
        ssds = allSSD(gray_image, i, j)
        # print("SSDList", ssds)
        temp_c = corner(ssds, threshold)
        # print(temp_c)
        tempc = []
        for k in range(0, len(temp_c)):
            temp_c[k] += np.array([i, j])
            x1 = temp_c[k][0]
            y1 = temp_c[k][1]
            tempc.append(gray_image[x1][y1])
        corners.extend(temp_c)
    
corners = np.array(corners)


# corners = cv2.goodFeaturesToTrack(gray_image, 128, 0.01, 10)
# print(*corners, sep="\n")
for i in corners:
    cv2.circle(image, (int(i[1]), int(i[0])), 1, (255, 255, 0), -1)
output = cv2.resize(image, (0, 0), fx=2.5, fy=2.5)

cv2.imshow("Cornered Image", output)
cv2.waitKey(0)

cv2.destroyAllWindows()
