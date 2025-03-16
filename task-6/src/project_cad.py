import numpy as np
import cv2
from helper import refineF, displayEpipolarF, epipolarMatchGUI
# import matplotlib.pyplot as plt

im1 = cv2.imread("task-6/resources/data/im1.png")
im2 = cv2.imread("task-6/resources/data/im2.png")

im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

data = np.load("task-6/resources/data/some_corresp.npz")
print(data.files)

pts1 = data["pts1"]
pts2 = data["pts2"]
# print(pts1[:5])
# print(pts2[:5])
M = max(len(im1), len(im1[0]))

def sqEucDist(x, y):
    return np.reshape(sum((x - y).T ** 2), (len(x), 1))

def normalize_pts(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)
    T = np.array([
        [1/std[0], 0, -mean[0]/std[0]],
        [0, 1/std[1], -mean[1]/std[1]],
        [0, 0, 1]
    ])
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T

def getSVD(A):
    U, S, Vt = np.linalg.svd(A)
    return U, S, Vt.T

def rankEnforce(F):
    U, S, Vt = np.linalg.svd(F)  # Compute SVD
    S[-1] = 0  # Set the smallest singular value to zero
    F_enforced = U @ np.diag(S) @ Vt  # Reconstruct F with rank 2
    # print("\n\n\n", U, np.diag(S), Vt, "\n\n\n", sep='\n')
    # print(U @ np.diag(S))
    return F_enforced

def eight_point(pts1, pts2):
    pts1_N, T1 = normalize_pts(pts1)
    pts2_N, T2 = normalize_pts(pts2)
    # print(pts1[:5])
    # print(pts1_N[:5])
    num = len(pts1)
    x2 = np.reshape(pts2_N[:, 0], (num, 1))
    y2 = np.reshape(pts2_N[:, 1], (num, 1))
    x1 = np.reshape(pts1_N[:, 0], (num, 1))
    y1 = np.reshape(pts1_N[:, 1], (num, 1))


    # Contraint Matrix formation
    A = np.hstack((
        x2 * x1, 
        x2 * y1,
        x2,
        y2 * x1,
        y2 * y1,
        y2,
        x1,
        y1,
        np.ones((num, 1))
    ))


    U, S, V = getSVD(A)

    f = V[:, -1]
    F = np.reshape(f, (3, 3))
    # print(f"f = \n\t{f},\n\nF = \n\t{F}")
    refineF(F, pts1, pts2)
    
    return F, T1, T2

def findPoint(im1, im2, x1, y1, a, b, c):
    # print(x1, y1, a, b, c)
    im1_ = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_ = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    mat1 = im1_[y1 - 2: y1 + 3, x1 - 2 : x1 + 3]

    h, w = im2_.shape[:2]
    min_ssd = float('inf')
    bestMatch = np.array([-1, -1])
    for x2 in range(w):
        y2 = (- c - a * x2) / b
        if y2 < 2 or y2 >= h - 2:  
            continue
        y2 = np.int32(np.round(y2, decimals=0))
        # print(x2, y2)
        mat2 = im2_[y2 - 2: y2 + 3, x2 - 2 : x2 + 3]
        if(mat2.shape != mat1.shape):
            continue
        diff = mat2 - mat1
        sqdiff = diff ** 2
        ssd = sum(sum(sqdiff))
        print(f"SSD = {ssd}, \nmat2 = {mat2}, \nmat1 = {mat1}")
        if(ssd < min_ssd):
            min_ssd = ssd
            bestMatch = np.array([x2, y2])
    return bestMatch[0], bestMatch[1]


def epipolar_correspondences(im1, im2, F, pts):

    pts_augmented = np.hstack((pts, np.ones((len(pts), 1))))
    l = F @ pts_augmented.T
    l = l.T
    # a = np.reshape(l[:, 0], (len(l), 1))
    # b = np.reshape(l[:, 1], (len(l), 1))
    # c = np.reshape(l[:, 2], (len(l), 1))
    # x = np.reshape(pts[:, 0], (len(pts), 1))
    # y = np.reshape(pts[:, 1], (len(pts), 1))
    a = l[:, 0]
    b = l[:, 1]
    c = l[:, 2]
    x = pts[:, 0]
    y = pts[:, 1]
    corrs = []
    for i in range(len(pts)):
        x_c, y_c = findPoint(im1, im2, x[i], y[i], a[i], b[i], c[i])
        corrs.append(np.array([x_c, y_c]))
    corrs = np.array(corrs)
    # print(np.shape(x_o))
    # print(np.shape(y_o))
    return corrs
    # a = np.reshape(l[:, 0], ())

F, T1, T2 = eight_point(pts1, pts2)
F_enf = rankEnforce(F)
# print(f"F = \n\t{F},\n\nF_enf = \n\t{F_enf}")
F_deN = T2.T @ F_enf @ T1
pts2_c = epipolar_correspondences(im1, im2, F_deN, pts1)
print(pts2_c)
# print("PTS_C: ")
# print(*pts2_c[:5], sep = "\n")
# print(np.shape(pts2_c))
# print("PTS2: ")
# print(*pts2[:5], sep = "\n")
# print(sqEucDist(pts2, pts2_c)[:5])
# displayEpipolarF(im1, im2, F_deN)
epipolarMatchGUI(im1, im2, F_deN)
# plt.show()
