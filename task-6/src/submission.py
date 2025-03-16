"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import cv2

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def rankEnforce(F):
    U, S, Vt = np.linalg.svd(F)  # Compute SVD
    S[-1] = 0  # Set the smallest singular value to zero
    F_enforced = U @ np.diag(S) @ Vt  # Reconstruct F with rank 2
    # print("\n\n\n", U, np.diag(S), Vt, "\n\n\n", sep='\n')
    # print(U @ np.diag(S))
    return F_enforced
def getSVD(A):
    U, S, Vt = np.linalg.svd(A)
    return U, S, Vt.T

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

def eight_point(pts1, pts2, M):
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
    
    return F, T1, T2


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def findPoint(im1, im2, x1, y1, a, b, c):
    windowSize = 4
    halfwidth = windowSize // 2
    # print(x1, y1, a, b, c)
    # im1_ = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # im2_ = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im2_ = im2
    im1_ = im1
    mat1 = im1_[y1 - halfwidth: y1 + halfwidth + 1, x1 - halfwidth : x1 + halfwidth + 1]

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
        diff = np.int64(mat2 - mat1)
        sqdiff = diff ** 2
        ssd = sum(sum(sum(sqdiff)))
        # print(f"SSD = {ssd}, \nmat2 = {mat2}, \nmat1 = {mat1}")
        if(ssd < min_ssd):
            min_ssd = ssd
            bestMatch = np.array([x2, y2])
    return bestMatch[0], bestMatch[1]

def epipolar_correspondences(im1, im2, F, pts):
    print(pts.shape)
    x_co = np.reshape(pts[:, 0], (len(pts), 1))
    y_co = np.reshape(pts[:, 1], (len(pts), 1))
    pts_augmented = np.hstack((x_co, y_co, np.ones((len(pts), 1))))
    l = F @ pts_augmented.T
    l = l.T
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


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    return K2.T @ F @ K1

def getRotationMatrices(K1, K2, F):
    E = essential_matrix(F, K1, K2)
    U, D, Vt = np.linalg.svd(E)
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = np.zeros((3, 0))
    t2 = U[:, 2]
    return R1, R2, t1, t2

def getP1(K1, R1, t1):
    temM1 = np.hstack((R1, t1))
    P1 = K1 @ temM1
    return P1

"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    pts1_a = np.hstack((pts1, np.ones((len(pts1), 1))))
    pts2_a = np.hstack((pts2, np.ones((len(pts2), 1))))
    pts1_a = pts1_a.T
    pts2_a = pts2_a.T
    P11 = P1[0, :]
    P12 = P1[1, :]
    P13 = P1[2, :]
    P21 = P2[0, :]
    P22 = P2[1, :]
    P23 = P2[2, :]

    crossProdMat1 = np.array([
        [0, P13.T, -P12.T],
        [-P13.T, 0, P11.T],
    ])

    crossProdMat2 = np.array([
        [0, P23.T, -P22.T],
        [-P23.T, 0, -P21.T],
    ])

    factor1 = crossProdMat1 @ pts1.T
    factor2 = crossProdMat2 @ pts2.T

    factor = np.vstack((factor1, factor2))
    U, D, Vt = np.linalg.svd(factor)

    print(np.shape(Vt))
"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    pass


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
