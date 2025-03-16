import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import cv2
import matplotlib.pyplot as plt
import submission as sub


# 1. Load the two temple images and the points from data/some_corresp.npz
data = np.load("task-6/resources/data/some_corresp.npz")
print(data.files)

pts1 = data["pts1"]
pts2 = data["pts2"]
# print(pts1[:5])
# print(pts2[:5])
im1 = cv2.imread("task-6/resources/data/im1.png")
im2 = cv2.imread("task-6/resources/data/im2.png")

im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

M = max(len(im1), len(im1[0]))

# 2. Run eight_point to compute F

F, T1, T2 = sub.eight_point(pts1, pts2, M)
# F = hlp.refineF(F, pts1, pts2)
F_enf = sub.rankEnforce(F)
F_deN = T2.T @ F_enf @ T1
# 3. Load points in image 1 from data/temple_coords.npz

pts1_2 = np.load("task-6/resources/data/temple_coords.npz")
print(pts1_2.files)

# 4. Run epipolar_correspondences to get points in image 2

pts2_c = sub.epipolar_correspondences(im1, im2, F_deN, pts1_2['pts1'])

# hlp.epipolarMatchGUI(im1, im2, F_deN)
# hlp.displayEpipolarF(im1, im2, F_deN)
# 5. Compute the camera projection matrix P1

KData = np.load("task-6/resources/data/intrinsics.npz")
K1 = KData['K1']
K2 = KData['K2']
E = sub.essential_matrix(F, K1, K2)

R1, R2, t1, t2 = sub.getRotationMatrices(K1, K2, F)
P1 =sub.getP1(K1, R1, t1)

# 6. Use camera2 to get 4 camera projection matrices P2
print(E)
P2s = hlp.camera2(E)
# print("P1 = ", P1, sep='\n')
print("P2 = ", *P2s, sep='\n\n')
# 7. Run triangulate using the projection matrices
sub.triangulate(P1, pts1, P2s[0], pts2)
# 8. Figure out the correct P2

# 9. Scatter plot the correct 3D points

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
