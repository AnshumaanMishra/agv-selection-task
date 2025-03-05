# corner detection

# Import necessary library
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the data
img = cv2.imread('./resources/chess.jpg')
cv2.imshow("Image", img)
if cv2.waitKey(0) & 0xFF == ord('q'):
# The following frees up resources and closes all windows
    cv2.destroyAllWindows()
# convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# We are detecting only 100 best corners here, if you want more accuracy change the values

corners = cv2.goodFeaturesToTrack(gray_img, 100000, 0.01, 10)

# convert corners values to integer
# So that we will be able to draw circles on them
corners = np.intp(corners)

# draw red color circles on all corners
for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

# resulting image
# cv2.imwrite('img.png',img)
cv2.imshow("Image", img)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()