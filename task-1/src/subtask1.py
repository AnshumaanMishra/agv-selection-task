import cv2 as cv
import numpy as np

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=2, blockSize=5)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("/home/anshumaan/Development/College/agv-selection-task/task-1/resources/task-1-clipped.mp4")
# Variable for color to draw optical flow track
color = (0, 250, 0)

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
mask = np.zeros_like(first_frame)

def LuasKanade(prev_gray, next_gray, prev_points, window_size=6, cutoff=1e-6):

    Ix = cv.Sobel(prev_gray, cv.CV_64F, 1, 0, ksize=5)  
    Iy = cv.Sobel(prev_gray, cv.CV_64F, 0, 1, ksize=5)  

    It = next_gray - prev_gray  #Its just gives the gradient by subtracting
    
    u = np.zeros_like(prev_gray)
    v = np.zeros_like(prev_gray)
    
    half_window = window_size // 2
    
    for point in prev_points:
        x, y = point.ravel()
        x, y = int(x), int(y)
        
        x_left = max(0, x - half_window)
        x_right = min(prev_gray.shape[1], x + half_window)
        y_down = max(0, y - half_window)
        y_up = min(prev_gray.shape[0], y + half_window)
        
        Ix_window = Ix[y_down:y_up, x_left:x_right]
        
        Iy_window = Iy[y_down:y_up, x_left:x_right]

        It_window = It[y_down:y_up, x_left:x_right]

        Ix_window = Ix_window.ravel()  
        Iy_window = Iy_window.ravel()

        
        product_XY = np.matmul(Ix_window.T, Iy_window)  
        square_x = np.matmul(Ix_window.T, Ix_window)  
        square_Y = np.matmul(Iy_window.T, Iy_window)  

        A = np.array([[square_x, product_XY], 
                      [product_XY, square_Y]])

        It_window = It_window.ravel() 

        product_XT = np.matmul(Ix_window.T, It_window)  

        product_YT = np.matmul(Iy_window.T, It_window)  

        b = np.array([-product_XT, -product_YT])

        if np.linalg.det(A)<cutoff:  
            continue
        
        flow = np.linalg.solve(A,b)
        
        u[y, x] = flow[0]
        v[y, x] = flow[1]
    
    return u, v

while cap.isOpened():
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    if not ret:
        break

    # Converts each frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Manually compute optical flow
    u, v = LuasKanade(prev_gray, gray, prev)
    
    # Flatten the points for easy indexing
    good_old = prev.reshape(-1, 2)
    
    # Compute new positions by applying optical flow (u, v)
    good_new = good_old + np.column_stack((u[good_old[:, 1].astype(int), good_old[:, 0].astype(int)],v[good_old[:, 1].astype(int), good_old[:, 0].astype(int)]))
    
    # Clip new coordinates to stay within the bounds of the image
    good_new[:, 0] = np.clip(good_new[:, 0], 0, frame.shape[1] - 1)  # x-coordinate, clip between 0 and width-1
    good_new[:, 1] = np.clip(good_new[:, 1], 0, frame.shape[0] - 1)  # y-coordinate, clip between 0 and height-1
    
    # Draw the optical flow tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)
        mask = cv.line(mask, (a, b), (c, d), color, 2)
        frame = cv.circle(frame, (a, b), 3, color, -1)

    # Overlay the optical flow tracks on the original frame
    output = cv.add(frame, mask)
    
    # Update previous frame and previous feature points
    prev_gray = gray.copy()
    prev = good_new.reshape(-1, 1, 2)
    # Display the output frame
    cv.imshow("sparse optical flow", output)
    
    # Find new good features to track in each frame
    prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    
    
    # Break out of the loop when 'q' key is pressed
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv.destroyAllWindows()
