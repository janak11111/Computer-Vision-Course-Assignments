# import required libraries
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt



# Question 1.1 

# Define the number of chessboard corners
num_cols = 10
num_rows = 7


# Define the real-world coordinates of the corners
objp = np.zeros((num_rows * num_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:num_rows, 0:num_cols].T.reshape(-1, 2)



# Create arrays to store object points and image points from all images
count = 0
obj_points = []
img_points = []
temp = []
temp1 = []


# Get the list of calibration images
calibration_images = glob.glob('./Janak_images/*.png')
# print(len(calibration_images))


# Loop through each image
for i, fname in enumerate(calibration_images):
  img = cv2.imread(fname)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


  # Find the chessboard corners
  ret, corners = cv2.findChessboardCorners(gray, (num_rows, num_cols), None)


  # If found, add object points and image points
  if ret == True:
      count += 1
      obj_points.append(objp)
      img_points.append(corners)
      temp1.append(i+1)
      
      # Draw and display the corners
      cv2.drawChessboardCorners(img, (num_rows, num_cols), corners, ret)
      cv2.imshow("a", img)
      cv2.waitKey(500)
  else:
    temp.append(fname.split("/")[-1])

cv2.destroyAllWindows()



# Calibrate the camera
print("gray :", gray.shape)
ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

rotation_matrices =  []
for x in rotation_vectors:
  y, _= cv2.Rodrigues(x)
  rotation_matrices.append(y)

# Print the camera matrix and distortion coefficients
print("Camera matrix:\n", camera_matrix)
print()
print("Distortion coefficients: ", distortion_coefficients.ravel())


# Question 1.2
for i, m in enumerate(rotation_matrices[:25]):
  print("rotation_matrix : {}".format(i+1))
  print(m)
  print()
  

for i, t in enumerate(translation_vectors[:25]):
	print("translation vectors : {}".format(i+1))
	print(t)
	print()


# Question 1.3
for img in calibration_images[20:]:
  img = cv2.imread(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  h, w = img.shape[:2]
  new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))
  undistorted_img = cv2.undistort(img, camera_matrix, distortion_coefficients, None, new_camera_matrix)
  cv2.imshow("b", np.hstack((undistorted_img, img)))
  cv2.waitKey(500)
cv2.destroyAllWindows()
  
  

# Question 1.4
error = []

for i in range(25):
  # extract image point, rotation and translation vector corresponding to that
  img_point = img_points[i]
  r = rotation_vectors[i]
  t = translation_vectors[i]

  # find projected point
  proj_point, _ = cv2.projectPoints(obj_points[i], r, t, camera_matrix, None)
  
  # compute reprojection error and append it
  err = cv2.norm(img_point, proj_point, cv2.NORM_L2) / len(proj_point)
  error.append(err)


mean = np.mean(error)
std = np.std(error)
print("mean of errors : ", mean)
print("stadard deviation of errors :", std)
print()

plt.bar(range(1, 26), error)
plt.xlabel("Image Index")
plt.ylabel("Re-Projection Error")
plt.title("Re-Projection Error For Each Images")
plt.show()





# Question 1.5
for i, image_path in enumerate(calibration_images):
  # Load the image
  img = cv2.imread(image_path)

  # Project the object points onto the image using the estimated camera parameters
  R = rotation_vectors[i]
  t = translation_vectors[i]
  corners_proj, _ = cv2.projectPoints(objp, R, t, camera_matrix, distortion_coefficients)
  corners_proj = corners_proj.reshape(-1, 2)

  # Draw the re-projected corners on the image
  for proj_corner, actual_corner in zip(corners_proj, img_points[i]):
    proj_corner = tuple(map(int, proj_corner))
    actual_corner = actual_corner.reshape(2)
    actual_corner = tuple(map(int, actual_corner))
    cv2.circle(img, actual_corner, 10, (0, 0, 255), 1)
    cv2.circle(img, proj_corner, 10, (0, 255, 0), 1)

  # Display the image with the detected and re-projected corners
  cv2.imshow("c", img)
  cv2.waitKey(500)
cv2.destroyAllWindows()



# checkerboard plane normals in camera coardinate frame of reference 
camera_normals_q1 = []


for i in range(len(calibration_images)):
    # Get rotation matrix and translation vector for current image
    r1,_ = cv2.Rodrigues(rotation_vectors[i])
    
    # Calculate checkerboard plane normal
    normal = np.dot(r1, np.array([0, 0, -1]))
    camera_normals_q1.append(normal)

print("camera normals")
for n in camera_normals_q1:
  print(n)