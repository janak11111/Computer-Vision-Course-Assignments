# import required libraries
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import pyntcloud


# Question 2.1

# Get the list of calibration images
calibration_images = glob.glob('./Janak_images/*.png')
print(len(calibration_images))



# get list of all lidar points file
lidar_scans_files = sorted(glob.glob('./lidar_scans/*.pcd'))
lidar_image_point = []


# iterate over each file and read the point and store it
for i, file in enumerate(lidar_scans_files):
  point = pyntcloud.PyntCloud.from_file(file).points
  lidar_image_point.append(point.values)



# compute normal and offset for each LIDAR image
lidar_normals = []
lidar_offsets = []
for point_cloud in lidar_image_point:
  mean_point = np.mean(point_cloud, axis=0)
  centerized_point = point_cloud - mean_point
  u, sigma, v_T = np.linalg.svd(centerized_point)
  normal = v_T[-1]
  lidar_normals.append(normal)
  lidar_offsets.append(-np.dot(normal, mean_point))
  
  
# print lidar normals and offsets
print("Lidar Nornals and offsets")
for i in range(len(lidar_normals)):
  print("Normal :", lidar_normals[i], " Offset :", lidar_offsets[i])
print()
  
  
  
# extract the list of files for rotation, translation and normals
translation_vectors_file = sorted(glob.glob("./camera_parameters/frame_*/translation_vectors.*"))
rotation_vectors_file = sorted(glob.glob("./camera_parameters/frame_*/rotation_matrix.*"))
camera_normals_file = sorted(glob.glob("./camera_parameters/frame_*/camera_normals.*"))

translation_vectors2 = []
rotation_vectors2 = []
camera_normals = []


# extract values from files
for i in range(len(camera_normals_file)):
  t = np.loadtxt(translation_vectors_file[i])
  r = np.loadtxt(rotation_vectors_file[i])
  n = np.loadtxt(camera_normals_file[i])

  translation_vectors2.append(t)
  rotation_vectors2.append(r)
  camera_normals.append(n)


 
# compute camera offsets
camera_offsets = []
p = np.array((2, 0, 0))

for n, r, t in zip(camera_normals, rotation_vectors2, translation_vectors2):
  temp = np.dot(r, p)
  temp = np.add(temp, t)
  offset = -np.dot(temp, n)
  camera_offsets.append(offset)
  
  
for i in range(len(camera_normals)):
  print("Normal :", camera_normals[i], " Offset :", camera_offsets[i])
print() 
  
  

# Question 2.3
# compute value of t and R
c_normals = np.array(camera_normals)
c_offsets = np.array(camera_offsets)
l_normals = np.array(lidar_normals)
l_offsets = np.array(lidar_offsets)

A = np.zeros((3,3))


for i in range(len(l_normals)):
    A += np.outer(l_normals[i], c_normals[i])


u, sigma, v_T = np.linalg.svd(A)
R = np.dot(v_T.T, u.T)


# ensure that determinant of R is postive
if np.linalg.det(R) < 0:
    v_T[2,:] *= -1
    R = np.dot(v_T.T, u.T)


temp1 = np.linalg.inv(np.dot(c_normals.T, c_normals))
temp2 = np.matmul(temp1, c_normals.T)
T = np.dot(temp2, c_offsets - l_offsets) 

L2C = np.eye(4)
L2C[:3, :3] = R
L2C[:3, 3] = T
print("Lidar to camera transformation matrix")
print(L2C)
print()


import numpy as np
from scipy.optimize import minimize


# Define objective function
def objective(x):
  r = x[:9].reshape(3, 3)
  t = x[9:].T
  temp = 0

  for i in range(len(lidar_normals)):
    for j in range(len(lidar_image_point[i])):
      t1 = camera_normals[i].T 
      t2 = np.dot(r, lidar_image_point[i][j]) + t
      t3 = np.power((np.dot(t1, t2) - camera_offsets[i]), 2)
      temp += t3
  return np.sqrt(temp)


# Define equality constraint
def eq_constraint(x):
    return np.linalg.det(x[:9].reshape(3, 3)) - 1



#  Define initial guess
R_init = R
T_init = T
x0 = np.concatenate([R_init.flatten(), T_init.flatten()])
max_iterations = 1000


# Define equality constraint dictionary
eq_cons = {'type': 'eq', 'fun': eq_constraint}


# Call the minimize function with equality constraint
result = minimize(objective, x0, method='SLSQP', constraints=[eq_cons], options={'maxiter': max_iterations})


# Print the optimization results
print(result)
print()


# extract the optmial R and T
optimal_R = result.x[:9].reshape(3, 3)
optimal_T = result.x[9:].reshape(3, 1)

L2C_2 = np.eye(4)
L2C_2[:3, :3] = optimal_R
L2C_2[:3, 3] = optimal_T.T

print("Optimal R :")
print(optimal_R)
print()
print("Optimal T :")
print(optimal_T)
print("Optimal L2C :")
print(L2C_2)
print()
print()


l2c_transformed_points = []
for points in lidar_image_point:
  l2c_transformed_points.append((points @ optimal_R) + optimal_T.T)



# Question 2.4
# load the intrinsic camera parameters
camera_matrix2 = np.loadtxt("./camera_parameters/camera_intrinsic.txt")
distortion_coefficient2 = np.loadtxt("./camera_parameters/distortion.txt")


image2_files = sorted(glob.glob("./camera_images/*.jpeg"))


for i, image_file in enumerate(image2_files):
  img = cv2.imread(image_file)

  # convert lidar points to homogenious coordinates
  p = np.vstack([lidar_image_point[i].T, np.ones((1, lidar_image_point[i].shape[0]))])

  # apply transformation
  p = L2C_2 @ p
  p = p[:-1,:]


  # project transformed points to camera image
  corners_proj, _ = cv2.projectPoints(p.T, np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), camera_matrix2, distortion_coefficient2)
  corners_proj.reshape(-1, 2)


  # draw point
  for proj_corner in corners_proj:
    proj_corner = tuple(map(int, proj_corner[0]))
    cv2.circle(img, proj_corner, 3, (0, 255, 0), -1)

  cv2.imshow("a", cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_AREA))
  cv2.waitKey(200)
cv2.destroyAllWindows()
  

# Question 2.5
from scipy.spatial.distance import cosine
import seaborn as sns


similarity_C_L = []
l = np.array(lidar_normals)
L2C_transformed_normals = np.dot(optimal_R, l.T)

# compute the cosine similarity
for x, y in zip(camera_normals, L2C_transformed_normals.T):
  sim = cosine(x, y)
  similarity_C_L.append(abs(1 - sim))


mean_err = np.mean(similarity_C_L)
std_err = np.std(similarity_C_L)
print("mean of similarity errors : ", mean_err)
print("stadnard deviation of similarity errors : ", std_err)
print()


sns.histplot(similarity_C_L)
plt.xlabel("Errors")
plt.ylabel("Counts")
plt.title("Histogram of errors")
plt.show()


# Question 1.5
for i, image_file in enumerate(image2_files[:5]):
  img = cv2.imread(image_file)
  temp1, _ = cv2.projectPoints((rotation_vectors2[i] @ np.array([1, 0, 0]) + translation_vectors2[i]), np.array([0.0, 0.0, 0.0]),np.array([0.0, 0.0, 0.0]) , camera_matrix2, distortion_coefficient2)
  temp2, _ = cv2.projectPoints(camera_normals[i], np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), camera_matrix2, distortion_coefficient2)

  temp3, _ = cv2.projectPoints((rotation_vectors2[i] @ np.array([1, 0, 0]) + translation_vectors2[i]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]) , camera_matrix2, distortion_coefficient2)
  temp4, _ = cv2.projectPoints(lidar_normals[i] // 50, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), camera_matrix2, distortion_coefficient2)
  cv2.arrowedLine(img, (int(temp1[0][0][0]), int(temp1[0][0][1])), (int(temp2[0][0][0]),int(temp2[0][0][1])), color=(0, 255, 255), thickness=6)
  cv2.arrowedLine(img, (int(temp3[0][0][0]), int(temp3[0][0][1])), (int(temp4[0][0][0]),int(temp4[0][0][1])), color=(0, 255, 0), thickness=6)
  cv2.imshow("a", img)
  cv2.waitKey(500)
cv2.destroyAllWindows()
  

for i in range(5):
  points = lidar_image_point[i]
  fig = plt.figure(figsize=(5, 5))
  ax = fig.add_subplot(projection='3d')
  x, y, z = [], [], []
  for d1, d2, d3 in points:
    x.append(d1)
    y.append(d2)
    z.append(d3)
  ax.scatter(x, y, z)

  # plot the lidar normals
  x, y, z = lidar_image_point[i][10]
  x1 = x + 1 * lidar_normals[i][0]
  y1 = y + 1 * lidar_normals[i][1]
  z1 = z + 1 * lidar_normals[i][2]

  ax.quiver(x, y, z, x1, y1, z1, color='red', length=0.3)
  plt.title("point of Lidar Image {}".format(i+1))
  ax.set_xlabel('X axis')
  ax.set_ylabel("Y axis")
  ax.set_zlabel("Z label")
  plt.show()