from laspy.file import File
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import numpy as np
import time
import math


#for how many point show 1 point in visualization
PPSP = 1

inFile = File('Build01 - Wall3.las',mode = 'r')
degree= -10

i = 0
x=inFile.x
y=inFile.y
z=inFile.z
points = np.vstack((x,y,z))

#print(points.shape) # (3,n)
#conputing the d- dimensional mean vector

mean_x = np.mean(points[0,:])
mean_y = np.mean(points[1,:])
mean_z = np.mean(points[2,:])

mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

#print('Mean Vector:\n', mean_vector)



#this is too slow
#computing the Scatter Matrix
'''
start = time.time()
scatter_matrix = np.zeros((3,3))
for i in range(points.shape[1]):
    scatter_matrix += (points[:,i].reshape(3,1) - mean_vector).dot((points[:,i].reshape(3,1) - mean_vector).T)
print('Scatter Matrix:\n', scatter_matrix)
end = time.time()
print(end - start)
'''

#computing th Covariance Matrix
cov_mat = np.cov([points[0,:],points[1,:],points[2,:]])
#print('Covariance Matrix:\n', cov_mat)

# Computing eigenvectors and corresponding eigenvalues
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)


for i in range(len(eig_val_cov)):
     eigvec_cov = eig_vec_cov[:,i].reshape(1,3).T
    



# Sorting the eigenvectors by decreasing eigenvalues
for ev in eig_vec_cov:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)


# Visually confirm that the list is correctly sorted by decreasing eigenvalues
'''
for i in eig_pairs:
    print(i[0])
'''

# Choosing k eigenvectors with the largest eigenvalues
matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
#print('Matrix W:\n', matrix_w)

# Transforming the samples onto the new subspace
transformed = matrix_w.T.dot(points)

rotation_matrix = np.zeros((2,2),dtype = float)
degree = degree /180.0 * math.pi
rotation_matrix[0,0] = math.cos(degree)
rotation_matrix[0,1] = -math.sin(degree)
rotation_matrix[1,0] = math.sin(degree)
rotation_matrix[1,1] = math.cos(degree)

transformed = np.matmul(rotation_matrix, transformed)


#find min max of transformed points
min_x = math.floor(min(transformed[0,:]))
min_y = math.floor(min(transformed[1,:]))
max_x = math.floor(max(transformed[0,:]))
max_y = math.floor(max(transformed[1,:]))

ncols =abs(max_y-min_y)+1
nrows = abs(max_x - min_x)+1
#calculate point density
count = np.zeros(( ncols , nrows) , dtype=int)
#print(count.shape)
#print(max_x , min_x , max_y, min_y)
start = time.time()
              
for i in range(len(transformed[0])):
    x = abs(math.floor(transformed[0,i])-min_x)
    y = abs(math.floor(transformed[1,i])-min_y)
    count[y,x] = count[y,x] +1
end = time.time()
print(end - start)

#output file
fo = open('density.asc',"w")

fo.write("ncols\t"+str(nrows)+"\n")
fo.write("nrows\t"+str(ncols)+"\n")
fo.write("xllcorner\t"+str(min_y)+"\n")
fo.write("yllcorner\t"+str(min_x)+"\n")
fo.write("cellsize\t"+str(1)+"\n")
fo.write("nodata_value\t"+str(-0.5)+"\n")
print(count.shape[0])
print(count.shape[1])
for i in reversed(range(count.shape[0])):
    for j in range(count.shape[1]):
        if j == 0:
            fo.write(str(count[i,j]))
        else:
            fo.write("\t"+str(count[i,j]) )
    fo.write("\n")
        
fo.close()


#visualize in 2D

plt.plot(transformed[0,0::PPSP], transformed[1,0::PPSP], 'o', markersize=1, color='blue', alpha=0.1, label='class1')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples with class labels')

plt.show()

#End of visualize in 2D


#visualize in 3D
'''
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10

ax.plot(x[0::PPSP], y[0::PPSP], z[0::PPSP] , 'o', markersize=1, color='blue', alpha=0.1, label='1')

for v in eig_vec_cov.T:
    a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)

ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')

plt.title('PointCloud')
ax.legend(loc='upper right')

plt.show()
'''
#End of visualize in 3D

'''
I = inFile.Classification
print(I)
'''
inFile.close()

