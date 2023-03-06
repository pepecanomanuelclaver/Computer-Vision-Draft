import glm
import numpy as np
import cv2 as cv

block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function

    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
    return data


def get_parameters(path):
    filename = path + '/config.xml'
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_READ)

    mtx = fs.getNode('mtx').mat()
    rvecs = fs.getNode('rvecs').mat()
    tvecs = fs.getNode('tvecs').mat()
    dist = fs.getNode('dist').mat()

    fs.release()
    return rvecs, tvecs, mtx, dist


def set_voxel_positions(width, height, depth, nframe):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    # create an array of the real world points in mm starting from the top left
    mask, data, rvecs, tvecs, mtx, dist, data_mesh = [], [], [], [], [], [], []

    for c in range(0, 4):
        cam = c + 1

        path = 'data/cam' + str(cam)
        r, t, m, d = get_parameters(path)
        rvecs.append(r)
        tvecs.append(t)
        mtx.append(m)
        dist.append(d)

        # Load the video
        cap = cv.VideoCapture('foreground'+str(cam)+'.avi')

        # Set the frame number to open
        cap.set(cv.CAP_PROP_POS_FRAMES, nframe)

        # Read the frame
        ret, frame = cap.read()

        # Change it to gray and append it to mask
        m = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask.append(m)


    # parameters for the reconstruction
    width = int(60)
    height = int(60)
    depth = int(60)
    resolution=2
    resolution2=2*resolution

    for x in range(width):
        for y in range(height):
            for z in range(depth):
                voxel = True
                for c in range(0, 4):

                    voxels = ((x - width / 2) * 40, (y - height / 2) * 40, -z * 40)
                    pixel_pts, _ = cv.projectPoints(voxels, rvecs[c], tvecs[c], mtx[c], dist[c])

                    pixel_pts = np.reshape(pixel_pts, 2)
                    pixel_pts = pixel_pts[::-1]

                    mask0 = mask[c]

                    (heightMask, widthMask) = mask0.shape

                    if 0 <= pixel_pts[0] < heightMask and 0 <= pixel_pts[1] < widthMask:
                        val = mask0[int(pixel_pts[0]), int(pixel_pts[1])]
                        if val == 0:
                            voxel = False

                if voxel:
                    data.append(
                        [(x * block_size/resolution - width / resolution2),(z * block_size/resolution), (y * block_size/resolution - depth / resolution2)])
                    data_mesh.append((x * block_size/resolution - width / resolution2,z * block_size/resolution,y * block_size/resolution - depth / resolution2))

    with open('data_mesh.txt', 'w') as f:
        f.write(str(data_mesh))
    f.close()

    data_mesh=np.array(data_mesh)

    """""
    # Create the mesh
    from scipy.spatial import Delaunay
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Define the list of points in 3D as float values
    points = data_mesh

    # Create a Delaunay triangulation using the points
    tri = Delaunay(points)

    # Create a 3D plot object
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Add the triangulation to the plot as a mesh
    for simplex in tri.simplices:
        vertices = points[simplex]
        poly = Poly3DCollection([vertices], alpha=0.25)
        poly.set_facecolor('blue')
        ax.add_collection3d(poly)

    # Set the axes limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    # Show the plot
    plt.show()
    """""

    return data


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    cameraposition = np.zeros((4, 3, 1))

    for c in range(0, 4):
        cam = c + 1
        path = 'data/cam' + str(cam)

        rvecs, tvecs, mtx, dist = get_parameters(path)

        rmtx, _ = cv.Rodrigues(rvecs)

        cameraposition[c] = (-np.dot(np.transpose(rmtx), tvecs / 115))

    cameraposition2 = [[cameraposition[0][0][0], -cameraposition[0][2][0], cameraposition[0][1][0]],
                       [cameraposition[1][0][0], -cameraposition[1][2][0], cameraposition[1][1][0]],
                       [cameraposition[2][0][0], -cameraposition[2][2][0], cameraposition[2][1][0]],
                       [cameraposition[3][0][0], -cameraposition[3][2][0], cameraposition[3][1][0]]]

    return cameraposition2


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.

    cam_rotations = []

    for c in range(0, 4):
        cam = c + 1
        path = 'data/cam' + str(cam)
        rvecs, tvecs, mtx, dist = get_parameters(path)

        rmtx, _ = cv.Rodrigues(rvecs)

        matrix = glm.mat4([
            [rmtx[0][0], rmtx[0][2], rmtx[0][1], tvecs[0][0]],
            [rmtx[1][0], rmtx[1][2], rmtx[1][1], tvecs[1][0]],
            [rmtx[2][0], rmtx[2][2], rmtx[2][1], tvecs[2][0]],
            [0, 0, 0, 1]
        ])

        glm_mat = glm.rotate(matrix, glm.radians(-90), (0, 1, 0))
        glm_mat = glm.rotate(glm_mat, glm.radians(180), (1, 0, 0))

        cam_rotations.append(glm_mat)

    return cam_rotations
