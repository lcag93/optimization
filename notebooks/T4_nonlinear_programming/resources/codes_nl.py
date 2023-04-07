import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



###########################################################################################
######################################## PLOTTING  ########################################
###########################################################################################




def Plot_Package(x0, figsize = [5, 5], axis = False):
    
    # Define the vertices of the cuboid
    verts = [
        [(0, 0, 0), (x0[0], 0, 0), (x0[0], x0[1], 0), (0, x0[1], 0)],
        [(0, 0, x0[2]), (x0[0], 0, x0[2]), (x0[0], x0[1], x0[2]), (0, x0[1], x0[2])],
        [(0, 0, 0), (0, x0[1], 0), (0, x0[1], x0[2]), (0, 0, x0[2])],
        [(x0[0], 0, 0), (x0[0], x0[1], 0), (x0[0], x0[1], x0[2]), (x0[0], 0, x0[2])],
        [(0, 0, 0), (x0[0], 0, 0), (x0[0], 0, x0[2]), (0, 0, x0[2])],
        [(0, x0[1], 0), (x0[0], x0[1], 0), (x0[0], x0[1], x0[2]), (0, x0[1], x0[2])],
    ]

    # Define the faces of the cuboid
    faces = Poly3DCollection(verts, linewidths=1, edgecolors='darkmagenta', alpha=0.25)
    faces.set_facecolor('plum')

    # Create a 3D plot
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Add the faces to the plot
    ax.add_collection3d(faces)

    # Set the axis limits and labels
    ax.set_xlim(0, x0[0])
    ax.set_ylim(0, x0[1])
    ax.set_zlim(0, x0[2])
    ax.set_aspect('equal')
    
    if not axis:
        ax.set_axis_off()
    
    return ax
