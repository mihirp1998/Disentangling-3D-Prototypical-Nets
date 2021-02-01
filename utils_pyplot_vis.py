from mpl_toolkits import mplot3d

import matplotlib
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


import vis_np


#color = ["r", "g", "b", "k", "m", "y"]

"""
ax.view_init(0,0):
x to the front, y to the right, z to the top

ax.view_init(90,0): (rotate along the y axis for 90 degree, clockwise toward z->x)
x to the bottom, y to the right, z to the top

ax.view_init(90, 0): rotate_along the z axis for 90 degree, clockwise toward y ->x
x to the left, y to the front, z to the top

"""

def set_coord(ax, xlims = [-0.3, 0.3], ylims = [-0.3, 0.3], zlims=[0, 0.4], coord="xright-ydown"):
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    assert(xlims[0] <= xlims[1])
    assert(ylims[0] <= ylims[1])
    assert(zlims[0] <= zlims[1])
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    ax.set_zlim(zlims[0], zlims[1])

    if coord == "xright-ydown":
        ax.view_init(-80, -90)
    elif coord == "xright-ydown-topview":
        ax.view_init(-10, -90)
    elif coord == "xright-yfront":
        ax.view_init(10, -90)
    elif coord == "xright-yback":
        ax.view_init(0, -90)
    elif coord == "xleft-ydown": #Adam coordinate from the back
        ax.view_init(110, 90)
    elif coord == "xleft-ydown-topview": #Adam coordinate from the back
        ax.view_init(170, 90)
    else:
        return ValueError


def create_fig(fig_id,  xlims = [-0.3, 0.3], ylims = [-0.3, 0.3], zlims=[-0.4, 0.4], coord="xright-ydown", title="", mode="3d"):
    fig = plt.figure(fig_id)
    if mode == "3d":
        title += coord
        if coord == "xright-ydown":
            title += "(adam)"
        elif coord == "xleft-ydown":
            title += "(adam back)"
        elif coord == "xright-ydown-topview":
            title += "(adam, top-view)"
        elif coord == "xleft-ydown-topview":
            title += "(adam back, top_view)"



    fig.suptitle(title)
    return fig

def plot_images(image, fig_id=1, fig=None, subplot=111, fig_title="", title=""):
    """
    images: SXHXWXC
    """
    if fig == None:
        fig = create_fig(fig_id, mode="2d", title=fig_title)

    ax = fig.add_subplot(subplot)
    if title is not "":
        ax.set_title(title)
    S, H, W, C = image.shape
    image = np.reshape(image, [S*H, W, C])
    if C == 1:
        image = np.tile(image, [1,1,3])
    ax.imshow(image)
    return fig, ax

def plot_pointcloud(points, ax=None, fig_id=1, fig=None, subplot=111, xlims = [-0.3, 0.3], ylims = [-0.3, 0.3], zlims=[0, 0.4], c='k',
                    coord="xright-ydown", title=""):
    """
    points: VX3
    """
    if ax == None: # need to create ax
        if fig == None:
            fig=create_fig(fig_id,  xlims=xlims, ylims=ylims, zlims=zlims, coord=coord, title = "pointclouds-")
        ax = fig.add_subplot(subplot, projection='3d')
        set_coord(ax, xlims=xlims, ylims=ylims, zlims=zlims, coord=coord)
        if title is not "":
            ax.set_title(title)

    ax.scatter(points[:,0], points[:,1], points[:,2], c=c, marker='.', linewidths=0.01)

    return fig, ax

def plot_pointclouds(points, ax=None, fig_id=1, fig=None, subplot=111, xlims = [-0.3, 0.3], ylims = [-0.3, 0.3], zlims=[0, 0.4],
	                 color=["k", "m", "y","r", "g", "b"], coord="xright-ydown", title="",  alpha=0.2):
    """
    points: BXVX3
    color: light grey, light brown, light purple, skin color: ["#C0C0C0", "#D2B48C", "#E0B0FF", "#EDC9Af", "m", "y"]
    """
    num_views = points.shape[0]
    for view_id in range(num_views):
        fig, ax = plot_pointcloud(points[view_id], ax=ax, fig_id=fig_id, fig=fig, subplot=subplot, c=color[view_id], xlims=xlims, ylims=ylims, zlims=zlims, coord=coord, title=title)

    return fig, ax


def plot_bounding_box():
	pass

def plot_cam(origin_T_cams, fig_id=1, fig=None, subplot=111, ax=None, color=["r", "g", "b", "k", "m", "y"],\
             xlims = [-0.3, 0.3], ylims = [-0.3, 0.3], zlims=[0, 0.4], msg="cam_", coord="xright-ydown", length=0.05, title=""):
    """
    origin_T_cams: NOBJSX4X4
    """
    if ax == None: # need to create ax
         if fig == None:
             fig = create_fig(fig_id,  xlims=xlims, ylims=ylims, zlims=zlims, coord=coord, title = "cams-")
         ax = fig.add_subplot(subplot, projection='3d')
         set_coord(ax, xlims=xlims, ylims=ylims, zlims=zlims, coord=coord)
         if title is not "":
            ax.set_title(title)

    num_cam = origin_T_cams.shape[0]

    msg_end = ""
    for cam_id in range(num_cam):
        if num_cam is not 1:
            msg_end = f"{cam_id}"
        vis_np.plot_cam(ax, origin_T_cams[cam_id, :, :], color=color[cam_id], msg=msg + msg_end, length=length)

    return fig, ax


def plot_cube(cube_definitions,fig=None, subplot=111, ax=None, color=["r", "g", "b", "k", "m", "y"],\
             xlims = [-0.3, 0.3], ylims = [-0.3, 0.3], zlims=[0, 0.4], msg="cam_", coord="xright-ydown", length=0.05, title=""):
    for cube_definition in cube_definitions:
        if ax == None: # need to create ax
             if fig == None:
                 fig = create_fig(fig_id,  xlims=xlims, ylims=ylims, zlims=zlims, coord=coord, title = "cams-")
             ax = fig.add_subplot(subplot, projection='3d')
             set_coord(ax, xlims=xlims, ylims=ylims, zlims=zlims, coord=coord)
             if title is not "":
                ax.set_title(title)
        cube_definition_array = [
            np.array(list(item))
            for item in cube_definition
        ]

        points = []
        points += cube_definition_array
        vectors = [
            cube_definition_array[1] - cube_definition_array[0],
            cube_definition_array[2] - cube_definition_array[0],
            cube_definition_array[3] - cube_definition_array[0]
        ]

        points += [cube_definition_array[0] + vectors[0] + vectors[1]]
        points += [cube_definition_array[0] + vectors[0] + vectors[2]]
        points += [cube_definition_array[0] + vectors[1] + vectors[2]]
        points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

        points = np.array(points)

        edges = [
            [points[0], points[3], points[5], points[1]],
            [points[1], points[5], points[7], points[4]],
            [points[4], points[2], points[6], points[7]],
            [points[2], points[6], points[3], points[0]],
            [points[0], points[2], points[4], points[1]],
            [points[3], points[6], points[7], points[5]]
        ]
        faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
        faces.set_facecolor((0,0,1,0.1))

        ax.add_collection3d(faces)

        # Plot the points themselves to force the scaling of the axes
        ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

        # ax.set_aspect('equal')

    return fig, ax


def plot_adam_voxel(voxel, fig_id=1, fig=None, subplot=111, ax=None,\
                    msg="", coord="xright-ydown", length=0.05, title=""):

    """
    plot voxel
    """
    assert(len(voxel.shape)==3)
    voxel = voxel.numpy()
    H, W, D = voxel.shape
    voxel = np.transpose(voxel, [1, 0, 2])


    xlims = [0, W]
    ylims = [0, H]
    zlims = [0, D]
    if ax == None: # need to create ax
         if fig == None:
             fig = create_fig(fig_id,  xlims=xlims, ylims=ylims, zlims=zlims, coord=coord, title = "cams-")
         ax = fig.add_subplot(subplot, projection='3d')
         set_coord(ax, xlims=xlims, ylims=ylims, zlims=zlims, coord=coord)
         if title is not "":
            ax.set_title(title)


    occ_indices = np.argwhere(voxel)

    colors = np.empty([H, W, D, 4], dtype=np.float)
    colors[voxel >= 0.5] = [0,0,1,0.2]
    colors[voxel < 0.5] = [0,0,0,0]

    #colors = np.reshape(colors, [-1 ,4])

    # and plot everything
    ax.voxels(voxel, facecolors=colors, edgecolor='k')
    return fig, ax


def plot_3dflow(flow, mem_coord=None, fig_id=1, fig=None, subplot=111, ax=None,\
                    msg="", coord="xright-ydown", length=0.05, title=""):

    """
    flow: H X W X D X 3
    """

    assert(len(flow.shape)==4, flow.shape[-1] == 3)
    flow = flow.numpy()
    H, W, D, _ = flow.shape
    flow = np.transpose(flow, [3, 1, 0, 2])


    #scalex = W/(mem_coord.coord.X_MAX - mem_coord.coord.X_MAX)
    #scaley = H/(mem_coord.coord.Y_MAX - mem_coord.coord.Y_MAX)
    #scalez = D/(mem_coord.coord.Z_MAX - mem_coord.coord.Z_MAX)


    if mem_coord == None:
        xlims = [0, W]
        ylims = [0, H]
        zlims = [0, D]
        y, x, z = np.meshgrid(np.linspace(0, H-1, H) + 0.5,
          np.linspace(0, W-1, W) + 0.5,
          np.linspace(0, D-1, D) + 0.5)
    else:
        xlims = [mem_coord.coord.XMIN, mem_coord.coord.XMAX]
        ylims = [mem_coord.coord.YMIN, mem_coord.coord.YMAX]
        zlims = [mem_coord.coord.ZMIN, mem_coord.coord.ZMAX]
        scale_y = (mem_coord.coord.YMAX - mem_coord.coord.YMIN)/(H-1)
        scale_x = (mem_coord.coord.XMAX - mem_coord.coord.XMIN)/(W-1)
        scale_z = (mem_coord.coord.ZMAX - mem_coord.coord.ZMIN)/(D-1)
        y, x, z = np.meshgrid(np.linspace(mem_coord.coord.YMIN + 0.5*scale_y, mem_coord.coord.YMAX - 0.5*scale_y, H),
                  np.linspace(mem_coord.coord.XMIN + 0.5*scale_x , mem_coord.coord.XMAX - 0.5*scale_x, W),
                  np.linspace(mem_coord.coord.ZMIN + 0.5*scale_z, mem_coord.coord.ZMAX - 0.5*scale_z, D))
    if ax == None: # need to create ax
         if fig == None:
             fig = create_fig(fig_id,  xlims=xlims, ylims=ylims, zlims=zlims, coord=coord, title = "cams-")
         ax = fig.add_subplot(subplot, projection='3d')
         set_coord(ax, xlims=xlims, ylims=ylims, zlims=zlims, coord=coord)
         if title is not "":
            ax.set_title(title)

    x = x.reshape(np.product(x.shape))
    y = y.reshape(np.product(y.shape))
    z = z.reshape(np.product(z.shape))
    u, v, w = flow
    #lengths = np.sqrt(u**2+v**2+w**2)
    u = u.reshape(np.product(x.shape))
    v = v.reshape(np.product(y.shape))
    w = w.reshape(np.product(z.shape))


    ax.quiver(x, y, z, u, v, w, linewidths=0.05)
    return fig, ax

def plot_voxel(voxel, fig_id=1, fig=None, subplot=111, ax=None,\
                    msg="", coord="xright-ydown", length=0.05, title=""):

    """
    plot voxel
    """
    assert(len(voxel.shape)==3)
    voxel = voxel.numpy()
    W, H, D = voxel.shape
    #voxel = np.transpose(voxel, [1, 0, 2])


    xlims = [0, W]
    ylims = [0, H]
    zlims = [0, D]
    if ax == None: # need to create ax
         if fig == None:
             fig = create_fig(fig_id,  xlims=xlims, ylims=ylims, zlims=zlims, coord=coord, title = "cams-")
         ax = fig.add_subplot(subplot, projection='3d')
         set_coord(ax, xlims=xlims, ylims=ylims, zlims=zlims, coord=coord)
         if title is not "":
            ax.set_title(title)


    occ_indices = np.argwhere(voxel)

    colors = np.empty([W, H, D, 4], dtype=np.float)
    colors[voxel > 0.5] = [0,0,1,.2]


    # and plot everything
    ax.voxels(voxel, facecolors=colors, edgecolor='k')
    return fig, ax



