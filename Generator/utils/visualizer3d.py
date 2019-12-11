# -*- coding: utf-8 -*-

import logging

try:
    import mayavi.mlab as mlab
except:
    try:
        import mayavi.mlab as mlab
    except ImportError:
        logging.error('Failed to import mayavi')


class MayaviVisualizer3D(object):  # Note: add by MrRen-sdhm
    @staticmethod
    def show_points(point, color='r', scale_factor=.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        elif color == 'lb':  # light blue
            color_f = (0.22, 1, 1)
        else:
            color_f = (1, 1, 1)
        if point.size == 3:  # vis for only one point, shape must be (3,), for shape (1, 3) is not work
            point = point.reshape(3, )
            mlab.points3d(point[0], point[1], point[2], color=color_f, scale_factor=scale_factor)
        else:  # vis for multiple points
            mlab.points3d(point[:, 0], point[:, 1], point[:, 2], color=color_f, scale_factor=scale_factor)

    @staticmethod
    def show_line(un1, un2, color='g', scale_factor=0.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        else:
            color_f = (1, 1, 1)
        mlab.plot3d([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=color_f, tube_radius=scale_factor)

    @staticmethod
    def show_arrow(point, direction, color='lb', scale_factor=.03):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        elif color == 'lb':  # light blue
            color_f = (0.22, 1, 1)
        else:
            color_f = (1, 1, 1)
        mlab.quiver3d(point[0], point[1], point[2], direction[0], direction[1], direction[2],
                      scale_factor=scale_factor, line_width=0.05, color=color_f, mode='arrow')

    def show_origin(self, scale_factor=.03):
        self.show_arrow([0, 0, 0], [1, 0, 0], 'r', scale_factor)
        self.show_arrow([0, 0, 0], [0, 1, 0], 'g', scale_factor)
        self.show_arrow([0, 0, 0], [0, 0, 1], 'b', scale_factor)

    def show_surface_points(self, obj, color='lb'):
        surface_points, _ = obj.sdf.surface_points(grid_basis=False)
        self.show_points(surface_points,color=color)

    @staticmethod
    def show(title=None):
        if title is not None:
            mlab.title(title, size=0.5, color=(0, 0, 0))
        mlab.show()
