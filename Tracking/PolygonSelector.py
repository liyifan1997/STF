import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path
import cv2

class PolygonSelector:
    def __init__(self, ax, image):
        self.ax = ax
        self.image = image
        self.xs = []
        self.ys = []
        self.line, = ax.plot([], [], 'r-')  # 空的线对象，用于绘制多边形
        self.cid = self.line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

    def create_polygon(self):
        poly = Polygon(np.column_stack([self.xs, self.ys]), closed=True, edgecolor='r', facecolor='none')
        self.ax.add_patch(poly)
        plt.draw()

    def generate_mask(self):
        poly_path = Path(np.column_stack([self.xs, self.ys]))
        x, y = np.meshgrid(np.arange(self.image.shape[1]), np.arange(self.image.shape[0]))  # 生成网格点坐标矩阵
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T

        grid = poly_path.contains_points(points)
        grid = grid.reshape((self.image.shape[0], self.image.shape[1]))
        return grid