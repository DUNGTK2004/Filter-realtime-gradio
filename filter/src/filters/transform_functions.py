import math
import cv2
import numpy as np

# find the third point of triangle from 2 point by rotate a point around another point
def find_new_point(x1, y1, x2, y2, alpha):
    dx = x2 - x1
    dy = y2 - y1

    #goc xoay
    alpha_radian = alpha * math.pi/180
    x_new = x1 + math.cos(alpha_radian) * dx + math.sin(alpha_radian) * dy
    y_new = y1 + math.sin(-alpha_radian) * dx + math.sin(alpha_radian) * dy
    return [int(x_new), int(y_new)]

# extract index (get the first index)
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

#Get indexes of triangles
def get_index_triangles(rect, points, point):
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((point == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((point == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((point == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    return indexes_triangles

#Affine transform for 2 set of points
def applyAffineTransform(src, points_1, points_2, w, h):
    M = cv2.getAffineTransform(points_2, points_1)
    dst = cv2.warpAffine(src, M, (w, h))
    return dst