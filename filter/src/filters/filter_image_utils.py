import cv2
import numpy as np
from filter.src.filters.transform_functions import applyAffineTransform

def get_filter_part_of_face(img_editable, filter, landmarks, points, index_1, index_2, index_3, ears=False, scale=1):
    
    filter = cv2.cvtColor(filter, cv2.COLOR_BGR2RGB)
    height, width = filter.shape[0:2]

    if ears == True:
        coordinate = np.float32([[width/2, 0],
                                [0, height-1],
                                [width - 1, height-1]])

        tmp_height = landmarks[30][1] - landmarks[index_1][1]
        d = points[index_2][1] - points[index_3][1]
        (a, b) = points[index_1]
        (a, b) = (a + d/4, b - 2 * tmp_height + d/4)
        points[index_1] = (a, b)
        landmarks[index_1][1] = b
    else:
        coordinate = np.float32([[width/2, height/4],
                [width/4, height-height/4],
                [width - width/4, height-height/4]])
        
    rect = cv2.boundingRect(np.array([points[index_1], points[index_2], points[index_3]], np.int32))
    (x, y, w, h) = rect

    # Thêm tỉ lệ phóng đại cho coordinate_nose
    cx, cy = x + w // 2, y + h // 2 

    # Điều chỉnh kích thước vùng mũi dựa trên scale
    w1 = int(w * scale)
    h1 = int(h * scale)
    x = cx - w1 // 2
    y = cy - h1 // 2
    
    coordinate_nose = np.float32([[landmarks[index_1][0] - x , landmarks[index_1][1] - y ],
                                    [landmarks[index_2][0] - x , landmarks[index_2][1] - y ],
                                    [landmarks[index_3][0] - x , landmarks[index_3][1] - y ]])
    
    transform_img = applyAffineTransform(filter, coordinate_nose, coordinate, w1, h1)
    ret, mask = cv2.threshold(transform_img, 1, 255, cv2.THRESH_BINARY)
    roi = img_editable[int(y): int(y + h1), int(x): int(x + w1)]
    if roi.shape[:2] == mask.shape[:2]:  # Kiểm tra xem roi và mask có cùng kích thước không
        roi[np.where(mask)] = 0
        roi += transform_img
    output = img_editable
    return output


# get rectangle, verticles of triangle, cropped_tr_mask from 3 points of triangles
def get_features_of_3_vertices(points, triangle_index):
        pt1 = points[triangle_index[0]]
        pt2 = points[triangle_index[1]]
        pt3 = points[triangle_index[2]]

        # filter the rectange of triangle
        triangle = np.array([pt1, pt2, pt3], np.int32)
        rect = cv2.boundingRect(triangle)
        (x, y, w, h) = rect
        cropped_tr_mask = np.zeros((h, w), np.uint8)
        points_triangle = np.array([[pt1[0] - x, pt1[1] - y],
                        [pt2[0] - x, pt2[1] - y],
                        [pt3[0] - x, pt3[1] - y]], np.int32)
        cv2.fillConvexPoly(cropped_tr_mask, points_triangle, 255)
        
        return rect, points_triangle, cropped_tr_mask

