import cv2
import numpy as np
from filter.src.predictors.pred_landmark import predict_landmarks
from filter.src.filters.filter_image_utils import get_filter_part_of_face, get_features_of_3_vertices
from filter.src.filters.transform_functions import get_index_triangles, applyAffineTransform

def filter_on_image(img, face_swap, points2, filter_name, model, transform, face_model):
    if img is None:
        return img
    img_editable = img.copy()
    rgb_image = cv2.cvtColor(img_editable, cv2.COLOR_BGR2RGB)

    face_result = face_model.predict(rgb_image, conf=0.40)
    if face_result:
        for info in face_result:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                crop_rgb_image = rgb_image[int(y1):int(y2), int(x1):int(x2)]
                landmarks = predict_landmarks(crop_rgb_image, x1, y1, x2, y2, model, transform)

                #Convert landmark to suitable type
                point = np.array(landmarks, np.int32)                
                points = [tuple(point) for point in point.tolist()]


                if face_swap is not None:
                    #Create mask of convexhull
                    convexhull = cv2.convexHull(point)
                    mask_merge = np.zeros_like(img_editable, dtype=np.uint8)
                    cv2.fillPoly(mask_merge, [convexhull], (255, 255, 255))

                    face_swap = cv2.cvtColor(face_swap, cv2.COLOR_BGR2RGB)

                    #create empty image
                    img_new_face = np.zeros_like(img_editable)
                    gray_image = cv2.cvtColor(img_editable, cv2.COLOR_BGR2GRAY)

                    #Get index of triangles for similar index in the second triangles
                    rect = cv2.boundingRect(convexhull)
                    indexes_triangles = get_index_triangles(rect, points, point)

                    # it will be used for fill mask to face 
                    check = True
                    for triangle_index in indexes_triangles:
                        #Get the necessary features from triangles
                        rect1, points_1_triangle, cropped_tr1_mask = get_features_of_3_vertices(points, triangle_index)
                        (x, y, w, h) = rect1
                        rect2, points_2_triangle, cropped_tr2_mask = get_features_of_3_vertices(points2, triangle_index) 
                        (_x, _y, _w, _h) = rect2
                        cropped_triangle2 = face_swap[_y: _y + _h, _x: _x + _w]
                        
                        # Transform triangle 1 to triangle 2 by affine
                        points_1_triangle = np.float32(points_1_triangle)
                        points_2_triangle = np.float32(points_2_triangle)
                        warped_triangle = applyAffineTransform(cropped_triangle2, points_1_triangle, points_2_triangle, w, h)
                        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr1_mask)

                        # reconstructe destination face
                        triangle_area = img_new_face[y : y + h, x : x + w]
                        try:
                            triangle_area_gray = cv2.cvtColor(triangle_area, cv2.COLOR_BGR2GRAY)
                            _, mask_triangles_designed = cv2.threshold(triangle_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
                            triangle_area = cv2.add(triangle_area, warped_triangle)
                            img_new_face[y : y + h, x : x + w] = triangle_area
                        except Exception as e:
                            check = False

                    # face swapped (putting 2st to 1st)
                    if check == True:
                        img1_face_mask = np.zeros_like(gray_image)
                        img1_head_mask = cv2.fillConvexPoly(img1_face_mask, convexhull, 255)
                        img1_face_mask = cv2.bitwise_not(img1_head_mask)
                        img1_head_noface = cv2.bitwise_and(img_editable, img_editable, mask=img1_face_mask)
                        result = cv2.add(img1_head_noface, img_new_face)

                        if filter_name == "Face Ronaldo":
                            (x, y, w, h) = cv2.boundingRect(convexhull)
                            center_face1 = (int((2 * x + w)/2), int((2 * y + h) /2))
                            result = cv2.seamlessClone(result, img_editable, img1_head_mask, center_face1, cv2.MIXED_CLONE)

                        img_editable = result

                # assign filter for nose
                elif "nose" in filter_name:
                    filter_part_name = cv2.imread(f"filter/filter_images/{filter_name}.png")
                    img_editable = get_filter_part_of_face(img_editable, filter_part_name, landmarks, points, 29, 31, 35, ears=False, scale=2.5)

                # assign filter for ears
                elif "ears" in filter_name:
                    filter_part_name = cv2.imread(f"filter/filter_images/{filter_name}.png")
                    img_editable = get_filter_part_of_face(img_editable, filter_part_name, landmarks, points, 27, 0, 16, ears=True, scale=1.5)

                # don't use filter
                elif filter_name == "landmarks":
                    for (x, y) in points:
                        cv2.circle(img_editable, (int(x), int(y)), 3, (255, 0, 0), -1)  # Điểm màu xanh lá

    return img_editable