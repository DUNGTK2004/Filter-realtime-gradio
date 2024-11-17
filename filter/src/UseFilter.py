import cv2
from ultralytics import YOLO
from filter.src.predictors.pred_landmark import load_model, predict_landmarks, get_points_of_mask
from filter.src.filters.filter_image import filter_on_image
import os


# Load your pre-trained model (replace with your actual model loading code)
model, transform = load_model("filter/src/models/checkpoint/model.pth")

# Load YOLO face detection model
face_model = YOLO('YOLO_model/yolov8n-face.pt')

# face for swap
## Ronaldo
face_ronaldo = cv2.imread("filter/filter_images/ronaldo_filter.jpeg")
height_test, width_test = face_ronaldo.shape[:2]
points_ronaldo = predict_landmarks(face_ronaldo, 0, 0, width_test-1, height_test-1, model, transform)

## Suzy 
face_suzy = cv2.imread("filter/filter_images/suzy_filter.jpg")
height_suzy, width_suzy = face_suzy.shape[:2]
points_suzy = predict_landmarks(face_suzy, 0, 0, width_suzy-1, height_suzy-1, model, transform)

## Suzy 
face_faker = cv2.imread("filter/filter_images/faker_filter.jpg")
height_faker, width_faker = face_faker.shape[:2]
points_faker = predict_landmarks(face_faker, 0, 0, width_faker-1, height_faker-1, model, transform)

# mask
## squid game
mask_path_squid_game = "filter/filter_images/squid_game_front_man.png"
mask_csv_path_squid_game = "filter/filter_images/squid_game_front_man.csv"
mask_squid_game = cv2.imread(mask_path_squid_game)
points_squid_game = get_points_of_mask( mask_csv_path_squid_game)

## anonymous
mask_path_anonymous = "filter/filter_images/anonymous.png"
mask_csv_path_anonymous = "filter/filter_images/anonymous_annotations.csv"
mask_anonymous = cv2.imread(mask_path_anonymous)
points_anonymous = get_points_of_mask( mask_csv_path_anonymous)


filter_img = {
    "Face Ronaldo": face_ronaldo,
    "Face Suzy": face_suzy,
    "Face Faker": face_faker,
    "Mask Squid Game": mask_squid_game,
    "Mask Anonymous": mask_anonymous
}

filter_point = {
    "Face Ronaldo": points_ronaldo,
    "Face Suzy": points_suzy,
    "Face Faker": points_faker,
    "Mask Squid Game": points_squid_game,
    "Mask Anonymous": points_anonymous
}

def apply_filter_on_image(img, filter_name):
    global filter_img
    global filter_point
    # Lấy giá trị từ từ điển, nếu không có thì trả về None
    img_filter = filter_img.get(filter_name, None)
    points_filter = filter_point.get(filter_name, None)

    output = filter_on_image(img, img_filter, points_filter, filter_name, model, transform, face_model)
    return output


def apply_filter_on_video(video, filter_name):
    vd = cv2.VideoCapture(video)
    if vd.isOpened() == False:
        print("error")
    frame_width = int(vd.get(3))
    frame_height = int(vd.get(4))

    size = (frame_width, frame_height)
    path_video = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "video_output/filename.avi")
    print(path_video)
    result = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'MJPG'),
                             15, size)
    while (True):
        ret, frame = vd.read()
        if ret == True:
            frame_drawed = apply_filter_on_image(frame, filter_name)
            result.write(frame_drawed)
        else: 
            break
    return path_video
