import torch
from filter.src.models.models import NN
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from filter.src.filters.transform_functions import find_new_point
import pandas as pd
# Load your pre-trained model (replace with your actual model loading code)
SIZE = 224
device = "cpu"


def load_model(model_file: str = "filter/src/models/checkpoint/model1_crop.pth"):
    pth = torch.load(model_file, weights_only=True)
    model = NN()
    model.load_state_dict(pth)

    # Transform 
    transform = A.Compose(
        [
        A.RandomBrightnessContrast(p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.Resize(SIZE, SIZE),
        ToTensorV2()],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        )   
    return model, transform


def predict_landmarks(image, x1, y1, x2, y2, model = None, transform = None):
    if model == None and transform == None:
        model, transform = load_model()
    image_transform = transform(image=image)
    image_tensor = image_transform["image"].float().unsqueeze(0).to(device)

    with torch.no_grad():  # Không cần lưu gradient cho dự đoán
        model.eval()  # Chuyển mô hình về chế độ evaluation
        test_pred_logits = model(image_tensor)

    predicted_points = test_pred_logits.view(-1, 68, 2)  # Giả sử bạn dự đoán 68 điểm (x, y)
    predicted_points[0][:, 0] = predicted_points[0][:, 0] * (x2-x1) + x1
    predicted_points[0][:, 1] = predicted_points[0][:, 1] * (y2-y1) + y1

    landmarks = predicted_points[0]

    #get new point in top of face
    (x_tmp, y_tmp) = landmarks[1]
    (x2, y2) = landmarks[15]
    (x1, y1) = (int((x_tmp + x2)/2), int((y_tmp + y2)/2))
    alpha_arr = [30, 60, 90, 120, 150]
    for alpha in alpha_arr:
        new_point = find_new_point(x1, y1, x2, y2, alpha)
        new_point_tensor = torch.tensor([new_point])
        landmarks = torch.cat((landmarks, new_point_tensor), dim=0)
    points = np.array(landmarks, np.int32)
    return points

def get_points_of_mask(mask_csv_path):
    df = pd.read_csv(mask_csv_path, header=None)
    points2 = []
    for i in range(0,68):
        points2.append((df[1][i], df[2][i]))
    (x_tmp, y_tmp) = points2[1]
    (x2, y2) = points2[15]
    (x1, y1) = (int((x_tmp + x2)/2), int((y_tmp + y2)/2))

    alpha_arr = [30, 60, 90, 120, 150]
    for alpha in alpha_arr:
        new_point = find_new_point(x1, y1, x2, y2, alpha)
        points2.append(new_point)
    return points2

