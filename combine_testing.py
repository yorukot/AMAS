import cv2
import torch
import matplotlib.pyplot as plt

from ultralytics import YOLO

# Initialization
filename = './data/thumb.jpg'

model_type = 'DPT_Hybrid'
midas = torch.hub.load('intel-isl/MiDaS', model_type)

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

if model_type == 'DPT_Large' or model_type == 'DPT_Hybrid':
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

model = YOLO('./models/yolo11s-seg.pt')

# Reading image
image = cv2.imread(filename)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)

# MiDaS detecting
with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode='bicubic',
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

# Yolo detecting
results = model(image)[0]

# Combining results
for result in results.boxes:
    cls_id = int(result.cls[0])
    class_name = model.names[cls_id]

    x1, y1, x2, y2 = map(int, result.xyxy[0])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Depth sampling at center
    if 0 <= cx < depth_map.shape[1] and 0 <= cy < depth_map.shape[0]:
        depth = depth_map[cy, cx]
        norm_depth = (depth_map - depth_map.min()) / \
            (depth_map.max() - depth_map.min())
        depth_at_point = norm_depth[cy, cx]

        if depth_at_point > 0.5:  # Works for now
            # Determine if it's center-ish
            h, w = image.shape[:2]
            roi_margin = 0.25

            x_min = int(w * roi_margin)
            x_max = int(w * (1 - roi_margin))
            y_min = int(h * roi_margin)
            y_max = int(h * (1 - roi_margin))

            # Check if box center lies within this ROI
            if x_min < cx < x_max and y_min < cy < y_max:
                # cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = str(depth_at_point)
                text_position = (x1, y1 - 10)

                cv2.putText(image, text, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(image, text, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.imshow('Debug View', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.imshow(depth_map)
# plt.show()
