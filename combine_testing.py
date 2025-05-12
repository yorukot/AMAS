import cv2
import torch

from ultralytics import YOLO

filename = 'dog.jpg'

model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

model = YOLO('./models/yolo11n-seg.pt')

image = cv2.imread(filename)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)

# MiDaS detecting
with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

results = model(image)

for result in results.boxes:
    cls_id = int(result.cls[0])
    class_name = model.names[cls_id]

    x1, y1, x2, y2 = map(int, result.xyxy[0])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Depth sampling at center
    if 0 <= cx < depth_map.shape[1] and 0 <= cy < depth_map.shape[0]:
        depth = depth_map[cy, cx]

        if depth < 0.5:  # You’ll need to tune this
            # Determine if it's center-ish
            frame_center = image.shape[1] // 2
            if abs(cx - frame_center) < image.shape[1] * 0.2:
                print(f"{class_name} ahead!")
