import torch, cv2, numpy as np
import math
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# 1. SET UP DEVICE & MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas_model_type = "DPT_Large"       # or "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
midas.to(device).eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# 2. LOAD IMAGE
img_bgr = cv2.imread("input.jpg")
if img_bgr is None:
    raise FileNotFoundError("input.jpg not found")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)

# 3. PREPROCESS
H, W = img_rgb.shape[:2]
newH, newW = (math.floor(H/32)*32, math.floor(W/32)*32)
custom_transform = Compose([
    Resize((newH, newW), interpolation=Image.BICUBIC),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std =[0.229, 0.224, 0.225])
])
x = custom_transform(img_pil)
x = x.unsqueeze(0).to(device)

# 4. DEPTH INFERENCE
with torch.no_grad():
    pred = midas(x)

depth_raw = torch.nn.functional.interpolate(
    pred.unsqueeze(1),
    size=(H, W),
    mode="bilinear",
    align_corners=False
).squeeze().cpu().numpy()

# 5. NORMALIZE DEPTH
d_min, d_max = depth_raw.min(), depth_raw.max()
depth_norm = ((depth_raw - d_min) / (d_max - d_min)).astype(np.float32)

# 6. SAVE RAW DEPTH (COLOR & GRAYSCALE)
depth_vis_raw = (255 * depth_norm).astype(np.uint8)
cv2.imwrite("depth_raw.png", cv2.applyColorMap(depth_vis_raw, cv2.COLORMAP_INFERNO))
cv2.imwrite("depth_raw_gray.png", depth_vis_raw)

# 7. EDGE-AWARE REFINEMENT
img_bgr_float = img_bgr.astype(np.float32)
depth_refined = cv2.ximgproc.jointBilateralFilter(
    img_bgr_float,
    depth_norm,
    9,
    75,
    75
)

# 8. SAVE REFINED DEPTH (COLOR & GRAYSCALE)
depth_vis_ref = (255 * depth_refined).astype(np.uint8)
cv2.imwrite("depth_refined.png", cv2.applyColorMap(depth_vis_ref, cv2.COLORMAP_INFERNO))
cv2.imwrite("depth_refined_gray.png", depth_vis_ref)

print("Saved depth_raw.png, depth_raw_gray.png, depth_refined.png, and depth_refined_gray.png")
