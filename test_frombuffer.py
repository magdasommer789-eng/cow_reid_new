
import warnings; warnings.filterwarnings("ignore")
import torch
from PIL import Image
import io
# Create a tiny 4x4 RGB PIL image
img = Image.new("RGB", (4, 4), color=(255, 0, 0))
raw = img.tobytes()
t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
H, W = img.size[1], img.size[0]
t = t.view(H, W, 3).permute(2, 0, 1).float() / 255.0
print("frombuffer OK:", t.shape, "val=", round(t[0,0,0].item(), 2))
