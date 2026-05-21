
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,".")
import numpy as np, torch
from scripts.dataset import frames_to_tensor, get_frame_transform

# Simulate a clip of 4 frames, 224x224
frames = np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8)
transform = get_frame_transform("train", 224)
clip = frames_to_tensor(frames, transform)
print("clip shape:", clip.shape)
print("clip dtype:", clip.dtype)
print("clip mean:", round(clip.mean().item(), 3))
print("Dataset test PASSED")
