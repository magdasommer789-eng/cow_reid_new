
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
arr = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
try:
    t = torch.tensor(arr)
    print("torch.tensor OK:", t.shape)
except Exception as e:
    print("torch.tensor FAILED:", e)
try:
    t2 = torch.from_numpy(arr)
    print("torch.from_numpy OK")
except Exception as e:
    print("torch.from_numpy FAILED:", e)
