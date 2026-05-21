
import warnings
warnings.filterwarnings("ignore")
import sys, torch
sys.path.insert(0, ".")
from scripts.models_cnn import create_cnn_model
from scripts.models_transformer import create_transformer_model

device = "cuda"
B, T, H, W = 2, 16, 224, 224
x = torch.randn(B, 3, T, H, W).to(device)

for name, fn, kw in [
    ("C3D",   create_cnn_model,         {"model_name":"c3d", "pretrained":False}),
    ("X3D",   create_cnn_model,         {"model_name":"x3d", "pretrained":True}),
    ("SWIN",  create_transformer_model, {"model_name":"swin", "pretrained":True}),
    ("VIVIT", create_transformer_model, {"model_name":"vivit","pretrained":True,"num_frames":16}),
]:
    try:
        m = fn(device=device, embedding_dim=512, **kw)
        out = m(x)
        norms = out.norm(dim=1).mean().item()
        print(f"  {name} OK: shape={list(out.shape)}, norm={norms:.3f}")
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {name} FAILED: {e}")
        torch.cuda.empty_cache()

print("Done.")
