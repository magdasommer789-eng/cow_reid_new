
import warnings
warnings.filterwarnings('ignore')
import sys, torch
sys.path.insert(0, '.')
from scripts.models_cnn import create_cnn_model
from scripts.models_transformer import create_transformer_model

device = 'cuda'
B, T, H, W = 2, 16, 224, 224
x = torch.randn(B, 3, T, H, W).to(device)

print("=== Testing C3D ===")
try:
    m = create_cnn_model('c3d', embedding_dim=512, pretrained=True, device=device)
    out = m(x)
    print(f"C3D OK: output={out.shape}, norm={out.norm(dim=1).mean():.3f}")
    del m
except Exception as e:
    print(f"C3D FAILED: {e}")
torch.cuda.empty_cache()

print("=== Testing X3D ===")
try:
    m = create_cnn_model('x3d', embedding_dim=512, pretrained=True, device=device)
    out = m(x)
    print(f"X3D OK: output={out.shape}, norm={out.norm(dim=1).mean():.3f}")
    del m
except Exception as e:
    print(f"X3D FAILED: {e}")
torch.cuda.empty_cache()

print("=== Testing Video Swin ===")
try:
    m = create_transformer_model('swin', embedding_dim=512, pretrained=True, device=device)
    out = m(x)
    print(f"Swin OK: output={out.shape}, norm={out.norm(dim=1).mean():.3f}")
    del m
except Exception as e:
    print(f"Swin FAILED: {e}")
torch.cuda.empty_cache()

print("=== Testing ViViT ===")
try:
    m = create_transformer_model('vivit', embedding_dim=512, pretrained=True, device=device, num_frames=16)
    out = m(x)
    print(f"ViViT OK: output={out.shape}, norm={out.norm(dim=1).mean():.3f}")
    del m
except Exception as e:
    print(f"ViViT FAILED: {e}")
torch.cuda.empty_cache()

print("Smoke test done.")
