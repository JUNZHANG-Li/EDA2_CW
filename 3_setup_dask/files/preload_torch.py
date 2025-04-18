# Filename: preload_torch.py
try:
    import torch
    import torchvision
    # Attempt to import potentially problematic submodules/extensions
    import torchvision.ops
    import torchvision.io
    import torchvision.models
    import torchvision.transforms
    print("Preloaded torch and torchvision (including ops, io, models, transforms) on worker.")
except Exception as e:
    print(f"ERROR during preload: {e}")
    # Re-raise the exception so the preload failure is clearer if it happens
    raise