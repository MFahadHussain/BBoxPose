import torch

def inspect_ckpt(path):
    print(f"\n--- {path} ---")
    data = torch.load(path, map_location='cpu')
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        if 'meta' in data:
            print(f"Meta: {list(data['meta'].keys())}")
            if 'dataset_meta' in data['meta']:
                print(f"Dataset Meta: {data['meta']['dataset_meta']}")
        if 'state_dict' in data:
             print(f"State dict keys sample: {list(data['state_dict'].keys())[:5]}")
        elif 'model' in data:
             print(f"Model keys sample: {list(data['model'].keys())[:5]}")
    else:
        print("Not a dict")

inspect_ckpt("checkpoints/rtmdet-ins-l-mask.pth")
inspect_ckpt("checkpoints/MaskPose/MaskPose-s-1.1.0.pth")
inspect_ckpt("checkpoints/SAM-pose2seg_hiera_b+.pt")
