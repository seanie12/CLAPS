import torch

ckpt_file = "../../save/qg-best/model.pt"
ckpt = torch.load(ckpt_file, map_location="cpu")
args = ckpt["args"]
print(args.lr)

