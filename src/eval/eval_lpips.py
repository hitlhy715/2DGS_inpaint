import os
import torch
from tqdm import tqdm
import lpips
from torchvision.io import read_image

device = 'cuda:0'

dir1 = ''  # TODO
dir2 = ''  # TODO

path1 = [os.path.join(dir1,  _ ) for _ in os.listdir(dir1)]
path2 = [os.path.join(dir2,  _ ) for _ in os.listdir(dir2)]

path1 = sorted(path1)
path2 = sorted(path2)


model = lpips.LPIPS().to(device)

res = []
for p1,p2 in tqdm(zip(path1, path2)):
    tensor1 = read_image(p1).to(device)
    tensor2 = read_image(p2).to(device)

    tensor1 = tensor1/255
    tensor2 = tensor2/255

    r=model(tensor1, tensor2)
    res.append(r.item())

print(sum(res)/len(res))
