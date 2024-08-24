import numpy as np
import torch

from canary_lib.canary_defense_method.img_preprocess.quilting.quilting import quilting
from canary_lib.canary_defense_method.img_preprocess.jpeg_trans import Transform
from canary_lib.canary_defense_method.img_preprocess.tvm.tvm import reconstruct as tvm
from canary_lib.canary_defense_method.img_preprocess.randomization import randomization
from canary_lib.canary_defense_method.img_preprocess.quantize_trans import Transform as quantize
from canary_lib.canary_defense_method.img_preprocess.webp.webp_trans import Transform as webp


def quilting_a(img):
    img = img[0].permute(1, 2, 0) * 255
    img = quilting(img.detach().cpu()).permute(2, 0, 1).unsqueeze(0).cuda() / 255
    return img


def jpeg_a(img, quality=50):
    img = img[0].permute(1, 2, 0) * 255
    img = Transform._jpeg_compression(img, quality).permute(2, 0, 1).unsqueeze(0).cuda() / 255
    return img


def tvm_a(img):
    img = tvm(img[0].detach().cpu(), 0.05, 'none', 0.03).unsqueeze(0).cuda()
    return img


def rand_a(img):
    defense = randomization.Randomization('cuda')
    img = img[0].permute(1, 2, 0) * 255
    img = defense.forward(img.permute(2, 0, 1).unsqueeze(0))[0]
    img = np.clip(img.detach().cpu().numpy() / 255, 0, 1, out=None) * 255
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    return img


def quantize_a(img, depth=3):
    img = quantize._quantize_img(img[0].detach().cpu().permute(1, 2, 0) * 255, depth).permute(2, 0, 1).unsqueeze(
        0).cuda() / 255
    return img


def webp_a(img, quality=60):
    img = webp.webp_compression(img[0].permute(1, 2, 0) * 255, quality).permute(2, 0, 1).unsqueeze(0).cuda() / 255
    return img


