import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision.transforms import ToTensor

def quilting(image, quilting_size=2, kemeans=16):
    M = quilting_size
    K = kemeans

    # 将图像转换为Tensor，并移到GPU上
    # image_tensor = ToTensor()(image).unsqueeze(0).cuda()
    # image_tensor = image.unsqueeze(0).cuda()

    # 获取图像宽和高
    # width, height = image.size
    width = image.shape[0]
    height = image.shape[1]
    # 将图像分割为N个大小为M x M的小块
    NX = width // M
    NY = height // M
    patches = []
    tensor_shape = (12544, 12)
    tensor = torch.zeros(tensor_shape)
    cnt = 0
    for i in range(NX):
        for j in range(NY):
            x1 = i * M
            y1 = j * M
            x2 = x1 + M
            y2 = y1 + M
            patch = image[y1:y2, x1:x2, :]
            patch_array = patch.reshape(-1).float()
            tensor[cnt] = patch_array
            cnt+=1
            patches.append(patch_array)

    # 运行Kmeans聚类
    # print(len(patches))
    # patches = torch.cat(patches, dim=0)
    # print(patches.shape)
    # patches = patches.cpu().numpy()
    kmeans = KMeans(n_clusters=K, random_state=0).fit(tensor)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 构建输出图像
    out_image = Image.new('RGB', (width, height))
    for i in range(NX):
        for j in range(NY):
            patch_index = i * NX + j
            label = labels[patch_index]
            centroid_patch = centroids[label].reshape((M, M, 3)).astype('uint8')
            centroid_image = Image.fromarray(centroid_patch)
            x = i * M
            y = j * M
            out_image.paste(centroid_image, (x, y))
    img = ToTensor()(out_image).permute(1, 2, 0).cpu()
    img *= 255
    # 返回输出图像
    return img
