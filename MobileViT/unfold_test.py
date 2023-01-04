import time
import torch

batch_size = 8
in_channels = 32
patch_h = 2
patch_w = 2
num_patch_h = 16  #高的方向上有16个
num_patch_w = 16  #宽的方向上有16个
num_patches = num_patch_h * num_patch_w  #一共有196个patches
patch_area = patch_h * patch_w  #一个patch的面积（像素）是4


def official(x: torch.Tensor):
    # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
    x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
    # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
    x = x.transpose(1, 2)
    # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
    x = x.reshape(batch_size, in_channels, num_patches, patch_area)
    # [B, C, N, P] -> [B, P, N, C]
    x = x.transpose(1, 3)
    # [B, P, N, C] -> [BP, N, C]
    x = x.reshape(batch_size * patch_area, num_patches, -1)

    return x


def my_self(x: torch.Tensor): #(8,32,32,32)
    # [B, C, H, W] -> [B, C, n_h, p_h, n_w, p_w]
    x = x.reshape(batch_size, in_channels, num_patch_h, patch_h, num_patch_w, patch_w) #(8,32,16,2,16,2)
    # [B, C, n_h, p_h, n_w, p_w] -> [B, C, n_h, n_w, p_h, p_w]
    x = x.transpose(3, 4) #(8,32,16,16,2,2)
    # [B, C, n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
    """
    P:一个patch里面一共有多少个像素 (4)
    N:一共有多少个patch (256)
    """
    x = x.reshape(batch_size, in_channels, num_patches, patch_area) #(8,32,256,4)
    # [B, C, N, P] -> [B, P, N, C]
    x = x.transpose(1, 3)  #(8,4,256,32)
    # [B, P, N, C] -> [BP, N, C]
    x = x.reshape(batch_size * patch_area, num_patches, -1)  #(32,256,32)
    return x


if __name__ == '__main__':
    t = torch.randn(batch_size, in_channels, num_patch_h * patch_h, num_patch_w * patch_w)
    print(torch.equal(official(t), my_self(t)))

    t1 = time.time()
    for _ in range(1000):
        official(t)
    print(f"official time: {time.time() - t1}")

    t1 = time.time()
    for _ in range(1000):
        my_self(t)
    print(f"self time: {time.time() - t1}")
