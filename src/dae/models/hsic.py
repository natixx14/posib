import torch


def pairwise_distances(x):
    d_batch, d_x = x.shape
    d = x.reshape(1, d_batch, d_x).repeat_interleave(
        d_batch, dim=0) - x.reshape(d_batch, 1, d_x).repeat_interleave(d_batch,
                                                                       dim=1)
    return torch.linalg.vector_norm(d, dim=2)**2


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    s = torch.median(pairwise_distances_).item()
    return torch.exp(-pairwise_distances_ / s)


def HSIC(x, y, s_x=1, s_y=1, device='cpu'):
    m, _ = x.shape  #batch size
    K = GaussianKernelMatrix(x, s_x).to(device)
    L = GaussianKernelMatrix(y, s_y).to(device)
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.to(device)
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1)**2)
    return HSIC
