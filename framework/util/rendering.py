import torch


def convert_sigmas(sigmas, z_vals):
    """
    Convert sigmas to the weights by calculating opacity and transmittance
    :param sigmas: output of the network. [N_rays, N_samples]
    :param z_vals: position along the ray for each sample. [N_rays, N_samples]
    :return: sample_weights [N_rays, N_Samples], estimated depth [N_rays], transparency [N_rays, N_Samples]
    """
    # define deltas, i.e. the length between the points in which the ray is discretized
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples-1)
    delta_inf = 1e10 * torch.ones_like(
        deltas[:, :1]
    )  # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples)

    # compute alpha as in the formula (3) of the nerf paper
    # noise set to 0 in satnerf paper
    # noise_std = cfgs["state"]["noise_std"]
    # noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
    # alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples)

    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (N_rays, N_samples)
    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
    )  # [1, a1, a2, ...]
    transparency = torch.cumprod(alphas_shifted, -1)[:, :-1]  # T in the paper
    weights = alphas * transparency  # (N_rays, N_samples)
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)

    return weights, depth_final, transparency, alphas
