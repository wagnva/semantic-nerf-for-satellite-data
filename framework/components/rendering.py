import abc

import torch
from framework.components.rays import ray_component_fn
import torch.nn.functional as torch_F


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Args:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Returns:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = (
        1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    )
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
        bins_g[..., 1] - bins_g[..., 0]
    )
    return samples


def sample_dists_from_pdf_sdf(bin, weights, intvs_fine):
    """Sample points on ray shooting from pixels using the weights from the coarse NeRF.
    Args:
        bin (tensor [batch_size, num_rays, intvs]): bins of distance values from the coarse NeRF.
        weights (tensor [batch_size, num_rays, intvs]): weights from the coarse NeRF.
        intvs_fine: (int): Number of fine-grained points sampled on a ray.
    Returns:
        dists (tensor [batch_size, num_ray, intvs, 1]): Sampled distance for all rays in a batch.
    """
    pdf = torch_F.normalize(weights, p=1, dim=-1)
    # Get CDF from PDF (along last dimension).
    cdf = pdf.cumsum(dim=-1)  # [B,R,N]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [B,R,N+1]
    # Take uniform samples.
    grid = torch.linspace(0, 1, intvs_fine + 1, device=pdf.device)  # [Nf+1]
    unif = 0.5 * (grid[:-1] + grid[1:]).repeat(*cdf.shape[:-1], 1)  # [B,R,Nf]
    idx = torch.searchsorted(cdf, unif, right=True)  # [B,R,Nf] \in {1...N}
    # Inverse transform sampling from CDF.
    low = (idx - 1).clamp(min=0)  # [B,R,Nf]
    high = idx.clamp(max=cdf.shape[-1] - 1)  # [B,R,Nf]
    dist_min = bin.gather(dim=2, index=low)  # [B,R,Nf]
    dist_max = bin.gather(dim=2, index=high)  # [B,R,Nf]
    cdf_low = cdf.gather(dim=2, index=low)  # [B,R,Nf]
    cdf_high = cdf.gather(dim=2, index=high)  # [B,R,Nf]
    # Linear interpolation.
    t = (unif - cdf_low) / (cdf_high - cdf_low + 1e-8)  # [B,R,Nf]
    dists = dist_min + t * (dist_max - dist_min)  # [B,R,Nf]
    return dists[..., None]  # [B,R,Nf,1]


def sample_rays(rays, n_samples, use_disp=False, perturb=1.0, given_z_vals=None):
    # get rays
    rays_o = ray_component_fn(rays, "origins")
    rays_d = ray_component_fn(rays, "directions")
    near = ray_component_fn(rays, "near")
    far = ray_component_fn(rays, "far")

    if given_z_vals is not None:
        z_vals = given_z_vals
    else:
        # sample depths for coarse model
        z_steps = torch.linspace(0, 1, n_samples, device=rays_o.device)
        if not use_disp:  # use linear sampling in depth space
            z_vals = near * (1 - z_steps) + far * z_steps
        else:  # use linear sampling in disparity space
            z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

        if perturb > 0:  # perturb sampling depths (z_vals)
            z_vals_mid = 0.5 * (
                z_vals[:, :-1] + z_vals[:, 1:]
            )  # (N_rays, N_samples-1) interval mid points
            # get intervals between samples
            upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
            lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

            perturb_rand = perturb * torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * perturb_rand

    # discretize rays into a set of 3d points (N_rays, N_samples_, 3), one point for each depth of each ray
    xyz_coarse = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(
        2
    )  # (N_rays, N_samples, 3)
    return xyz_coarse, z_vals


class BaseRenderer:
    def __init__(self, cfgs: dict) -> None:
        super().__init__()
        self.cfgs = cfgs
        self.N_samples = cfgs.pipeline.n_samples

    def render_rays(
        self,
        models: dict,
        rays: torch.tensor,
        extras: torch.tensor,
        epoch=None,
        progress=1.0,
        render_options={},
    ):
        rays_d = ray_component_fn(rays, "directions")
        xyz_coarse, z_vals = sample_rays(rays, self.N_samples)

        model_results = self._model_rendering(
            models,
            "coarse",
            self.cfgs,
            rays,
            extras,
            xyz_coarse,
            z_vals,
            rays_d,
            epoch=epoch,
            progress=progress,
            render_options=render_options,
        )

        # append coarse postfix to results
        # used to potentially differentiate if coarse/fine models are used (like in og. NeRF paper)
        results = {}
        for k in model_results.keys():
            results[f"{k}_coarse"] = model_results[k]

        return results

    @abc.abstractmethod
    def _model_rendering(
        self,
        models: dict,
        typ: str,
        cfgs: dict,
        rays: torch.tensor,
        extras: torch.tensor,
        xyz: torch.tensor,
        z_vals: torch.tensor,
        rays_d: torch.tensor,
        epoch=None,
        progress=1.0,
        render_options=None,
    ) -> dict:
        pass
