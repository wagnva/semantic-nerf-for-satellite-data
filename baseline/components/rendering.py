import torch

from framework.components.rendering import BaseRenderer
from baseline.models.nerf import inference as nerf_inference
from baseline.models.snerf import inference as snerf_inference
from baseline.models.satnerf import inference as satnerf_inference

from framework.components.rays import ray_component_fn, extras_component_fn
from framework.logger import logger


class SatNeRFRendering(BaseRenderer):
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
        render_options={},
    ) -> dict:
        rays_o = ray_component_fn(rays, "origin")
        sun_d = extras_component_fn(extras, "sun_d")
        ts = (
            extras_component_fn(extras, "ts")
            .squeeze()
            .type(torch.LongTensor)
            .to(sun_d.device)
        )

        rays_t = models["t"](ts) if ts is not None else None

        result = satnerf_inference(
            models[typ],
            self.cfgs,
            xyz,
            z_vals,
            rays_d=None,
            sun_d=sun_d,
            rays_t=rays_t,
            epoch=epoch,
        )
        if self.cfgs.pipeline.sc_lambda > 0:
            # solar correction
            xyz_coarse = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(
                2
            )  # (N_rays, N_samples, 3)
            result_tmp = satnerf_inference(
                models[typ],
                self.cfgs,
                xyz_coarse,
                z_vals,
                rays_d=None,
                sun_d=sun_d,
                rays_t=rays_t,
                epoch=epoch,
            )
            result["weights_sc"] = result_tmp["weights"]
            result["transparency_sc"] = result_tmp["transparency"]
            result["sun_sc"] = result_tmp["sun"]

        return result


class SNeRFRendering(BaseRenderer):
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
        render_options={},
    ) -> dict:
        rays_o = ray_component_fn(rays, "origin")
        sun_d = extras_component_fn(extras, "sun_directions")
        # render using main set of rays
        result = snerf_inference(models[typ], cfgs, xyz, z_vals, rays_d=None, sun_d=sun_d)
        if cfgs.pipeline.sc_lambda > 0:
            # solar correction
            xyz_sun = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(
                2
            )  # (N_rays, N_samples, 3)
            result_ = snerf_inference(
                models[typ], cfgs, xyz_sun, z_vals, rays_d=None, sun_d=sun_d
            )
            result["weights_sc"] = result_["weights"]
            result["transparency_sc"] = result_["transparency"]
            result["sun_sc"] = result_["sun"]
        return result


class NeRFRendering(BaseRenderer):
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
        render_options={},
    ) -> dict:
        return nerf_inference(models[typ], cfgs, xyz, z_vals, rays_d=rays_d, epoch=epoch)
