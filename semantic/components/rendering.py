import torch

from framework.components.rendering import BaseRenderer
from semantic.models.rs_semantic import (
    inference as rs_semantic_inference,
)

from framework.components.rays import ray_component_fn, extras_component_fn
from framework.logger import logger


class RSSemanticRendering(BaseRenderer):

    def __init__(self, cfgs, inference=rs_semantic_inference):
        super().__init__(cfgs)
        self.inference_func = inference

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
        rays_fars = ray_component_fn(rays, "fars")
        sun_d = extras_component_fn(extras, "sun_d")
        ts = (
            extras_component_fn(extras, "ts")
            .squeeze()
            .type(torch.LongTensor)
            .to(sun_d.device)
        )

        rays_t = models["t"](ts) if ts is not None else None
        rays_t_s = (
            models["t_s"](ts) if ts is not None and "t_s" in models.keys() else None
        )

        result = self.inference_func(
            models[typ],
            self.cfgs,
            xyz,
            z_vals,
            rays_d=None,
            sun_d=sun_d,
            rays_t=rays_t,
            rays_t_s=rays_t_s,
            epoch=epoch,
            render_options=render_options,
        )
        if self.cfgs.pipeline.sc_lambda > 0:
            # solar correction
            xyz_coarse = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(
                2
            )  # (N_rays, N_samples, 3)
            result_tmp = self.inference_func(
                models[typ],
                self.cfgs,
                xyz_coarse,
                z_vals,
                rays_d=None,
                sun_d=sun_d,
                rays_t=rays_t,
                rays_t_s=rays_t_s,
                epoch=epoch,
                render_options={},
            )
            result["weights_sc"] = result_tmp["weights"]
            result["transparency_sc"] = result_tmp["transparency"]
            result["sun_sc"] = result_tmp["sun"]

        return result
