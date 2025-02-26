import torch

from framework.logger import logger
import framework.util.rendering as rendering_utils
from baseline.models.commons import Mapping, Siren, sine_init, first_layer_sine_init


def inference(
    model,
    cfgs,
    rays_xyz: torch.tensor,
    z_vals: torch.tensor,
    rays_d=None,
    sun_d=None,
    rays_t=None,
    epoch=None,
):
    """
    Runs sat-nerf model using a batch of input rays

    :param model: nerf model
    :param cfgs: configs
    :param rays_xyz: (N_rays, N_samples_, 3) xyz points of the rays to be sampled
                        N_samples_ is the number of sampled points in each ray;
                            = N_samples for coarse model
                            = N_samples+N_importance for fine model
    :param z_vals: (N_rays, N_samples_) depths of the sampled positions
    :param rays_d: (N_rays, 3) direction vectors of the rays
    :param sun_d: (N_rays, 3) sun direction vectors
    :param rays_t: (N_rays, t_embedding_tau): embedded time stamps
    :param epoch: current epoch
    :return: dictionary containing the accumulated results
    """
    N_rays = rays_xyz.shape[0]
    N_samples = rays_xyz.shape[1]
    xyz_ = rays_xyz.view(-1, 3)  # (N_rays*N_samples, 3)

    # check if there are additional inputs, which are used or not depending on the nerf variant
    rays_d_ = (
        None
        if rays_d is None
        else torch.repeat_interleave(rays_d, repeats=N_samples, dim=0)
    )
    sun_d_ = (
        None
        if sun_d is None
        else torch.repeat_interleave(sun_d, repeats=N_samples, dim=0)
    )
    rays_t_ = (
        None
        if rays_t is None
        else torch.repeat_interleave(rays_t, repeats=N_samples, dim=0)
    )

    # the input batch is split in chunks to avoid possible problems with memory usage
    chunk = cfgs.pipeline.render_chunk_size
    batch_size = xyz_.shape[0]

    # run model
    out_chunks = []
    for i in range(0, batch_size, chunk):
        out_chunks += [
            model(
                xyz_[i : i + chunk],
                input_dir=None if rays_d_ is None else rays_d_[i : i + chunk],
                input_sun_dir=None if sun_d_ is None else sun_d_[i : i + chunk],
                input_t=None if rays_t_ is None else rays_t_[i : i + chunk],
                epoch=epoch,
            )
        ]
    out = torch.cat(out_chunks, 0)

    # retrieve outputs
    out = out.view(N_rays, N_samples, -1)
    rgbs = out[..., :3]  # (N_rays, N_samples, 3)
    sigmas = out[..., 3]  # (N_rays, N_samples)
    sun_v = out[..., 4:5]  # (N_rays, N_samples, 1)
    sky_rgb = out[..., 5:8]  # (N_rays, N_samples, 3)
    uncertainty = out[..., 8:9]  # (N_rays, N_samples, 1)

    weights, depth_final, transparency, _ = rendering_utils.convert_sigmas(sigmas, z_vals)

    irradiance = sun_v + (1 - sun_v) * sky_rgb  # equation 2 of the s-nerf paper
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs * irradiance, -2)  # (N_rays, 3)
    rgb_final = torch.clamp(rgb_final, min=0.0, max=1.0)
    result = {
        "rgb": rgb_final,
        "depth": depth_final,
        "weights": weights,
        "transparency": transparency,
        "albedo": rgbs,
        "sun": sun_v,
        "sky": sky_rgb,
        "beta": uncertainty,
        "sigmas": sigmas,
    }

    return result


class SatNeRF(torch.nn.Module):
    def __init__(
        self,
        cfgs,
        layers=8,
        feat=256,
        mapping=False,
        mapping_sizes=[10, 4],
        skips=[4],
        siren=True,
        t_embedding_dims=16,
    ):
        super(SatNeRF, self).__init__()
        self.layers = layers
        self.skips = skips
        self.t_embedding_dims = t_embedding_dims
        self.mapping = mapping
        self.input_sizes = [3, 0]
        self.rgb_padding = 0.001
        self.number_of_outputs = 9  # rgb (3) + sigma (1) + sun visibility (1) + rgb from sky color (3) + beta (1)

        # if enabled, then use the full feature set for the last layers behind the main fc
        fc_use_full_features = cfgs.pipeline.fc_use_full_features
        self.feat_last = feat if fc_use_full_features else feat // 2

        # activation function
        nl = Siren() if siren else torch.nn.ReLU()

        # use positional encoding if specified
        in_size = self.input_sizes.copy()
        if mapping:
            self.mapping = [
                Mapping(map_sz, in_sz)
                for map_sz, in_sz in zip(mapping_sizes, self.input_sizes)
            ]
            in_size = [
                2 * map_sz * in_sz
                for map_sz, in_sz in zip(mapping_sizes, self.input_sizes)
            ]
        else:
            self.mapping = [torch.nn.Identity(), torch.nn.Identity()]

        # define the main network of fully connected layers, i.e. FC_NET
        fc_layers = []
        fc_layers.append(torch.nn.Linear(in_size[0], feat))
        fc_layers.append(Siren(w0=30.0) if siren else nl)
        for i in range(1, layers):
            if i in skips:
                fc_layers.append(torch.nn.Linear(feat + in_size[0], feat))
            else:
                fc_layers.append(torch.nn.Linear(feat, feat))
            fc_layers.append(nl)
        self.fc_net = torch.nn.Sequential(
            *fc_layers
        )  # shared 8-layer structure that takes the encoded xyz vector

        # FC_NET output 1: volume density
        self.sigma_from_xyz = torch.nn.Sequential(
            torch.nn.Linear(feat, 1), torch.nn.Softplus()
        )

        # FC_NET output 2: vector of features from the spatial coordinates
        self.feats_from_xyz = torch.nn.Linear(
            feat, feat
        )  # No non-linearity here in the original paper

        # the FC_NET output 2 is concatenated to the encoded viewing direction input
        # and the resulting vector of features is used to predict the rgb color
        self.rgb_from_xyzdir = torch.nn.Sequential(
            torch.nn.Linear(feat + in_size[1], self.feat_last),
            nl,
            torch.nn.Linear(self.feat_last, 3),
            torch.nn.Sigmoid(),
        )

        # Shadow-NeRF (s-nerf) additional layers
        sun_dir_in_size = 3
        sun_v_layers = []
        sun_v_layers.append(torch.nn.Linear(feat + sun_dir_in_size, self.feat_last))
        sun_v_layers.append(Siren() if siren else nl)
        for i in range(1, 3):
            sun_v_layers.append(torch.nn.Linear(self.feat_last, self.feat_last))
            sun_v_layers.append(nl)
        sun_v_layers.append(torch.nn.Linear(self.feat_last, 1))
        sun_v_layers.append(torch.nn.Sigmoid())
        self.sun_v_net = torch.nn.Sequential(*sun_v_layers)

        self.sky_color = torch.nn.Sequential(
            torch.nn.Linear(sun_dir_in_size, self.feat_last),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feat_last, 3),
            torch.nn.Sigmoid(),
        )

        if siren:
            self.fc_net.apply(sine_init)
            self.fc_net[0].apply(first_layer_sine_init)
            self.sun_v_net.apply(sine_init)
            self.sun_v_net[0].apply(first_layer_sine_init)

        self.beta_from_xyz = torch.nn.Sequential(
            torch.nn.Linear(self.t_embedding_dims + feat, self.feat_last),
            nl,
            torch.nn.Linear(self.feat_last, 1),
            torch.nn.Softplus(),
        )

    def forward(
        self, input_xyz, input_dir=None, input_sun_dir=None, input_t=None, epoch=None
    ):
        """
        Predicts the values rgb, sigma from a batch of input rays
        the input rays are represented as a set of 3d points xyz

        Args:
            input_xyz: (B, 3) input tensor, with the 3d spatial coordinates, B is batch size

        Returns:
            out: (B, 4) first 3 columns are rgb color, last column is volume density
        """
        # compute shared features
        input_xyz = self.mapping[0](input_xyz)
        xyz_ = input_xyz
        for i in range(self.layers):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = self.fc_net[2 * i](xyz_)
            xyz_ = self.fc_net[2 * i + 1](xyz_)
        shared_features = xyz_

        # compute volume density
        sigma = self.sigma_from_xyz(shared_features)

        # compute color
        xyz_features = self.feats_from_xyz(shared_features)
        if self.input_sizes[1] > 0:
            input_xyzdir = torch.cat([xyz_features, self.mapping[1](input_dir)], -1)
        else:
            input_xyzdir = xyz_features
        rgb = self.rgb_from_xyzdir(input_xyzdir)
        # improvement suggested by Jon Barron to help stability (same paper as soft+ suggestion)
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
        out = torch.cat([rgb, sigma], 1)  # (B, 4)

        # extra outputs
        input_sun_v_net = torch.cat([xyz_features, input_sun_dir], -1)
        sun_v = self.sun_v_net(input_sun_v_net)
        sky_color = self.sky_color(input_sun_dir)
        out = torch.cat([out, sun_v, sky_color], 1)  # (B, 8)

        input_for_beta = torch.cat([xyz_features, input_t], -1)
        beta = self.beta_from_xyz(input_for_beta)
        out = torch.cat([out, beta], 1)  # (B, 12)

        return out
