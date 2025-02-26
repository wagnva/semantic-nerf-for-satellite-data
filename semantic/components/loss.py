import torch

from baseline.components.loss import solar_correction


def uncertainty_aware_semantic_loss(
    cross_entropy_loss,
    inputs,
    targets,
    typ="coarse",
    beta_min=0.05,
    ignore_mask=None,
    detach_gradient=False,
):
    beta_input = inputs.get("beta_semantic_coarse", inputs["beta_coarse"])

    if detach_gradient:
        beta_input = beta_input.detach().clone()

    beta = torch.sum(inputs[f"weights_{typ}"].unsqueeze(-1) * beta_input, -2) + beta_min
    loss_term = torch.mean(
        cross_entropy_loss(
            inputs[f"semantic_logits_{typ}"][ignore_mask], targets[ignore_mask].squeeze()
        )
    )
    loss_dict = {f"{typ}_semantic": (loss_term / (2 * beta**2)).mean()}
    if "beta_semantic_coarse" in inputs:
        # only apply second logbeta loss if a seperate semantic beta is used
        # if color beta is used, this would lead to this loss being applied twice
        loss_dict[f"{typ}_semantic_logbeta"] = (3 + torch.log(beta).mean()) / 2

    return loss_dict


class SemanticLoss(torch.nn.Module):

    def __init__(self, lambda_s, car_index, ignore_car_index=False):
        super().__init__()
        self.lambda_s = lambda_s
        if not ignore_car_index:
            car_index = -100
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=car_index)

    def forward(self, inputs, targets, ignore_mask=None):
        loss_dict = {}
        typ = "coarse"
        ignore_mask = (
            torch.ones(targets.shape[0], dtype=torch.bool)
            if ignore_mask is None
            else ignore_mask
        )
        loss_dict[f"{typ}_semantic"] = self.loss(
            inputs[f"semantic_logits_{typ}"][ignore_mask], targets[ignore_mask].squeeze()
        )
        if "semantic_logits_fine" in inputs:
            typ = "fine"
            loss_dict[f"{typ}_semantic"] = self.loss(
                inputs[f"semantic_logits_{typ}"][ignore_mask],
                targets[ignore_mask].squeeze(),
            )
        # apply weights
        for k in loss_dict.keys():
            loss_dict[k] = self.lambda_s * torch.mean(loss_dict[k])
        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict


class SemanticUncertaintyLoss(torch.nn.Module):

    def __init__(
        self, lambda_s, car_index, detach_beta_for_s=False, ignore_car_index=False
    ):
        super().__init__()
        self.lambda_s = lambda_s
        if not ignore_car_index:
            car_index = -100
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=car_index)
        self.detach_beta_for_s = detach_beta_for_s

    def forward(self, inputs, targets, ignore_mask=None):
        loss_dict = {}
        typ = "coarse"
        ignore_mask = (
            torch.ones(targets.shape[0], dtype=torch.bool)
            if ignore_mask is None
            else ignore_mask
        )
        loss_dict.update(
            uncertainty_aware_semantic_loss(
                self.cross_entropy,
                inputs,
                targets.squeeze(),
                typ,
                ignore_mask=ignore_mask,
                detach_gradient=self.detach_beta_for_s,
            )
        )
        if "semantic_logits_fine" in inputs:
            typ = "fine"
            loss_dict.update(
                uncertainty_aware_semantic_loss(
                    self.cross_entropy,
                    inputs,
                    targets.squeeze(),
                    typ,
                    ignore_mask=ignore_mask,
                    detach_gradient=self.detach_beta_for_s,
                )
            )
        # apply weight
        for k in loss_dict.keys():
            loss_dict[k] *= self.lambda_s
        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict


class SemanticCarRegLoss(torch.nn.Module):

    def __init__(self, lambda_c, car_label):
        super().__init__()
        self.lambda_c = lambda_c
        self.loss = torch.nn.MSELoss()
        self.car_label = car_label

    def forward(self, inputs, targets, ignore_mask=None):
        loss_dict = {}
        typ = "coarse"

        ignore_mask = (
            torch.ones(targets.shape[0], dtype=torch.bool)
            if ignore_mask is None
            else ignore_mask
        )

        # sum up uncertainty along rays
        uncertainty = torch.sum(
            inputs[f"weights_{typ}"].unsqueeze(-1) * inputs[f"beta_{typ}"], -2
        )  # (N_rays, 1)
        car_mask = targets == self.car_label  # (N_rays, 1)
        # car_mask = car_mask.to(torch.int)
        # make sure ignore_mask is applied
        car_mask = torch.logical_and(
            car_mask.squeeze(), ignore_mask.squeeze()
        )  # (N_rays)

        # this contains all non-ignored rays with a GT car label
        uncertainty_loss_input = uncertainty[car_mask]

        loss_dict[f"{typ}_car_reg_loss"] = self.loss(
            torch.ones_like(uncertainty_loss_input), uncertainty_loss_input
        )

        # apply weight
        for k in loss_dict.keys():
            loss_dict[k] *= self.lambda_c
        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict
