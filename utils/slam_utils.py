import torch
import wandb
from utils.logging_utils import Log
from torch.nn import functional as F


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(
    config, image, decoded_semantics, depth, opacity, viewpoint, initialization=False
):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    else:
        if config["Training"]["semantic"]:
            return get_loss_tracking_semantic_rgbd(
                config, image, decoded_semantics, depth, opacity, viewpoint
            )
        else:
            return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    """
    Note:
        1. Use opacity as weight
        2. Use rgb_boundary_threshold to filter black margins
    """
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_semantics(config, segmentation_logits, depth, opacity, viewpoint):
    segmentation_label = viewpoint.segmentation_label.cuda()
    loss_segmentation_label = F.cross_entropy(
        segmentation_logits,
        segmentation_label.view(1, -1),
    )
    return loss_segmentation_label


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = (
        config["Training"]["tracking_alpha"]
        if "tracking_alpha" in config["Training"]
        else 0.95
    )

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_tracking_semantic_rgbd(
    config,
    image,
    segmentation_logits,
    depth,
    opacity,
    viewpoint,
    initialization=False,
    log=True,
):
    alpha = (
        config["Training"]["tracking_alpha"]
        if "tracking_alpha" in config["Training"]
        else 0.9
    )
    beta = (
        config["Training"]["tracking_beta"]
        if "tracking_beta" in config["Training"]
        else 0.05
    )
    gamma = (
        config["Training"]["tracking_gamma"]
        if "tracking_gamma" in config["Training"]
        else 0.05
    )
    if log:
        try:
            get_loss_tracking_semantic_rgbd.idx += 1
        except AttributeError:
            get_loss_tracking_semantic_rgbd.idx = 0
            Log(
                f"loss weights, appearance={alpha}, depth={beta}, semantics={gamma}",
                tag="Track",
            )

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    loss_appearance = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    loss_semantics = get_loss_tracking_semantics(
        config=config,
        segmentation_logits=segmentation_logits,
        depth=depth,
        opacity=opacity,
        viewpoint=viewpoint,
    )
    depth_mask = depth_pixel_mask * opacity_mask
    loss_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask).mean()
    loss = alpha * loss_appearance + beta * loss_depth + gamma * loss_semantics

    return loss


def get_loss_mapping(
    config,
    image,
    semantics,
    segmentation_logits,
    depth,
    viewpoint,
    opacity,
    initialization=False,
):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    loss = 0
    if config["Training"]["monocular"]:
        loss = get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    else:
        if not (
            config["Training"].get("semantic")
            and config["Training"]["semantic"] is True
        ):
            loss = get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)
        else:
            loss = get_loss_mapping_semantic_rgbd(
                config=config,
                image=image_ab,
                semantics=semantics,
                segmentation_logits=segmentation_logits,
                depth=depth,
                viewpoint=viewpoint,
            )

    return loss


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = (
        config["Training"]["mapping_alpha"]
        if "mapping_alpha" in config["Training"]
        else 0.95
    )
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_loss_mapping_semantic_rgbd(
    config,
    image,
    depth,
    viewpoint,
    semantics,
    segmentation_logits,
    segmentation_uncertainty_mask=None,
    log=False,
):
    alpha = (
        config["Training"]["mapping_alpha"]
        if "mapping_alpha" in config["Training"]
        else 0.9
    )
    beta = (
        config["Training"]["mapping_beta"]
        if "mapping_beta" in config["Training"]
        else 0.05
    )
    gamma = (
        config["Training"]["mapping_gamma"]
        if "mapping_gamma" in config["Training"]
        else 0.05
    )

    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    # TODO: Does exposure affect segmentation?
    gt_image = viewpoint.original_image.cuda()
    # gt_segmentation_map = viewpoint.segmentation_map.cuda()
    gt_segmentation_label = viewpoint.segmentation_label.cuda()
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    # TODO: semantic_uncertainty_mask

    loss_appearance = torch.abs(
        image * rgb_pixel_mask - gt_image * rgb_pixel_mask
    ).mean()

    loss_depth = torch.abs(
        depth * depth_pixel_mask - gt_depth * depth_pixel_mask
    ).mean()

    loss_segmentation = F.cross_entropy(
        segmentation_logits,
        gt_segmentation_label.view(1, -1),
    )

    if log:
        try:
            get_loss_mapping_semantic_rgbd.idx += 1
            if config["Results"]["use_wandb"]:
                wandb.log(
                    {
                        "Mapping/Segmentation Loss": loss_segmentation,
                        "Mapping/Depth Loss": loss_depth,
                        "Mapping/Apperance Loss": loss_appearance,
                    },
                )
            else:
                Log(
                    f"step={get_loss_mapping_semantic_rgbd.idx}, loss_appearance={loss_appearance}, loss_segmentation={loss_segmentation}, loss_depth={loss_depth}",
                    tag="Map",
                )
        except AttributeError:
            get_loss_mapping_semantic_rgbd.idx = 0
            Log(
                f"loss weights, appearance={alpha}, depth={beta}, semantics={gamma}",
                tag="Map",
            )

    return alpha * loss_appearance + beta * loss_depth + gamma * loss_segmentation


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    # TODO:
    # Compensate the noise from real-world depth estimates.
    # Example:
    #   1. Depth Uncertainty Model
    #   2. A maximum threshold (the range of the ToF sensor...)
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
