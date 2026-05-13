from typing import Iterator

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers import UNet2DModel
from tqdm import tqdm


@torch.no_grad()
def sample_with_cfg(
    unet: torch.nn.Module,
    scheduler: DDIMScheduler | DDPMScheduler,
    labels: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float = 3.0,
    device: str = "cuda",
    generator: torch.Generator | None = None,
    image_size: int = 64,
) -> torch.Tensor:
    unet.eval()
    batch = labels.size(0)
    labels = labels.to(device)
    null_labels = torch.zeros_like(labels)

    scheduler.set_timesteps(num_inference_steps)
    latents = torch.randn(
        batch, 3, image_size, image_size, device=device, generator=generator
    )

    for t in tqdm(scheduler.timesteps, desc="sample", leave=False):
        latent_in = torch.cat([latents, latents], dim=0)
        cls_in = torch.cat([null_labels, labels], dim=0)
        eps = unet(latent_in, t, cls_in)
        eps_uncond, eps_cond = eps.chunk(2, dim=0)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        latents = scheduler.step(eps, t, latents).prev_sample

    return latents


def _predict_x0(
    sample: torch.Tensor,
    eps: torch.Tensor,
    scheduler: DDIMScheduler | DDPMScheduler,
    timestep: torch.Tensor,
) -> torch.Tensor:
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(sample.device)
    while alpha_prod_t.ndim < sample.ndim:
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
    beta_prod_t = 1 - alpha_prod_t
    return (sample - beta_prod_t.sqrt() * eps) / alpha_prod_t.sqrt()


def _classifier_guidance_grad(
    classifier: torch.nn.Module,
    x0: torch.Tensor,
    labels: torch.Tensor,
    positive_only: bool,
) -> torch.Tensor:
    with torch.enable_grad():
        x0 = x0.detach().requires_grad_(True)
        probs = classifier(x0.clamp(-1, 1)).clamp(1e-6, 1 - 1e-6)
        if positive_only:
            score = (labels * probs.log()).sum(dim=1).mean()
        else:
            score = -F.binary_cross_entropy(probs, labels, reduction="mean")
        return torch.autograd.grad(score, x0)[0].detach()


def sample_with_cfg_and_classifier_guidance(
    unet: torch.nn.Module,
    scheduler: DDIMScheduler | DDPMScheduler,
    labels: torch.Tensor,
    classifier: torch.nn.Module,
    num_inference_steps: int,
    guidance_scale: float = 3.0,
    classifier_guidance_scale: float = 0.0,
    classifier_guidance_start: float = 0.0,
    classifier_guidance_stop: float = 0.9,
    classifier_positive_only: bool = True,
    device: str = "cuda",
    generator: torch.Generator | None = None,
    image_size: int = 64,
) -> torch.Tensor:
    unet.eval()
    classifier.eval()
    batch = labels.size(0)
    labels = labels.to(device)
    null_labels = torch.zeros_like(labels)

    scheduler.set_timesteps(num_inference_steps)
    latents = torch.randn(
        batch, 3, image_size, image_size, device=device, generator=generator
    )
    guidance_first = int(num_inference_steps * classifier_guidance_start)
    guidance_last = int(num_inference_steps * classifier_guidance_stop)

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="sample", leave=False)):
        with torch.no_grad():
            latent_in = torch.cat([latents, latents], dim=0)
            cls_in = torch.cat([null_labels, labels], dim=0)
            eps_pred = unet(latent_in, t, cls_in)
            eps_uncond, eps_cond = eps_pred.chunk(2, dim=0)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        if classifier_guidance_scale != 0 and guidance_first <= i <= guidance_last:
            x0 = _predict_x0(latents, eps, scheduler, t)
            grad = _classifier_guidance_grad(
                classifier, x0, labels, classifier_positive_only
            )
            eps = eps - classifier_guidance_scale * grad

        latents = scheduler.step(eps, t, latents).prev_sample.detach()

    return latents


@torch.no_grad()
def sample_with_trace(
    unet: torch.nn.Module,
    scheduler: DDIMScheduler | DDPMScheduler,
    labels: torch.Tensor,
    num_inference_steps: int,
    num_snapshots: int = 8,
    guidance_scale: float = 3.0,
    device: str = "cuda",
    generator: torch.Generator | None = None,
    image_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    unet.eval()
    batch = labels.size(0)
    labels = labels.to(device)
    null_labels = torch.zeros_like(labels)

    scheduler.set_timesteps(num_inference_steps)
    latents = torch.randn(
        batch, 3, image_size, image_size, device=device, generator=generator
    )

    n_steps = len(scheduler.timesteps)
    snapshot_ids = set(
        int(round(i)) for i in torch.linspace(0, n_steps - 1, num_snapshots).tolist()
    )
    snapshots: list[torch.Tensor] = []

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="trace", leave=False)):
        latent_in = torch.cat([latents, latents], dim=0)
        cls_in = torch.cat([null_labels, labels], dim=0)
        eps = unet(latent_in, t, cls_in)
        eps_uncond, eps_cond = eps.chunk(2, dim=0)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        latents = scheduler.step(eps, t, latents).prev_sample
        if i in snapshot_ids:
            snapshots.append(latents.clone())

    trace = torch.stack(snapshots, dim=0)
    return latents, trace
