from diffusers import DDIMScheduler, DDPMScheduler


def build_train_scheduler(num_train_timesteps: int = 1000) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )


def build_ddim_scheduler(num_train_timesteps: int = 1000) -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False,
    )


def build_ddpm_scheduler(num_train_timesteps: int = 1000) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False,
    )
