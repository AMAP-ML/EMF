import torch
import numpy as np

class SILoss_adaptive:
    def __init__(
            self,
            path_type="linear",
            weighting="adaptive",
            time_sampler="logit_normal",
            time_mu_start=-0.5,
            time_mu_end=0.5,
            time_sigma_start=1.0,
            time_sigma_end=2.0,
            # ratio schedule: from fewer r!=t early -> more r!=t late
            ratio_r_not_equal_t_start=0.25,
            ratio_r_not_equal_t_end=0.75,
            schedule_type="linear",   # "linear" or "cos"
            ratio_r_not_equal_t=0.25, # kept for backward compat if user ignores schedule
            adaptive_p=1.0,
        ):
        print("ratio_r_not_equal_t (init)", ratio_r_not_equal_t, 
              "schedule start->end:", ratio_r_not_equal_t_start, "->", ratio_r_not_equal_t_end)
        self.weighting = weighting
        self.path_type = path_type

        # Time sampling config (start/end)
        self.time_sampler = time_sampler
        self.time_mu_start = time_mu_start
        self.time_mu_end = time_mu_end
        self.time_sigma_start = time_sigma_start
        self.time_sigma_end = time_sigma_end

        # Ratio schedule config
        self.ratio_r_not_equal_t_start = ratio_r_not_equal_t_start
        self.ratio_r_not_equal_t_end = ratio_r_not_equal_t_end
        # fallback scalar (old param) if not using schedule
        self.ratio_r_not_equal_t = ratio_r_not_equal_t

        self.schedule_type = schedule_type
        self.adaptive_p = adaptive_p

    def _interp_schedule(self, progress):
        """Interpolate mu, sigma and ratio given training progress in [0,1]."""
        p = float(progress)
        p = max(0.0, min(1.0, p))
        if self.schedule_type == "linear":
            mu = self.time_mu_start + (self.time_mu_end - self.time_mu_start) * p
            sigma = self.time_sigma_start + (self.time_sigma_end - self.time_sigma_start) * p
            ratio = self.ratio_r_not_equal_t_start + (self.ratio_r_not_equal_t_end - self.ratio_r_not_equal_t_start) * p
        elif self.schedule_type == "cos":
            cos_p = 0.5 * (1 - np.cos(np.pi * p))
            mu = self.time_mu_start + (self.time_mu_end - self.time_mu_start) * cos_p
            sigma = self.time_sigma_start + (self.time_sigma_end - self.time_sigma_start) * cos_p
            ratio = self.ratio_r_not_equal_t_start + (self.ratio_r_not_equal_t_end - self.ratio_r_not_equal_t_start) * cos_p
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")
        return mu, sigma, ratio

    def interpolant(self, t):
        """Define interpolation function"""
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1.0 * torch.ones_like(t)
            d_sigma_t =  1.0 * torch.ones_like(t)
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * torch.pi / 2)
            sigma_t = torch.sin(t * torch.pi / 2)
            d_alpha_t = -torch.pi / 2 * torch.sin(t * torch.pi / 2)
            d_sigma_t =  torch.pi / 2 * torch.cos(t * torch.pi / 2)
        else:
            raise NotImplementedError()
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def sample_time_steps(self, batch_size, device, progress=None, current_step=None, total_steps=None):
        """
        Sample time steps (r, t).
        progress: float in [0,1] OR current_step/total_steps pair.
        If none provided, use start values (backward compatible).
        Returns r, t (each shape (batch,))
        """
        # compute progress if needed
        if progress is None and current_step is not None and total_steps is not None:
            progress = float(current_step) / float(max(1, total_steps))
        if progress is None:
            mu = self.time_mu_start
            sigma = self.time_sigma_start
            ratio = self.ratio_r_not_equal_t  # fallback scalar
        else:
            mu, sigma, ratio = self._interp_schedule(progress)

        # Step1: Sample two time points
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * float(sigma) + float(mu)
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")

        # Step2: Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]

        # Step3: Control the proportion of r=t samples
        # ratio is proportion of samples where r != t
        ratio = float(ratio)
        fraction_equal = 1.0 - ratio  # e.g., ratio=0.2 -> 80% r==t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        r = torch.where(equal_mask, t, r)

        return r, t

    def __call__(self, model, images, encoder_hidden_states, model_kwargs=None,
                 training_progress=None, current_step=None, total_steps=None):
        """
        Compute MeanFlow loss function with schedule support.
        Provide either training_progress (0..1) or current_step & total_steps.
        """
        if model_kwargs is None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()

        batch_size = images.shape[0]
        device = images.device

        # Sample time steps, using schedule info if provided
        r, t = self.sample_time_steps(batch_size, device,
                                      progress=training_progress,
                                      current_step=current_step,
                                      total_steps=total_steps)

        noises = torch.randn_like(images)

        # Calculate interpolation and z_t (ensure shapes)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises
        v_t = d_alpha_t * images + d_sigma_t * noises
        time_diff = (t - r).view(-1, 1, 1, 1)

        # call model (keeps behavior same as original; user model must accept these args)
        model = model.to(device=images.device, dtype=torch.bfloat16)

        u = model(
                hidden_states=z_t.to(device=images.device, dtype=torch.bfloat16),
                timestep_t=t.to(device=images.device, dtype=torch.bfloat16),
                timestep_r=r.to(device=images.device, dtype=torch.bfloat16),
                encoder_hidden_states=(encoder_hidden_states.to(device=images.device, dtype=torch.bfloat16)
                                       if encoder_hidden_states is not None else None),
                encoder_attention_mask=None,
            ).sample

        # JVP part (kept as in original code)
        primals = (z_t, r, t)
        tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))

        primals = tuple(p.to(device=images.device, dtype=torch.float32) for p in primals)
        tangents = tuple(tg.to(device=images.device, dtype=torch.float32) for tg in tangents)

        model = model.to(device=images.device, dtype=torch.float32)
        def fn_current(z, cur_r, cur_t):
            return model(
                hidden_states=z.to(device=images.device, dtype=torch.float32),
                timestep_t=cur_t.to(device=images.device, dtype=torch.float32),
                timestep_r=cur_r.to(device=images.device, dtype=torch.float32),
                encoder_hidden_states=(encoder_hidden_states.to(device=images.device, dtype=torch.float32)
                                       if encoder_hidden_states is not None else None),
                encoder_attention_mask=None,
            ).sample

        # Use torch.func.jvp if available; else ensure you have a fallback
        try:
            _, dudt = torch.func.jvp(fn_current, primals, tangents)
        except Exception as e:
            raise RuntimeError("torch.func.jvp failed or unavailable. "
                               "Please ensure your PyTorch supports jvp or add a fallback. Orig err: " + str(e))

        model = model.to(device=images.device, dtype=torch.bfloat16)
        u_target = v_t - time_diff * dudt

        # Detach the target to prevent gradient flow
        error = u - u_target.detach()
        loss_mid = torch.sum((error**2).reshape(error.shape[0], -1), dim=-1)

        if self.weighting == "adaptive":
            weights = 1.0 / (loss_mid.detach() + 1e-3).pow(self.adaptive_p)
            loss = weights * loss_mid
        else:
            loss = loss_mid
        loss_mean_ref = torch.mean((error**2))
        return loss, loss_mean_ref
