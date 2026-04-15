import torch

@torch.no_grad()
def meanflow_sampler(
    model, 
    latents, 
    encoder_hidden_states,
    y=None, 
    cfg_scale=1.0,
    num_steps=5, 
    **kwargs
):
    """
    MeanFlow sampler supporting both single-step and multi-step generation
    
    Based on Eq.(12): z_r = z_t - (t-r)u(z_t, r, t)
    For single-step: z_0 = z_1 - u(z_1, 0, 1)
    For multi-step: iteratively apply the Eq.(12) with intermediate steps
    """
    batch_size = latents.shape[0]
    device = latents.device
    
    
    if num_steps == 1:
        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)

        z_combined = torch.cat([latents, latents], dim=0)
        r_combined = torch.cat([r, r], dim=0)
        t_combined = torch.cat([t, t], dim=0)

        u = model(
            hidden_states=z_combined.to(device=model.device,dtype=torch.bfloat16),
            encoder_hidden_states=encoder_hidden_states.to(device=model.device,dtype=torch.bfloat16),
            timestep_r=r_combined.to(device=model.device,dtype=torch.bfloat16),
            timestep_t=t_combined.to(device=model.device,dtype=torch.bfloat16),
            encoder_attention_mask=None
        ).sample

        noise_pred_uncond, noise_pred= u.chunk(2)

        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)
        
        x0 = latents - noise_pred
        
    else:
        z = latents
        
        time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
        
        for i in range(num_steps):
            print('multi_step')
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]
            
            t = torch.full((batch_size,), t_cur, device=device)
            r = torch.full((batch_size,), t_next, device=device)

            z_combined = torch.cat([z, z], dim=0)
            r_combined = torch.cat([r, r], dim=0)
            t_combined = torch.cat([t, t], dim=0)
            
            u = model(
                    hidden_states=z_combined.to(device=model.device,dtype=torch.bfloat16),
                    encoder_hidden_states=encoder_hidden_states.to(device=model.device,dtype=torch.bfloat16),
                    timestep_r=r_combined.to(device=model.device,dtype=torch.bfloat16),
                    timestep_t=t_combined.to(device=model.device,dtype=torch.bfloat16),
                    encoder_attention_mask=None
                ).sample
            noise_pred_uncond, noise_pred= u.chunk(2)

            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)
            z = z - (t_cur - t_next) * noise_pred
        
        x0 = z
    
    return x0
