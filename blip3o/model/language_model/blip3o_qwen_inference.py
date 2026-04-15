from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3Model,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from blip3o.model.blip3o_arch import blip3oMetaForCausalLM, blip3oMetaModel
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers import DDPMScheduler, DDIMScheduler, LCMScheduler, FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
import numpy as np
from tqdm import tqdm
import PIL

from blip3o.model.language_model.meanflow_sampler import meanflow_sampler
def numpy_to_pil(images: np.ndarray):
    """
    Convert a NumPy array of shape (batch, height, width, channels) to a list of PIL Images.
    """
    pil_images = []
    for img in images:
        img_uint8 = (img * 255).round().astype("uint8")
        if img_uint8.shape[2] == 1:
            img_uint8 = img_uint8[..., 0]
        pil_images.append(PIL.Image.fromarray(img_uint8))
    return pil_images


class blip3oQwenConfig(Qwen3Config):
    model_type = "blip3o_qwen_inference"

class blip3oQwenModel(blip3oMetaModel, Qwen3Model):
    config_class = blip3oQwenConfig

    def __init__(self, config: Qwen3Config):
        super(blip3oQwenModel, self).__init__(config)

class blip3oQwenForInferenceLM(Qwen3ForCausalLM, blip3oMetaForCausalLM):
    config_class = blip3oQwenConfig

    def __init__(self, config):
        Qwen3ForCausalLM.__init__(self, config)
        config.model_type = "blip3o_qwen"
        config.rope_scaling = None

        self.model = blip3oQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.model.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.model.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma



    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)





    @torch.no_grad()
    def decode_latents(self, latents, normalize=True, return_tensor=False):
        if self.model.sana_vae is not None:
            print('sana_vae is not none')
            latents = latents / self.model.sana_vae.config.scaling_factor
            if "shift_factor" in self.model.sana_vae.config and self.model.sana_vae.config.shift_factor is not None:
                latents = latents + self.model.sana_vae.config.shift_factor
            samples = self.model.sana_vae.decode(latents).sample
        else:
            samples = latents
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)
        if return_tensor:
            return samples
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples



    @torch.no_grad()
    def generate_images(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        top_p: Optional[torch.Tensor] = None,
        top_k: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.5,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        **kwargs,
    ):

        gen_ids = super(blip3oQwenForInferenceLM, self).generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            attention_mask=attention_mask,
            top_p=top_p,
            top_k=top_k)

        # breakpoint()
        with torch.no_grad():
            outs = self.model(
                input_ids = gen_ids, 
                output_hidden_states = True,
                return_dict = True,
            )
        hidden_states = outs.hidden_states[-1]   


        start_pos = (gen_ids == self.config.image_start_tag_id).float().argmax(dim=1)   
        end_pos   = (gen_ids == self.config.image_end_tag_id).float().argmax(dim=1)   


        selected_hidden_states = []                       
        for b in range(hidden_states.size(0)):          
            start = start_pos[b].item() + 1         
            # end = end_pos[b].item()              
            selected_hidden_states.append(hidden_states[b, start:, :]) 
        pred_latent = torch.stack(selected_hidden_states, dim=0)
        


        img_hidden_states_null = torch.zeros_like(pred_latent)
        pred_latent = torch.cat([img_hidden_states_null, pred_latent], 0)
        ## sample images from here
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        bsz = len(pred_latent) // 2
        # latent_size = self.config.input_size
        latent_size = 32
        latent_channels = self.model.sana.config.in_channels


        latents = randn_tensor(
            shape=(bsz * num_images_per_prompt, latent_channels, latent_size, latent_size),
            generator=None,
            device=device,
            dtype=torch.bfloat16,
        )

        pred=meanflow_sampler(
            model=self.model.sana,
            latents=latents,
            encoder_hidden_states=self.model.diffusion_connector(pred_latent),
            cfg_scale=guidance_scale,
            num_steps=num_inference_steps,
        )
       
        samples = self.decode_latents(pred.to(self.model.sana_vae.dtype) if self.model.sana_vae is not None else pred, return_tensor=return_tensor)      
        return gen_ids, samples


AutoConfig.register("blip3o_qwen_inference", blip3oQwenConfig)
AutoModelForCausalLM.register(blip3oQwenConfig, blip3oQwenForInferenceLM)

