from dataclasses import dataclass
import torch
from PIL import Image
from transformers import AutoTokenizer
from blip3o.model import *
import os



import torch
import random
import numpy as np


# 固定随机种子

seed=1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

@dataclass
class T2IConfig:
    
    model_path: str = "../checkpoint"
    device: str = "cuda:1"
    dtype: torch.dtype = torch.bfloat16
    scale: int = 0  
    seq_len: int = 729  
    top_p: float = 0.95
    top_k: int = 1200

class TextToImageInference:
    def __init__(self, config: T2IConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._load_models()
        
    def _load_models(self):
        self.model = blip3oQwenForInferenceLM.from_pretrained(self.config.model_path, torch_dtype=self.config.dtype).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

    def generate_image(self, prompt: str,cfg_guidance,num_step) -> Image.Image:

        batch_messages = []


        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please generate image based on the following caption: {prompt}"}
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        input_text += f"<im_start><S{self.config.scale}>"
        
        batch_messages.append(input_text)

        # tokenize as a batch
        inputs = self.tokenizer(batch_messages, return_tensors="pt", padding=True, truncation=True, padding_side="left")
        _, output_image = self.model.generate_images(
            inputs.input_ids.to(self.device),
            inputs.attention_mask.to(self.device),
            max_new_tokens=self.config.seq_len,
            # image_sizes=512,
            do_sample=True,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            guidance_scale=cfg_guidance,
            num_inference_steps=num_step,)

        print(output_image)
        return output_image[0]


def main():
    
    
    config = T2IConfig()
    inference = TextToImageInference(config)

    prompts = [
        'A surreal scene on a lunar-like surface, where a brown horse is standing on the back of an astronaut. The horse, which has a dark mane and tail, is equipped with a brown leather saddle and bridle. The astronaut is on their hands and knees on the grey, dusty ground, wearing a white spacesuit with a patch on the shoulder. The astronaut helmet has a dark, reflective visor. The background is the blackness of space, with the blue and white Earth visible in the distance.',
    ]   

    num_step=2
    cfg_guidance=2

    output_dir = "../output_dir"

    os.makedirs(output_dir, exist_ok=True)

    for idx, prompt in enumerate(prompts):
        image_sana = inference.generate_image(prompt,cfg_guidance,num_step) 

        save_path = os.path.join(output_dir, f"blip3o_next_{idx:02d}.png")
        image_sana.save(save_path)

        print(f"Saved: {save_path}")



if __name__ == "__main__":
    main()  

