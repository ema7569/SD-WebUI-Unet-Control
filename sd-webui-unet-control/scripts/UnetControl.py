import os
import torch
import modules.scripts as scripts
import gradio as gr
from modules.processing import Processed, process_images, StableDiffusionProcessing
from modules import devices, prompt_parser
from ldm.modules.diffusionmodules.util import timestep_embedding
from modules.ui_components import InputAccordion

from scripts.UnetDebug import UnetDebug
from scripts.UnetParser import UnetParser
from scripts.Shared import set_current_unet_blocks

class UnetControl(scripts.Script):

    def title(self):
        return "U-Net Control"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):

        with InputAccordion(True, label=self.title()) as gr_enabled:
            with gr.Row():
                gr_debug = gr.Checkbox(label="Debug in console", value=True)
                gr_unet_type = gr.Radio(
                    label="U-Net Structure",
                    choices=["unet", "free"],
                    value="unet"
                    #,type="index"  # on peut récupérer l’index ou la valeur directement
                )

        return [gr_enabled, gr_debug, gr_unet_type]

    def postprocess(self, p: StableDiffusionProcessing, result, *args):
        p.sd_model.model.diffusion_model.forward = self.org_forward

    def process(self, p: StableDiffusionProcessing, gr_enabled, gr_debug, gr_unet_type):
        if not gr_enabled:
            print("U-Net Control disabled")
            return
        
        set_current_unet_blocks(gr_unet_type)
        
        parser = UnetParser()
        prompts = parser.parse(prompt=p.prompt)
        negative_prompts = parser.parse(prompt=p.negative_prompt)
        
        if gr_debug:
            UnetDebug.display(prompts, negative_prompts)


        #############################################
        #Generation start here
        #############################################
      
        
        # get conditional
        with devices.autocast():
            uc = prompt_parser.get_learned_conditioning(p.sd_model, negative_prompts, p.steps)
            c = prompt_parser.get_learned_conditioning(p.sd_model, prompts, p.steps)

        blocks_cond = []
        for uc1, c1 in zip(uc, c):
            cond = torch.cat([c1[0].cond.unsqueeze(0), uc1[0].cond.unsqueeze(0)]) # ignore scheduled cond
            blocks_cond.append(cond)

        _self = p.sd_model.model.diffusion_model

        def new_forward(x, timesteps=None, context=None, y=None, **kwargs):
            """
            Apply the model to an input batch.
            :param x: an [N x C x ...] Tensor of inputs.
            :param timesteps: a 1-D batch of timesteps.
            :param context: conditioning plugged in via crossattn
            :param y: an [N] Tensor of labels, if class-conditional.
            :return: an [N x C x ...] Tensor of outputs.
            """
            # print("replaced", x.size(), timesteps, context.size() if context is not None else context)
            assert (y is not None) == (
                _self.num_classes is not None
            ), "must specify y if and only if the model is class-conditional"
            hs = []
            t_emb = timestep_embedding(timesteps, _self.model_channels, repeat_only=False)
            emb = _self.time_embed(t_emb)

            if _self.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + _self.label_emb(y)

            cond_index = 0
            h = x.type(_self.dtype)
            for module in _self.input_blocks:
                h = module(h, emb, blocks_cond[cond_index] if context is not None else None)  # context)
                cond_index += 1
                hs.append(h)
            h = _self.middle_block(h, emb, blocks_cond[cond_index] if context is not None else None)  # context)
            cond_index += 1
            for module in _self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, blocks_cond[cond_index] if context is not None else None)  # context)
                cond_index += 1
            h = h.type(x.dtype)
            if _self.predict_codebook_ids:
                return _self.id_predictor(h)
            else:
                return _self.out(h)

        # replace U-Net forward
        self.org_forward = p.sd_model.model.diffusion_model.forward
        p.sd_model.model.diffusion_model.forward = new_forward


