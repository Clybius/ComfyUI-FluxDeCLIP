import comfy.supported_models_base
from comfy.supported_models import Flux
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
import torch
from comfy import model_management

class FluxDeCLIP(Flux):
    unet_config = {
        "image_model": "flux",
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = comfy.model_base.Flux(self, model_type=comfy.model_base.ModelType.FLOW, device=device)
        return out

def load_FluxDeCLIP(model_path):
	state_dict = comfy.utils.load_torch_file(model_path)
	state_dict = state_dict.get("model", state_dict)
	parameters = comfy.utils.calculate_parameters(state_dict)
	unet_dtype = model_management.unet_dtype(model_params=parameters)
	load_device = comfy.model_management.get_torch_device()
	offload_device = comfy.model_management.unet_offload_device()

	# ignore fp8/etc and use directly for now
	manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device)
	if manual_cast_dtype:
		print(f"FluxDeCLIP: falling back to {manual_cast_dtype}")
		unet_dtype = manual_cast_dtype

	model_conf = FluxDeCLIP.model_config
	model = comfy.model_base.Flux(
		model_conf,
		device=model_management.get_torch_device()
	)

	from .model import FluxDeCLIP as FluxDeCLIP_Model
	model.diffusion_model = FluxDeCLIP_Model(**model_conf.unet_config)

	model.diffusion_model.load_state_dict(state_dict)
	model.diffusion_model.dtype = unet_dtype
	model.diffusion_model.eval()
	model.diffusion_model.to(unet_dtype)

	model_patcher = comfy.model_patcher.ModelPatcher(
		model,
		load_device = load_device,
		offload_device = offload_device,
	)
	return model_patcher
