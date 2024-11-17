import os
import json
import torch
import folder_paths

from .loader import load_FluxDeCLIP

class FluxDeCLIPCheckpointLoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
				"adapter": (folder_paths.get_filename_list("checkpoints"),),
			}
		}
	RETURN_TYPES = ("MODEL",)
	RETURN_NAMES = ("model",)
	FUNCTION = "load_checkpoint"
	CATEGORY = "ExtraModels/FluxDeCLIP"
	TITLE = "FluxDeCLIPCheckpointLoader"

	def load_checkpoint(self, ckpt_name, adapter):
		ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
		distillation_adapter = folder_paths.get_full_path("checkpoints", adapter)
		fluxdeclip = load_FluxDeCLIP(
			model_path = ckpt_path,
			adapter_path = distillation_adapter
		)
		return (fluxdeclip,)

NODE_CLASS_MAPPINGS = {
	"FluxDeCLIPCheckpointLoader" : FluxDeCLIPCheckpointLoader,
}
