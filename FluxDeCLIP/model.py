#Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
    ModulationOut,
)

from einops import rearrange, repeat
import comfy.ldm.common_dit

@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    qkv_bias: bool
    guidance_embed: bool

class FluxDeCLIP(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        params = FluxParams(**kwargs)
        self.params = params
        self.in_channels = params.in_channels * 2 * 2
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = operations.Linear(self.in_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
        self.txt_in = operations.Linear(params.context_in_dim, self.hidden_size, dtype=dtype, device=device)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, dtype=dtype, device=device, operations=operations)
                for _ in range(params.depth_single_blocks)
            ]
        )

        if final_layer:
            self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, dtype=dtype, device=device, operations=operations)
    
    @staticmethod
    def distribute_modulations(tensor: torch.Tensor):
        """
        Distributes slices of the tensor into the block_dict as ModulationOut objects.

        Args:
            tensor (torch.Tensor): Input tensor with shape [batch_size, vectors, dim].
        """
        batch_size, vectors, dim = tensor.shape

        block_dict = {}

        # HARD CODED VALUES! lookup table for the generated vectors
        # Add 38 single mod blocks
        for i in range(38):
            key = f"single_blocks.{i}.modulation.lin"
            block_dict[key] = None

        # Add 19 image double blocks
        for i in range(19):
            key = f"double_blocks.{i}.img_mod.lin"
            block_dict[key] = None

        # Add 19 text double blocks
        for i in range(19):
            key = f"double_blocks.{i}.txt_mod.lin"
            block_dict[key] = None

        # Add the final layer
        block_dict["final_layer.adaLN_modulation.1"] = None


        idx = 0  # Index to keep track of the vector slices

        for key in block_dict.keys():
            if "single_blocks" in key:
                # Single block: 1 ModulationOut
                block_dict[key] = ModulationOut(
                    shift=tensor[:, idx:idx+1, :],
                    scale=tensor[:, idx+1:idx+2, :],
                    gate=tensor[:, idx+2:idx+3, :]
                )
                idx += 3  # Advance by 3 vectors

            elif "img_mod" in key:
                # Double block: List of 2 ModulationOut
                double_block = []
                for _ in range(2):  # Create 2 ModulationOut objects
                    double_block.append(
                        ModulationOut(
                            shift=tensor[:, idx:idx+1, :],
                            scale=tensor[:, idx+1:idx+2, :],
                            gate=tensor[:, idx+2:idx+3, :]
                        )
                    )
                    idx += 3  # Advance by 3 vectors per ModulationOut
                block_dict[key] = double_block

            elif "txt_mod" in key:
                # Double block: List of 2 ModulationOut
                double_block = []
                for _ in range(2):  # Create 2 ModulationOut objects
                    double_block.append(
                        ModulationOut(
                            shift=tensor[:, idx:idx+1, :],
                            scale=tensor[:, idx+1:idx+2, :],
                            gate=tensor[:, idx+2:idx+3, :]
                        )
                    )
                    idx += 3  # Advance by 3 vectors per ModulationOut
                block_dict[key] = double_block

            elif "final_layer" in key:
                # Final layer: 1 ModulationOut
                block_dict[key] = [
                    tensor[:, idx:idx+1, :],
                    tensor[:, idx+1:idx+2, :],
                ]
                idx += 2  # Advance by 3 vectors

        return block_dict

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        guidance: Tensor = None,
        control=None,
        transformer_options={},
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)

        # distilled vector guidance
        device = img.device
        dtype = img.dtype
        mod_index_length = 344
        distill_timestep = timestep_embedding(torch.tensor(timesteps), 16).to(device=device, dtype=dtype)
        distil_guidance = timestep_embedding(torch.tensor(guidance), 16).to(device=device, dtype=dtype)
        # get all modulation index
        modulation_index = timestep_embedding(torch.tensor(list(range(mod_index_length))), 32).to(device=device, dtype=dtype)
        # we need to broadcast the modulation index here so each batch has all of the index
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
        # and we need to broadcast timestep and guidance along too
        timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1)
        # then and only then we could concatenate it together
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)

        mod_vectors = self.distilled_guidance_layer(input_vec)

        mod_vectors_dict = self.distribute_modulations(mod_vectors)


        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            img_mod = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
            txt_mod = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
            double_mod = [img_mod, txt_mod]

            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], distill_vec=args["distill_vec"])
                    return out

                out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": None, "pe": pe, "distill_vec": double_mod}, {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=None, pe=pe, distill_vec=double_mod)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]

                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], distill_vec=args["distill_vec"])
                    return out

                out = blocks_replace[("single_block", i)]({"img": img, "vec": None, "pe": pe, "distill_vec": single_mod}, {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=None, pe=pe, distill_vec=single_mod)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] :, ...] += add

        img = img[:, txt.shape[1] :, ...]
        final_mod = mod_vectors_dict["final_layer.adaLN_modulation.1"]
        img = self.final_layer(img, None, distill_vec=final_mod)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(self, x, timestep, context, y, guidance, control=None, transformer_options={}, **kwargs):
        bs, c, h, w = x.shape
        patch_size = 2
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance, control, transformer_options)
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]