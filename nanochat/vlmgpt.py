from re import S
from gpt import GPT, GPTConfig
from typing import Optional, Tuple, List, Union
from torch import nn
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithPast

@dataclass
class VLMConfig:
    model_name: str = 'clip-vit-base-patch16'
    image_special_token: str = '@' * 196
    image_ids: List = [34] * 196
    hidden_size: int = 768

class VisionProj(nn.Module):
    def __init__(self, vision_encoder_hidden_size=768, hidden_size=512):
        super().__init__()
        self.vision_encoder_hidden_size = vision_encoder_hidden_size
        self.hidden_size = hidden_size
        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_encoder_hidden_size, self.hidden_size)
            # TODO: add activation function?
        )

    def forward(self, image_encoders):
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj

def get_vision_models(model_path: str):
    # TODO: support vision models beyond CLIP
    from transformers import logging as hf_logging
    from transformers import CLIPProcessor, CLIPModel    
    hf_logging.set_verbosity_error()
    # if not os.path.exists(model_path):
    #     return None, None
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    # freeze vision_encoder's parameters
    for param in model.parameters():
        param.requires_grad = False
    return model.eval(), processor

def image2tensor(image, processor):
    if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
    inputs = processor(images=image, return_tensors="pt")['pixel_values']
    return inputs

def get_image_embeddings(image_tensors, vision_model):
    with torch.no_grad():
        outputs = vision_model.vision_model(pixel_values=image_tensors)
    img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
    return img_embedding

class VLMGPT(GPT):

    def __init__(self, gpt_config: GPTConfig, vlm_config: VLMConfig):
        super().__init__(gpt_config)
        self.vlm_config = vlm_config

        # load CLIP from transformers
        self.vision_encoder, self.vision_processor = get_vision_models(vlm_config.model_name)
        self.vision_proj = VisionProj(vision_encoder_hidden_size=vlm_config.hidden_size,
                                      hidden_size=gpt_config.hidden_size)
        
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        def find_indices(tokens, image_ids):
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                            matches[batch_idx].nonzero(as_tuple=True)[0]]
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
            } or None

        image_indices = find_indices(tokens, self.params.image_ids)
        if vision_tensors is not None and image_indices:
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[
                                  :seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h


    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', pixel_values: Optional[torch.FloatTensor] = None):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)

        # add image embeddings if provided
        if pixel_values is not None and start_pos == 0:
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            bs, num, c, im_h, im_w = pixel_values.shape
            stack_dim = 1 if bs > 1 else 0
            vision_tensors = torch.stack([
                get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                for i in range(num)
            ], dim=stack_dim)
            hidden_states = self.count_vision_proj(tokens=idx, h=hidden_states, vision_tensors=vision_tensors,
                                                   seqlen=idx.shape[1])

        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            return logits

    def forward2(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            bs, num, c, im_h, im_w = pixel_values.shape
            stack_dim = 1 if bs > 1 else 0
            vision_tensors = torch.stack([
                get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                for i in range(num)
            ], dim=stack_dim)
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors,
                                                   seqlen=input_ids.shape[1])

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        # aux_loss = sum(
        #     layer.mlp.aux_loss
        #     for layer in self.model.layers
        #     if isinstance(layer.mlp, MOEFeedForward)
        # )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        output = CausalLMOutputWithPast(logits=logits, past_key_values=presents, hidden_states=hidden_states)
        # output.aux_loss = aux_loss
        return output
