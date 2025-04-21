import torch
from typing import List, Union, Optional, Dict, Any, Callable
from diffusers.models.attention_processor import Attention, F
from .lora_controller import enable_lora


def attn_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    condition_latents: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb: Optional[torch.Tensor] = None,
    model_config: Optional[Dict[str, Any]] = {},
) -> torch.FloatTensor:
    batch_size, _, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )

    with enable_lora(
        (attn.to_q, attn.to_k, attn.to_v), model_config.get("latent_lora", False)
    ):
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
    if encoder_hidden_states is not None:
        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(
                encoder_hidden_states_query_proj
            )
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(
                encoder_hidden_states_key_proj
            )

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    if condition_latents is not None:
        cond_query = attn.to_q(condition_latents)
        cond_key = attn.to_k(condition_latents)
        cond_value = attn.to_v(condition_latents)

        cond_query = cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_value = cond_value.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        if attn.norm_q is not None:
            cond_query = attn.norm_q(cond_query)
        if attn.norm_k is not None:
            cond_key = attn.norm_k(cond_key)

    if cond_rotary_emb is not None:
        cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
        cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)

    if condition_latents is not None:
        query = torch.cat([query, cond_query], dim=2)
        key = torch.cat([key, cond_key], dim=2)
        value = torch.cat([value, cond_value], dim=2)

    if not model_config.get("union_cond_attn", True):
        # If we don't want to use the union condition attention, we need to mask the attention
        # between the hidden states and the condition latents
        attention_mask = torch.ones(
            query.shape[2], key.shape[2], device=query.device, dtype=torch.bool
        )
        condition_n = cond_query.shape[2]
        attention_mask[-condition_n:, :-condition_n] = False
        attention_mask[:-condition_n, -condition_n:] = False
    if hasattr(attn, "c_factor"):
        attention_mask = torch.zeros(
            query.shape[2], key.shape[2], device=query.device, dtype=query.dtype
        )
        condition_n = cond_query.shape[2]
        bias = torch.log(attn.c_factor[0])
        attention_mask[-condition_n:, :-condition_n] = bias
        attention_mask[:-condition_n, -condition_n:] = bias
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim
    )
    hidden_states = hidden_states.to(query.dtype)

    if encoder_hidden_states is not None:
        if condition_latents is not None:
            encoder_hidden_states, hidden_states, condition_latents = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[
                    :, encoder_hidden_states.shape[1] : -condition_latents.shape[1]
                ],
                hidden_states[:, -condition_latents.shape[1] :],
            )
        else:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

        with enable_lora((attn.to_out[0],), model_config.get("latent_lora", False)):
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if condition_latents is not None:
            condition_latents = attn.to_out[0](condition_latents)
            condition_latents = attn.to_out[1](condition_latents)

        return (
            (hidden_states, encoder_hidden_states, condition_latents)
            if condition_latents is not None
            else (hidden_states, encoder_hidden_states)
        )
    elif condition_latents is not None:
        # if there are condition_latents, we need to separate the hidden_states and the condition_latents
        hidden_states, condition_latents = (
            hidden_states[:, : -condition_latents.shape[1]],
            hidden_states[:, -condition_latents.shape[1] :],
        )
        return hidden_states, condition_latents
    else:
        return hidden_states


def block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    condition_latents: torch.FloatTensor,
    temb: torch.FloatTensor,
    cond_temb: torch.FloatTensor,
    cond_rotary_emb=None,
    image_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
):
    use_cond = condition_latents is not None
    with enable_lora((self.norm1.linear,), model_config.get("latent_lora", False)):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        self.norm1_context(encoder_hidden_states, emb=temb)
    )

    if use_cond:
        (
            norm_condition_latents,
            cond_gate_msa,
            cond_shift_mlp,
            cond_scale_mlp,
            cond_gate_mlp,
        ) = self.norm1(condition_latents, emb=cond_temb)

    # Attention.
    result = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        condition_latents=norm_condition_latents if use_cond else None,
        image_rotary_emb=image_rotary_emb,
        cond_rotary_emb=cond_rotary_emb if use_cond else None,
    )
    attn_output, context_attn_output = result[:2]
    cond_attn_output = result[2] if use_cond else None

    # Process attention outputs for the `hidden_states`.
    # 1. hidden_states
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
    # 2. encoder_hidden_states
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output
    # 3. condition_latents
    if use_cond:
        cond_attn_output = cond_gate_msa.unsqueeze(1) * cond_attn_output
        condition_latents = condition_latents + cond_attn_output
        if model_config.get("add_cond_attn", False):
            hidden_states += cond_attn_output

    # LayerNorm + MLP.
    # 1. hidden_states
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = (
        norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    )
    # 2. encoder_hidden_states
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = (
        norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    )
    # 3. condition_latents
    if use_cond:
        norm_condition_latents = self.norm2(condition_latents)
        norm_condition_latents = (
            norm_condition_latents * (1 + cond_scale_mlp[:, None])
            + cond_shift_mlp[:, None]
        )

    # Feed-forward.
    with enable_lora((self.ff.net[2],), model_config.get("latent_lora", False)):
        # 1. hidden_states
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    # 2. encoder_hidden_states
    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output
    # 3. condition_latents
    if use_cond:
        cond_ff_output = self.ff(norm_condition_latents)
        cond_ff_output = cond_gate_mlp.unsqueeze(1) * cond_ff_output

    # Process feed-forward outputs.
    hidden_states = hidden_states + ff_output
    encoder_hidden_states = encoder_hidden_states + context_ff_output
    if use_cond:
        condition_latents = condition_latents + cond_ff_output

    # Clip to avoid overflow.
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states, condition_latents if use_cond else None


def single_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    condition_latents: torch.FloatTensor = None,
    cond_temb: torch.FloatTensor = None,
    cond_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
):

    using_cond = condition_latents is not None
    residual = hidden_states
    with enable_lora(
        (
            self.norm.linear,
            self.proj_mlp,
        ),
        model_config.get("latent_lora", False),
    ):
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    if using_cond:
        residual_cond = condition_latents
        norm_condition_latents, cond_gate = self.norm(condition_latents, emb=cond_temb)
        mlp_cond_hidden_states = self.act_mlp(self.proj_mlp(norm_condition_latents))

    attn_output = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **(
            {
                "condition_latents": norm_condition_latents,
                "cond_rotary_emb": cond_rotary_emb if using_cond else None,
            }
            if using_cond
            else {}
        ),
    )
    if using_cond:
        attn_output, cond_attn_output = attn_output

    with enable_lora((self.proj_out,), model_config.get("latent_lora", False)):
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
    if using_cond:
        condition_latents = torch.cat([cond_attn_output, mlp_cond_hidden_states], dim=2)
        cond_gate = cond_gate.unsqueeze(1)
        condition_latents = cond_gate * self.proj_out(condition_latents)
        condition_latents = residual_cond + condition_latents

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states if not using_cond else (hidden_states, condition_latents)
