"In response to `transformer_accounting` problem of assignment 1."


def num_parameters(
    vocab_size: int,
    num_layers: int,
    d_model: int,
    d_ff: int,
):
    "Compute number of params in a transformer LLM with given hyperparams."
    # One d_model embedding per token, see embedding.py, Embedding.weight
    num_embedding_params = vocab_size * d_model
    # See rmsnorm.py, RMSNorm.gain
    num_layer_norm_params = d_model  # One gain param per d_model dim
    # See multi_head_self_attention.py, MultiHeadSelfAttention attributes
    num_attn_projection_params = 3 * d_model * d_model  # attn_projections
    num_out_projection_params = d_model * d_model  # out_projection
    num_MHSA_params = num_attn_projection_params + num_out_projection_params
    # See swiglu.py, SwiGLU attributes
    num_ffn_params = (  # w1  + w2  + w3
        (d_model * d_ff) + (d_ff * d_model) + (d_model * d_ff)
    )
    # See transformer_block.py, TransformerBlock attributes
    num_block_params = (
        num_layer_norm_params  # ln1
        + num_MHSA_params  # attn
        + num_layer_norm_params  # ln2
        + num_ffn_params  # ffn
    )
    # See transformer_lm.py, TransformerLM attributes
    return (
        num_embedding_params  # token_embeddings
        + num_layers * num_block_params  # layers
        + num_layer_norm_params  # lm_final
        + d_model * vocab_size  # lm_head
    )


def compute_params():
    """Solve (a) two ways

    Hand computation using num_parameters and direct counting of trainable
    params in a TransformerLM. Both give the same answer.

    """
    print(num_parameters(50_257, 48, 1_600, 4_288))
    # => 1_640_452_800

    from cs336_basics.transformer_lm import TransformerLM

    t = TransformerLM(50_257, 1024, 1_600, 48, 25, 4_288, rope_theta=10_000)
    print(sum(p.numel() for p in t.parameters() if p.requires_grad))
    # => 1_640_452_800


def num_flops(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    d_ff: int,
):
    "Solve (b)"
    # FLOPs for computing all the q/k/v projections, using the Rule on p. 27 of
    # the assignment.. Internal dimension n is d_model, m is external dimension
    # of attn_projections, 3*d_model, p is external dimension of x,
    # context_length. See MultiHeadSelfAttention.forward.
    #
    # Not including head breakdown, here, since calculations for heads combine
    # linearly.
    projections_flops = (2 * d_model) * (3 * d_model) * context_length
    # See first einx.dot in scaled_dot_product_attention.py
    attention_weight_flops = (2 * d_model) * context_length * context_length
    # See final einx.dot in that function. Here internal dimension is number of
    # keys, external dimensions are number of queries and the hidden dimension
    attn_weighted_sum_flops = (2 * context_length) * context_length * d_model
    # Back to MultiHeadSelfAttention, final self.out_projection
    out_projection_flops = (2 * d_model) * d_model * context_length
    # See swiglu.py
    w1_and_w3_flops = (2 * d_model) * d_ff * context_length
    w2_flops = (2 * d_ff) * context_length * d_model
    total_swiglu_flops = 2 * w1_and_w3_flops + w2_flops
    transformer_block_flops = (
        projections_flops
        + attention_weight_flops
        + attn_weighted_sum_flops
        + out_projection_flops
        + total_swiglu_flops
    )
    total_transformer_flops = num_layers * transformer_block_flops
    lm_head_flops = (2 * d_model) * context_length * vocab_size
    return total_transformer_flops + lm_head_flops


print(num_flops(50_257, 1024, 48, 1_600, 4_288))  # GPT-2 XL
# => 3_516_769_894_400

d_ff = (768 * 8) // 3  # 8/3rds of d_model=768 for GPT-2 small
d_ff = round(d_ff / 64) * 64  # Round to nearest multiple of 64
print(num_flops(50_257, 1024, 12, 768, d_ff))

d_ff = (1024 * 8) // 3  # 8/3rds of d_model=768 for GPT-2 small
d_ff = round(d_ff / 64) * 64  # Round to nearest multiple of 64
print(num_flops(50_257, 1024, 24, 1024, d_ff))

d_ff = (1280 * 8) // 3  # 8/3rds of d_model=768 for GPT-2 small
d_ff = round(d_ff / 64) * 64  # Round to nearest multiple of 64
print(num_flops(50_257, 1024, 36, 1280, d_ff))

print(num_flops(50_257, 16384, 48, 1_600, 4_288))  # GPT-2 XL
