"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # TODO-02 contract: [B,N,D] to [B,H,N,D/H].
        # Image model: D=768, H=12, D/H=64.
        batch, tokens, width = x.shape

        # Step 1: cut the single feature axis into (head, head_dim) pairs.
        #   Contiguous features belong to the same head, so this split is safe.
        #   [B,N,D] -> [B,N,H,D/H]
        grouped = x.reshape(batch, tokens, self.heads, self.head_dim)

        # Step 2: move the head axis in front of the token axis so each head
        #   owns its own token-by-token comparison: [B,N,H,D/H] -> [B,H,N,D/H].
        #   Token order inside every head is untouched.
        return grouped.transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        # Step 1: exchange only the final two axes of k, because q and k are 4D
        #   here (the 2D teaching example only has one axis pair to swap).
        #   Keeping image and head groups fixed turns this into one batched
        #   matrix product per image and per head.
        #   q [B,H,N,D/H] @ k^T [B,H,D/H,N] -> [B,H,N,N]
        # Step 2: divide by sqrt(head_dim)=8 so the dot products keep a
        #   comparable magnitude before the supplied softmax conversion.
        scale = np.float32(np.sqrt(self.head_dim))
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].
        # Step 1: every token row of v becomes a weighted sum of all value rows.
        #   coefficients [B,H,N,N] @ v [B,H,N,D/H] -> [B,H,N,D/H]
        context = coefficients @ v

        # Step 2: undo the split exactly in reverse order. Heads must be back
        #   side by side before the output projection can mix them.
        #   [B,H,N,D/H] -> [B,N,H,D/H] -> [B,N,D]
        merged = context.transpose(0, 2, 1, 3).reshape(
            x.shape[0], x.shape[1], self.width)

        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        # out_weight is [D,D] stored as [out,in], so it is transposed here.
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        # The untouched x keeps the original token values; only the attention
        # update is added, so both operands stay [B,N,D] and no shape changes.
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        # Step 1: widen each token from 768 to 3072 features (c_fc is [4D,D]).
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        # Step 2: project back from 3072 to 768 (c_proj is [D,4D]).
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        # Step 3: second residual addition, on the residual branch rather than
        # on x, so the attention result carries forward unchanged.
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # TODO-01: separate patch-grid axes; reorder; flatten.
        # Input [B,3,224,224], return [B,49,3072].
        batch, channels, height, width = images.shape

        # The supplied checkpoint stores 50 positions: one CLS plus 49 patches,
        # so the grid is 7x7 and each patch is 224/7 = 32 pixels wide.
        patch_tokens = self.position.shape[0] - 1
        grid = int(round(patch_tokens ** 0.5))
        patch = height // grid

        # Step 1: cut both spatial axes into (grid, patch) pairs. Axis order is
        #   preserved, so nothing has been reordered yet.
        #   [B,3,224,224] -> [B,3,7,32,7,32]
        split = images.reshape(batch, channels, grid, patch, grid, patch)

        # Step 2: reorder to (patch row, patch column, channels, h, w). The two
        #   grid axes move forward in row-major order, which produces the
        #   required "left to right, then top to bottom" patch order.
        #   [B,3,7,32,7,32] -> [B,7,7,3,32,32]
        ordered = split.transpose(0, 2, 4, 1, 3, 5)

        # Step 3: flatten the grid into 49 rows and each patch into
        #   3*32*32 = 3072 values, ordered C then H then W.
        #   [B,7,7,3,32,32] -> [B,49,3072]
        return ordered.reshape(batch, grid * grid, channels * patch * patch)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        # patch_weight was reshaped to [768,3072] = [out,in], so it is
        # transposed to [3072,768] before multiplying the patch rows.
        projected = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        # Step 1: CLS becomes token row 0, the 49 patch rows follow in order.
        #   [B,1,768] and [B,49,768] -> [B,50,768]
        tokens = np.concatenate((cls, projected), axis=1)
        # Step 2: one learned position per token row; the [50,768] table
        #   broadcasts across the batch axis without copying per image.
        return tokens + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        # Step 1: keep only the CLS row and drop all 49 patch rows.
        #   [B,50,768] -> [B,768]
        cls = tokens[:, 0, :]
        # Step 2: a separate post-Norm instance, applied to the [B,768] features.
        normalized = self.post_norm(cls)
        # Step 3: visual.proj is [768,512] already stored as [in,out], so it is
        #   used directly. Transposing it here would be wrong.
        output = normalized @ self.projection

        return unit_vector(output)
