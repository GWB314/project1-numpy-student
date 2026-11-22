"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.

Lines led by ``# Given:``, the constructors and the class skeletons come with
the course package and are unchanged. The six TODO blocks are the part written
here; they are also the orange boxes of Figure 1 and the red frames in Section 3
of report/main.tex.
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
        # D=768、H=12，每个头 64 维。
        batch, tokens, width = x.shape

        # 一条特征轴切成 (head, head_dim) 两段。相邻 64 个特征属于同一个头，
        # 直接切不会串。 [B,N,D] -> [B,N,H,D/H]
        grouped = x.reshape(batch, tokens, self.heads, self.head_dim)

        # 头轴挪到 token 轴前面，每个头各拿一份完整的 token 序列去两两比较。
        # 每个头内部的 token 顺序不动。 [B,N,H,D/H] -> [B,H,N,D/H]
        return grouped.transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a：相似度矩阵 scores，形状 [B,H,N,N]。
        # k 只换最后两个轴。这里 q、k 是 4 维的，不像课本那个二维例子只有
        # 一对轴可换；照抄 .T 会把图像轴和头轴一起翻掉。这样相乘等于
        # “每张图、每个头各算一次”。 q [B,H,N,64] @ k^T [B,H,64,N] -> [B,H,N,N]
        # 再除 sqrt(64)=8，把点积压回合理量级，交给课程给的 softmax。
        scale = np.float32(np.sqrt(self.head_dim))
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b：每个 token 的行变成所有 value 行的加权和。
        # coefficients [B,H,N,N] @ v [B,H,N,64] -> [B,H,N,64]
        context = coefficients @ v

        # 拼回去正好是 split_heads 的逆操作：各头先并排还原成 [B,N,H,64]，
        # 再压平成 [B,N,768]，后面那层输出投影才有东西可混。
        # [B,H,N,64] -> [B,N,H,64] -> [B,N,768]
        merged = context.transpose(0, 2, 1, 3).reshape(
            x.shape[0], x.shape[1], self.width)

        # TODO-03c：输出投影加偏置。out_weight 按 [out,in] 存，要先转置。
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

        # TODO-04a：第一次残差相加。x 不动，只把注意力的更新量加上去，
        # 两个操作数都是 [B,N,D]，形状不变。
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b：MLP，先升到 3072 维再降回 768 维。
        # c_fc 是 [4D,D]，c_proj 是 [D,4D]，两个都按 [out,in] 存，所以都要 .T。
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        # 第二次残差加在 residual 上，不是加在 x 上，注意力那部分才能原样带下来。
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
        # TODO-01：拆网格轴、重排、展平。
        batch, channels, height, width = images.shape

        # 位置表 50 个位置 = 1 个 CLS + 49 个块，所以是 7x7 网格，
        # 每块 224/7 = 32 像素。
        patch_tokens = self.position.shape[0] - 1
        grid = int(round(patch_tokens ** 0.5))
        patch = height // grid

        # 1) 两个空间轴各切一刀。轴顺序没动，这一步还没发生重排。
        #    [B,3,224,224] -> [B,3,7,32,7,32]
        split = images.reshape(batch, channels, grid, patch, grid, patch)

        # 2) 两个网格轴按行优先挪到最前，正好得到“先左右、再上下”的块顺序。
        #    [B,3,7,32,7,32] -> [B,7,7,3,32,32]
        ordered = split.transpose(0, 2, 4, 1, 3, 5)

        # 3) 压成 49 行，每行 3*32*32 = 3072 个值，块内按 C、H、W 排。
        #    [B,7,7,3,32,32] -> [B,49,3072]
        return ordered.reshape(batch, grid * grid, channels * patch * patch)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a：块投影，没有偏置。patch_weight 已经是 [768,3072]，
        # 也就是 [out,in]，所以要 .T 之后才能和 3072 维的块向量相乘。
        projected = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b：CLS 插到第 0 行，49 个块按原顺序排在后面。
        # [B,1,768] 和 [B,49,768] 拼成 [B,50,768]
        tokens = np.concatenate((cls, projected), axis=1)
        # 位置表是 [50,768]，沿 batch 轴广播，不用为每张图各复制一份。
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

        # TODO-06：只留 CLS 那一行，过 post-Norm，投影到 512 维，再归一化。
        # [B,50,768] -> [B,768] -> [B,512]
        # 只取第 0 行，49 个块行全部丢掉。
        cls = tokens[:, 0, :]
        # 这里用的是另一个独立的 post-Norm，不是前面那个 pre_norm。
        normalized = self.post_norm(cls)
        # visual.proj 是 [768,512]，本身就按 [in,out] 存，直接乘；加 .T 就错了。
        output = normalized @ self.projection

        return unit_vector(output)
