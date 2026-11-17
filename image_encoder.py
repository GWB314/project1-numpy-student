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
        # TODO-02 的要求：[B,N,D] 变成 [B,H,N,D/H]。
        # 这个图像模型里 D=768、H=12、D/H=64。
        batch, tokens, width = x.shape

        # 第 1 步：把单一的特征轴切成 (head, head_dim) 两段。
        #   连续的 64 个特征属于同一个头，所以这样切是安全的。
        #   [B,N,D] -> [B,N,H,D/H]
        grouped = x.reshape(batch, tokens, self.heads, self.head_dim)

        # 第 2 步：把“头”这一轴挪到“token”轴前面，让每个头各自持有自己那份
        #   token 序列去做两两比较：[B,N,H,D/H] -> [B,H,N,D/H]。
        #   每个头内部的 token 顺序完全没有被打乱。
        return grouped.transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a：算 query 与 key 的相似度，并除以 sqrt(head_dim) 缩放。
        # q、k、v 的形状都是 [B,H,N,64]，scores 必须是 [B,H,N,N]。
        # 第 1 步：只交换 k 的最后两个轴——这里的 q、k 是 4 维的，不像课本上
        #   那个二维例子只有一对轴可交换。图像轴和头轴保持不动，
        #   这一次矩阵乘就等于“每张图、每个头各做一次”。
        #   q [B,H,N,D/H] @ k^T [B,H,D/H,N] -> [B,H,N,N]
        # 第 2 步：除以 sqrt(head_dim)=8，让点积的量级保持在合理范围，
        #   再交给课程提供的 softmax 转成概率分布。
        scale = np.float32(np.sqrt(self.head_dim))
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b：用注意力系数对 value 的行做加权求和，再把各个头拼回去。
        # context 是 [B,H,N,64]，merged 是 [B,N,D]。
        # 第 1 步：v 的每个 token 行都变成所有 value 行的加权和。
        #   coefficients [B,H,N,N] @ v [B,H,N,D/H] -> [B,H,N,D/H]
        context = coefficients @ v

        # 第 2 步：按拆分的逆序原样拼回。各头必须先并排还原，
        #   输出投影才能把它们混合起来。
        #   [B,H,N,D/H] -> [B,N,H,D/H] -> [B,N,D]
        merged = context.transpose(0, 2, 1, 3).reshape(
            x.shape[0], x.shape[1], self.width)

        # TODO-03c：做输出投影，用 out_weight 和 out_bias。
        # 返回 [B,N,D]，注意权重存储的方向。
        # out_weight 是 [D,D]，按 [out,in] 存储，所以这里要转置。
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

        # TODO-04a：第一次残差相加，结果命名为 residual。
        # x 本身保持原样，只把注意力算出来的更新量加进去，
        # 所以两个操作数都是 [B,N,D]，形状不变。
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b：MLP = GELU(u @ W1.T + b1) @ W2.T + b2。
        # 隐藏层是 [B,N,3072]，mlp_update 是 [B,N,768]（图像塔的宽度）。
        # 返回 residual + mlp_update。两条残差支路都保持 [B,N,D]。
        # 第 1 步：把每个 token 从 768 维升到 3072 维（c_fc 是 [4D,D]）。
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        # 第 2 步：再投影回 768 维（c_proj 是 [D,4D]）。
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        # 第 3 步：第二次残差相加，加在 residual 支路上而不是加在 x 上，
        # 这样注意力那部分结果会原封不动地带下来。
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
        # TODO-01：拆开网格轴、重新排序、再展平。
        # 输入 [B,3,224,224]，输出 [B,49,3072]。
        batch, channels, height, width = images.shape

        # 课程给的权重里存了 50 个位置：1 个 CLS 加 49 个图像块，
        # 所以网格是 7x7，每块宽 224/7 = 32 像素。
        patch_tokens = self.position.shape[0] - 1
        grid = int(round(patch_tokens ** 0.5))
        patch = height // grid

        # 第 1 步：把两个空间轴各自切成 (grid, patch) 两段。轴顺序没变，
        #   所以此刻还没有发生任何重排。
        #   [B,3,224,224] -> [B,3,7,32,7,32]
        split = images.reshape(batch, channels, grid, patch, grid, patch)

        # 第 2 步：重排成 (块行, 块列, 通道, h, w)。两个网格轴按行优先挪到
        #   前面，正好得到“先从左到右、再从上到下”的块顺序。
        #   [B,3,7,32,7,32] -> [B,7,7,3,32,32]
        ordered = split.transpose(0, 2, 4, 1, 3, 5)

        # 第 3 步：把网格压成 49 行，每块压成 3*32*32 = 3072 个值，
        #   顺序是先 C、再 H、再 W。
        #   [B,7,7,3,32,32] -> [B,49,3072]
        return ordered.reshape(batch, grid * grid, channels * patch * patch)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a：把图像块投影到 [B,49,768]，这里没有偏置。
        # patch_weight 已被重塑为 [768,3072]，也就是 [out,in]，
        # 所以要先转置成 [3072,768] 再和块向量相乘。
        projected = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b：把 CLS 加到 token 轴最前面，再加上可学习的位置编码。
        # 返回 [B,50,768]。self.position 会沿 batch 轴自动广播。
        # 第 1 步：CLS 成为第 0 个 token 行，49 个图像块按顺序排在后面。
        #   [B,1,768] 和 [B,49,768] -> [B,50,768]
        tokens = np.concatenate((cls, projected), axis=1)
        # 第 2 步：每个 token 行配一个可学习位置；[50,768] 的表沿 batch 轴
        #   广播，不需要为每张图各复制一份。
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

        # 取出 CLS 行，做 post_norm，再用 self.projection 投影。
        # [B,50,768] -> [B,768] -> [B,512]，visual.proj 本来就是 [in,out]。
        # 投影结果命名为 output，最后的单位向量化由课程提供的函数完成。
        # 第 1 步：只保留 CLS 那一行，丢掉全部 49 个图像块行。
        #   [B,50,768] -> [B,768]
        cls = tokens[:, 0, :]
        # 第 2 步：用另一个独立的 post-Norm 实例，作用在 [B,768] 特征上。
        normalized = self.post_norm(cls)
        # 第 3 步：visual.proj 是 [768,512]，已经按 [in,out] 存储，直接乘即可。
        #   在这里转置就是错的。
        output = normalized @ self.projection

        return unit_vector(output)
