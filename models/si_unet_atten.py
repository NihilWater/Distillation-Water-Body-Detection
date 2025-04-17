# TODO 这里还没写完，可以在减小计算量上继续开展
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from os.path import join as pjoin
import numpy as np

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class Attention(nn.Module):
    def __init__(self, hidden_size, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = 8
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, hidden_size, vis):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, 512)
        self.attn = Attention(hidden_size, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    # def load_from(self, weights, n_block):
    #     ROOT = f"Transformer/encoderblock_{n_block}"
    #     with torch.no_grad():
    #         query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
    #         key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
    #         value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
    #         out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()
    #
    #         query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
    #         key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
    #         value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
    #         out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)
    #
    #         self.attn.query.weight.copy_(query_weight)
    #         self.attn.key.weight.copy_(key_weight)
    #         self.attn.value.weight.copy_(value_weight)
    #         self.attn.out.weight.copy_(out_weight)
    #         self.attn.query.bias.copy_(query_bias)
    #         self.attn.key.bias.copy_(key_bias)
    #         self.attn.value.bias.copy_(value_bias)
    #         self.attn.out.bias.copy_(out_bias)
    #
    #         mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
    #         mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
    #         mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
    #         mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()
    #
    #         self.ffn.fc1.weight.copy_(mlp_weight_0)
    #         self.ffn.fc2.weight.copy_(mlp_weight_1)
    #         self.ffn.fc1.bias.copy_(mlp_bias_0)
    #         self.ffn.fc2.bias.copy_(mlp_bias_1)
    #
    #         self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
    #         self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
    #         self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
    #         self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, hidden_size, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        for _ in range(4):
            layer = Block(hidden_size, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class S1UNetAtten(nn.Module):
    def __init__(self, n_classes=2, in_channels=3):
        super(S1UNetAtten, self).__init__()

        # Contraction path
        self.c1_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1) # Assuming input has 3 channels (e.g., RGB)
        self.c1_dropout = nn.Dropout(0.1)
        self.c1_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c2_conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.c2_dropout = nn.Dropout(0.1)
        self.c2_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c3_conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.c3_dropout = nn.Dropout(0.2)
        self.c3_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c4_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.c4_dropout = nn.Dropout(0.2)
        self.c4_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c5_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.c5_dropout = nn.Dropout(0.3)
        self.c5_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # for attention
        self.position_embeddings = nn.Parameter(torch.zeros(1, 256, 256))
        self.att_dropout = Dropout(0.1)
        self.encoder = Encoder(hidden_size=256, vis=False)

        # Expansive path
        self.u6_convt = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c6_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.c6_dropout = nn.Dropout(0.2)
        self.c6_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.u7_convt = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c7_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.c7_dropout = nn.Dropout(0.2)
        self.c7_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.u8_convt = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.c8_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.c8_dropout = nn.Dropout(0.1)
        self.c8_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.u9_convt = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.c9_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.c9_dropout = nn.Dropout(0.1)
        self.c9_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.outputs = nn.Conv2d(16, n_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        # 初始化上采样卷积层和输出卷积层
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # Contraction path
        c1 = F.relu(self.c1_conv1(x))
        c1 = self.c1_dropout(c1)
        c1 = F.relu(self.c1_conv2(c1))
        p1 = self.p1(c1)

        c2 = F.relu(self.c2_conv1(p1))
        c2 = self.c2_dropout(c2)
        c2 = F.relu(self.c2_conv2(c2))
        p2 = self.p2(c2)

        c3 = F.relu(self.c3_conv1(p2))
        c3 = self.c3_dropout(c3)
        c3 = F.relu(self.c3_conv2(c3))
        p3 = self.p3(c3)

        c4 = F.relu(self.c4_conv1(p3))
        c4 = self.c4_dropout(c4)
        c4 = F.relu(self.c4_conv2(c4))
        p4 = self.p4(c4)

        c5 = F.relu(self.c5_conv1(p4))
        c5 = self.c5_dropout(c5)
        c5 = F.relu(self.c5_conv2(c5))

        att_x = c5.flatten(2)
        att_x = att_x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = att_x + self.position_embeddings
        embedding_output = self.att_dropout(embeddings)
        encoded, _ = self.encoder(embedding_output)  # (B, n_patch, hidden)

        B, n_patch, hidden = encoded.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        encoded = encoded.permute(0, 2, 1)
        encoded = encoded.contiguous().view(B, hidden, h, w)

        # Expansive path
        u6 = self.u6_convt(encoded)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = F.relu(self.c6_conv1(u6))
        c6 = self.c6_dropout(c6)
        c6 = F.relu(self.c6_conv2(c6))

        u7 = self.u7_convt(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = F.relu(self.c7_conv1(u7))
        c7 = self.c7_dropout(c7)
        c7 = F.relu(self.c7_conv2(c7))

        u8 = self.u8_convt(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = F.relu(self.c8_conv1(u8))
        c8 = self.c8_dropout(c8)
        c8 = F.relu(self.c8_conv2(c8))

        u9 = self.u9_convt(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = F.relu(self.c9_conv1(u9))
        c9 = self.c9_dropout(c9)
        c9 = F.relu(self.c9_conv2(c9))

        res = self.outputs(c9)

        return res

if __name__ == "__main__":
    model = S1UNetAtten()
    print(model)

    # 假设输入是一个 RGB 图像，大小为 512 x 512
    x = torch.randn(1, 2, 512, 512)
    preds = model(x)
    print(preds.shape)  # 应该输出 torch.Size([1, 1, 512, 512])