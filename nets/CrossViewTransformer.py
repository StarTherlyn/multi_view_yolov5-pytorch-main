import torch
from torch import nn
import math


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(80, num_pos_feats)
        self.col_embed = nn.Embedding(80, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        # x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos



class CrossViewTransformer(nn.Module):  # 该transformer侧视图与俯视图进行结合
    def __init__(self, in_dim):
        super(CrossViewTransformer, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.down_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//2,kernel_size=1)
        self.posencod = PositionEmbeddingLearned(in_dim//2)

    def forward(self, topview, sideview):
        # topview_with_pos = self.posencod(topview)
        # sideview_with_pos = self.posencod(sideview)
        m_batchsize, C, width, height = topview.size()

        proj_query = self.query_conv(topview).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x N x C
        proj_key = self.key_conv(sideview).view(m_batchsize, -1, width * height)  # B x C x (N)
        proj_value = self.value_conv(sideview).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x N x C

        energy = torch.bmm(proj_query, proj_key)  # 计算乘法transpose check   B*N*N
        softmax = nn.Softmax(dim=2)
        energy = softmax(energy)
        trans_topview = torch.bmm(energy, proj_value).permute(0, 2, 1)  # B x C x N
        trans_topview = trans_topview.view(m_batchsize, C, width, height)
        # topview = self.down_conv(topview)
        output = topview + trans_topview
        # output = torch.cat((topview,trans_topview),1)
        # output = self.down_conv(output)
        return output


'''
多头注意力机制需在拼接后进行降维
downSample = nn.Conv2d(1024,128,kernel_size=1)
'''
class MultiHeadTransformer(nn.Module):

    def __init__(
            self,
            in_dim,
            num_head,
    ):
        super(MultiHeadTransformer, self).__init__()
        self.transformers = nn.ModuleList()
        for _ in range(num_head):
            self.transformers.append(
                CrossViewTransformer(
                    in_dim
                )
            )

    def forward(self, query, keyval):
        return torch.cat([m(query, keyval) for m in self.transformers], dim=1)


if __name__ == "__main__":
    x1= torch.rand(16,128,40,40)
    x2 = torch.rand(16,128,40,40)

    cross1 = CrossViewTransformer(128)
    output1 = cross1(x1,x2)
    print(output1.size())

    cross2 = MultiHeadTransformer(128,8)
    downSample = nn.Conv2d(1024,128,kernel_size=1)
    output2 = cross2(x1,x2)
    linear_output2 = downSample(output2)
    print(linear_output2.size())
