import torch
from torch.nn import ReLU, Linear, Sequential, SyncBatchNorm
from torch_geometric.nn import DynamicEdgeConv, EdgeConv, global_mean_pool

def MLP(channels, activation, batch_norm=True):
    layers = list()

    for i in range(1, len(channels)):
        l = [Linear(channels[i - 1], channels[i])]
        if activation is not None:
            l.extend([activation])
        if batch_norm:
            l.extend([SyncBatchNorm(channels[i])])

        layers.append(Sequential(*l))

    return Sequential(*layers)


class ECLIPSE(torch.nn.Module):

    def __init__(self, input_size, k=5, aggr='max', embedding_size=32, hidden_dims_ec=None,
                 hidden_dims_dec=None, batch_norm=True, activation=None, use_dynamic=True, **kwargs):

        super(GNNAutoencoder, self).__init__()

        if hidden_dims_ec is None:
            hidden_dims_ec = [64, 64, 64]  # defaults for Verifyber encoder
        if hidden_dims_dec is None:
            hidden_dims_dec = [128]

        self.hidden_dims_ec = hidden_dims_ec
        self.hidden_dims_dec = hidden_dims_dec
        self.embedding_size = embedding_size
        self.batch_norm = batch_norm
        self.activation = activation
        self.use_dynamic = use_dynamic

        self.enc_hidden_dims_ec = [2* input_size] + self.hidden_dims_ec
        self.enc_hidden_dims_dec = [self.enc_hidden_dims_ec[-1] * 2] + self.hidden_dims_dec

        self.dec_hidden_dims_dec = ([(self.enc_hidden_dims_ec[-1] + self.enc_hidden_dims_dec[-1]) * 2]
                                    + self.hidden_dims_dec[::-1] + [self.enc_hidden_dims_ec[-1]])
        self.dec_hidden_dims_ec = [2 * self.dec_hidden_dims_dec[-1]] + self.hidden_dims_ec[:-1:-1] + [input_size]

        self.conv1 = EdgeConv(MLP(self.enc_hidden_dims_ec, batch_norm=self.batch_norm, activation=self.activation), aggr)

        if self.use_dynamic:
            self.conv2 = DynamicEdgeConv(MLP(self.enc_hidden_dims_dec, batch_norm=self.batch_norm, activation=self.activation), k, aggr)
        else:
            self.conv2 = EdgeConv(MLP(self.enc_hidden_dims_dec, batch_norm=self.batch_norm, activation=self.activation), aggr)

        self.lin1 = MLP([self.hidden_dims_ec[-1] + self.hidden_dims_dec[-1], self.embedding_size], activation=self.activation)
        self.lin2 = MLP([self.embedding_size, self.hidden_dims_ec[-1] + self.hidden_dims_dec[-1]], activation=self.activation)

        if self.use_dynamic:
            self.conv3 = DynamicEdgeConv(MLP(self.dec_hidden_dims_dec, batch_norm=self.batch_norm, activation=self.activation), k, aggr)
        else:
            self.conv3 = EdgeConv(MLP(self.dec_hidden_dims_dec, batch_norm=self.batch_norm, activation=self.activation), aggr)

        self.conv4 = EdgeConv(MLP(self.dec_hidden_dims_ec, batch_norm=self.batch_norm, activation=self.activation), aggr)


    def forward(self, data):
        pos, batch, eidx = data.pos, data.batch, data.edge_index

        x1 = self.conv1(pos, eidx)

        if self.use_dynamic:
            x2 = self.conv2(x1, batch)
        else:
            x2 = self.conv2(x1, eidx)

        emb = self.lin1(torch.cat([x1, x2], dim=1))
        rev_emb = self.lin2(emb)

        if self.use_dynamic:
            rec_x = self.conv3(rev_emb, batch)
        else:
            rec_x = self.conv3(rev_emb, eidx)
        rec_x = self.conv4(rec_x, eidx)

        return rec_x


    def embed(self, data):
        pos, batch, eidx = data.pos, data.batch, data.edge_index

        x1 = self.conv1(pos, eidx)

        if self.use_dynamic:
            x2 = self.conv2(x1, batch)
        else:
            x2 = self.conv2(x1, eidx)

        x3 = self.lin1(torch.cat([x1, x2], dim=1))
        return global_mean_pool(x3, batch)

