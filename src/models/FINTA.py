import torch
from torch.nn import ReLU, Conv1d, Sequential, Linear, Upsample


def conv1x3(channels, stride=1):
    return Sequential(*[
        Sequential(
            Conv1d(channels[i - 1],
                   channels[i],
                   3,
                   stride=stride,
                   padding=1,
                   padding_mode='replicate'), ReLU())
        for i in range(1, len(channels))
    ])


def conv1x3_up(channels, scale=2):
    return Sequential(*[
        Sequential(Upsample(
            scale_factor=scale), conv1x3([channels[i - 1], channels[i]]))
        for i in range(1, len(channels))
    ])


class FINTA(torch.nn.Module):
    def __init__(self, input_size, embedding_size=32, hidden_dims=None, streamline_length = 256, **kwargs):
        super(FINTA, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32,64,128,256,512,1024]  # default for FINTA

        self.hidden_dims = hidden_dims
        self.emb_size = embedding_size
        self.in_size = input_size
        self.scaling_factor = streamline_length // 2**(len(hidden_dims) - 1)
        self.encoder = Sequential(conv1x3([input_size, self.hidden_dims[0]]),
                           conv1x3(self.hidden_dims, stride=2))

        self.compress = Linear(self.hidden_dims[-1] * self.scaling_factor, embedding_size)
        self.decompress = Sequential(Linear(embedding_size, self.hidden_dims[-1] * self.scaling_factor), ReLU())

        self.decoder = Sequential(
            conv1x3_up(self.hidden_dims[::-1]),
            Conv1d(self.hidden_dims[0], input_size,3, stride=1, padding=1, padding_mode='replicate')
        )


    def forward(self, data):

        # Each batch is formed by a certain number of subjects (data.shape[0]). For each subject, we load N streamlines (data.shape[1])
        # When using the linear layer in the model we need an input of shape (1024, 8) for each streamline. 
        # The total number of streamlines in the batch is given by the number of subjects times the number of streamlines per subjects
        
        data_shape = data.shape
        nb_streamlines = data.shape[0] * data.shape[1]  
        x = data.view(-1, *data.shape[2:])

        x = x.permute(0, 2, 1).contiguous()
        x = self.encoder(x)

        emb = self.compress(x.view(nb_streamlines, -1))
        x = self.decompress(emb).view(nb_streamlines, x.size(1), -1)

        x = self.decoder(x)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(*data_shape)

        return x

    def embed(self, data):

        nb_streamlines = data.shape[0] * data.shape[1]
        x = data.view(-1, *data.shape[2:])

        x = x.permute(0, 2, 1).contiguous()
        x = self.encoder(x)

        return self.compress(x.view(nb_streamlines, -1))
