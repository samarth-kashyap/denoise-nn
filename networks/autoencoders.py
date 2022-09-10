import torch.nn as nn

class LinearAutoencoder(nn.Module):
    __inputs__ = ["input_dim",
                  "encoding_dim",
                  "num_layers"]
    __methods__ = ["forward"]
    """ Class for Linear autoencoders.

    Inputs:
    -------
    input_dim - int
        dimension of input data
    encoding_dim - int
        dimension of encoding data
    num_layers - int
        number of layers in encoder and decoder

    Methods:
    --------
    forward - 
    """
    def __init__(self, input_dim, encoding_dim, num_layers=3):
        super(LinearAutoencoder, self).__init__()
        dim_diff = input_dim - encoding_dim
        assert dim_diff > 0, "Please ensure encoding_dim < input_dim"
        ddim = dim_diff//num_layers
        self.encoders = []
        self.decoders = []

        dim1 = input_dim
        for i in range(num_layers):
            dim2 = dim1 - ddim if i != num_layers - 1 else encoding_dim
            self.encoders.append(nn.Linear(dim1, dim2))
            dim1 = dim2

        dim1 = encoding_dim
        for i in range(num_layers):
            dim2 = dim1 + ddim if i != num_layers - 1 else input_dim
            self.decoders.append(nn.Linear(dim1, dim2))

    def forward(self, x):
        # define feedforward behavior 
        # and scale the *output* layer with a sigmoid activation function
        for i in range(num_layers):
            x = F.relu(self.encoders[i](x))

        for i in range(num_layers):
            x = torch.sigmoid(self.decoders[i](x))
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        #Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))    
        return x
