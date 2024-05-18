import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu

class RNN(nn.Module):
    name = "run"
    rnn_class = nn.RNN

    def __init__(self, input_size, hidden_size, n_layer, **kwargs):
        super().__init__()
        self.model = self.rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layer,
            batch_first=False,
            bias=True,
        )
        self.hidden_size = hidden_size
        self.num_layer = n_layer
        self._initialize()

    def _initialize(self):
        for name, param in self.model.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    def forward(self, inputs, h_0):
        """
        inputs: (T, B, input_dim)
        h_0: (num_layers=1, B, hidden_size)
        return
        output: (T, B, hidden_size)
        h_n: (num_layers=1, B, hidden_size), only used in inference
        """
        if h_0 is None:
            h_0 = self.get_zero_internal_state(inputs.size(1))
        output, h_n = self.model(inputs, h_0)
        return output, h_n


    def get_zero_internal_state(self, batch_size = 1):
        return ptu.zeros((self.num_layers, batch_size, self.hidden_size)).float()

class GRU(nn.Module):
    name = "gru"
    rnn_class = nn.GRU
    def __init__(self, input_size, hidden_size, n_layer, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = n_layer
        self.output_size = output_size

        self.model = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True,
            bias = True,
            dropout = 0.2 if self.num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self._initialize()

    def _initialize(self):
        for name, param in self.model.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    def forward(self, inputs, h_0):
        if h_0 is None:
            h_0 = self.get_zero_internal_state(inputs.size(0))
        
        output, h_n = self.model(inputs, h_0)
        output = self.fc(output[:, -1, :])  # Shape of out: (batch_size, output_size)
        print(output)
        return output

    
    def get_zero_internal_state(self, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.model.weight_ih_l0.device)



# class GRU(nn.Module):
#     name = "gru"
#     rnn_class = nn.GRU

#     def __init__(self, input_size, hidden_size, n_layer, output_size):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = n_layer
#         self.output_size = output_size

#         self.model = nn.GRU(
#             input_size=self.input_size,
#             hidden_size=self.hidden_size,
#             num_layers=self.num_layers,
#             batch_first=True,
#             bias=True,
#             dropout=0.2 if self.num_layers > 1 else 0.0
#         )
#         self.fc = nn.Linear(self.hidden_size, self.output_size)
#         self.layer_norm = nn.LayerNorm(self.hidden_size)  # Layer normalization
#         self._initialize()

#     def _initialize(self):
#         for name, param in self.model.named_parameters():
#             if "bias" in name:
#                 nn.init.constant_(param, 0)
#             elif "weight" in name:
#                 nn.init.orthogonal_(param)

#     def forward(self, inputs, h_0=None):
#         if h_0 is None:
#             h_0 = torch.zeros((self.num_layers, inputs.size(0), self.hidden_size)).to(inputs.device)

#         output, h_n = self.model(inputs, h_0)
#         output = self.layer_norm(output[:, -1, :])  # Applying Layer Norm
#         output = self.fc(output)
#         return output