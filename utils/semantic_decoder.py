import torch
from torch import nn
class SemanticDecoder(nn.Module):
    def __init__(self):
        super(SemanticDecoder, self).__init__()

    def init(self, input_size, num_classes):
        self.fc1 = nn.Linear(input_size, num_classes).to("cuda")

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(dim=0)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        print('x_shape:',x.shape)
        return x


