import torch
import torch.nn as nn



class WideAndDeep(nn.Module):
    def __init__(self, wide_input_dim, deep_input_dim):
        super().__init__()
        # Wide part: linear layer
        self.wide = nn.Linear(wide_input_dim, 1)

        # Deep part: MLP
        self.deep = nn.Sequential(
            nn.Linear(deep_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 최종 출력
        self.output = nn.Sigmoid()

    def forward(self, wide_input, deep_input):
        wide_out = self.wide(wide_input)         # (batch_size, 1)
        deep_out = self.deep(deep_input)         # (batch_size, 1)
        out = wide_out + deep_out                # element-wise sum
        return self.output(out)