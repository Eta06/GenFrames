"""Small standard-operator blocks shared by research candidates."""

from __future__ import annotations

from torch import Tensor, nn


class ConvAct(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
            ),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvAct(channels, channels, dilation=dilation),
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.body(inputs))

