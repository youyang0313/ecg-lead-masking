"""Compact residual 1D classifier instantiated from protocol_v1.json."""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(incoming, outgoing, 5, stride=stride, padding=2, bias=False),
            nn.BatchNorm1d(outgoing), nn.ReLU(),
            nn.Conv1d(outgoing, outgoing, 5, padding=2, bias=False),
            nn.BatchNorm1d(outgoing))
        self.shortcut = (nn.Identity() if incoming == outgoing and stride == 1 else
                         nn.Sequential(nn.Conv1d(incoming, outgoing, 1, stride=stride, bias=False),
                                       nn.BatchNorm1d(outgoing)))
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.main(x)+self.shortcut(x))


class ECGClassifier(nn.Module):
    def __init__(self, protocol):
        super().__init__()
        architecture = protocol["architecture"]
        stem = architecture["stem"]
        self.stem = nn.Sequential(
            nn.Conv1d(architecture["input_channels"], stem["out_channels"],
                      stem["kernel"], stride=stem["stride"], padding=stem["padding"], bias=False),
            nn.BatchNorm1d(stem["out_channels"]), nn.ReLU())
        incoming = stem["out_channels"]
        blocks = []
        for block in architecture["blocks"]:
            blocks.append(ResidualBlock(incoming, block["out_channels"], block["first_conv_stride"]))
            incoming = block["out_channels"]
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(incoming, len(protocol["dataset"]["classes"]))

    def forward(self, x):
        return self.head(self.pool(self.blocks(self.stem(x))).squeeze(-1))


def sample_lead_mask(batch_size, generator):
    """50% complete; else K uniform 1..11 and subset uniform conditional on K."""
    masks = torch.ones((batch_size, 12), dtype=torch.bool)
    partial = torch.nonzero(torch.rand(batch_size, generator=generator) >= 0.5).flatten()
    counts = torch.randint(1, 12, (len(partial),), generator=generator)
    for row, count in zip(partial.tolist(), counts.tolist()):
        masks[row] = False
        masks[row, torch.randperm(12, generator=generator)[:count]] = True
    return masks
