import torch
from torch import nn

from torchvision.models.resnet import resnet18, resnet50


class DETR(nn.Module):
    """
    DETR (DEtection TRansformer) model, as per https://arxiv.org/abs/2005.12872.
    """

    def __init__(self, num_classes, hidden_dim, num_heads,
                 num_encoder_layers, num_decoder_layers,
                 backbone='resnet18'):
        super().__init__()
        # We take only convolutional layers from ResNet model
        if backbone == 'resnet18':
            self.backbone = nn.Sequential(*list(resnet18().children())[:-2])
            backbone_out = 512
        elif backbone == 'resnet50':
            self.backbone = nn.Sequential(*list(resnet50().children())[:-2])
            backbone_out = 2048
        else:
            raise NotImplementedError(f"Backbone {backbone} is not supported")
        # Get number of output channels of the backbone
        self.conv = nn.Conv2d(backbone_out, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, num_heads,
                                          num_encoder_layers, num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1))
        return self.linear_class(h), self.linear_bbox(h).sigmoid()


# detr = DETR(num_classes=10, hidden_dim=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6)
detr = DETR(num_classes=2, hidden_dim=256, num_heads=2, num_encoder_layers=2, num_decoder_layers=2)
detr.eval()
inputs = torch.randn(1, 3, 800, 1200)
logits, bboxes = detr(inputs)
print(logits)
print(bboxes)
