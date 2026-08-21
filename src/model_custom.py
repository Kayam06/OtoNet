import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Utilities ----------
def hard_swish(x):
    return x * F.hardtanh(x + 3.0, 0.0, 6.0) / 6.0


class HSwish(nn.Module):
    def forward(self, x):
        return hard_swish(x)


class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=4):
        super().__init__()
        hidden = max(8, in_ch // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, in_ch, 1)

    def forward(self, x):
        s = self.pool(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


# Stochastic Depth / DropPath
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * random_tensor


# ---------- MBConv with SE ----------
class MBConv(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        expand_ratio=4,
        se=True,
        drop_path=0.0,
        activation="hswish",
    ):
        super().__init__()
        hidden = int(in_ch * expand_ratio)
        use_res = stride == 1 and in_ch == out_ch

        act = HSwish() if activation == "hswish" else nn.ReLU(inplace=True)

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                act,
            ]

        # depthwise
        layers += [
            nn.Conv2d(
                hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False
            ),
            nn.BatchNorm2d(hidden),
            act,
        ]

        if se:
            layers += [SEBlock(hidden)]

        # project
        layers += [nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]

        self.block = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path)
        self.use_res = use_res
        self.act = act

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = self.drop_path(out) + x
        return out


class CustomNetMBConv(nn.Module):

    def __init__(self, num_classes=5, drop_path_rate=0.1):
        super().__init__()
        self.num_classes = num_classes

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),  # 288 -> 144
            nn.BatchNorm2d(32),
            HSwish(),
        )

        # Stages: (in, out, stride, repeats, expand)
        cfg = [
            # in, out, s, n, t
            (32, 64, 2, 2, 4),
            (64, 96, 2, 3, 4),
            (96, 160, 2, 3, 4),
            (160, 256, 1, 3, 4),
        ]

        blocks = []
        total_blocks = sum(n for _, _, _, n, _ in cfg)
        b_idx = 0

        in_ch = 32
        for cin, cout, s, n, t in cfg:
            assert cin == in_ch
            for i in range(n):
                stride = s if i == 0 else 1
                drop_prob = drop_path_rate * b_idx / max(1, total_blocks - 1)
                blocks.append(
                    MBConv(
                        in_ch,
                        cout,
                        stride=stride,
                        expand_ratio=t,
                        se=True,
                        drop_path=drop_prob,
                    )
                )
                in_ch = cout
                b_idx += 1

        self.blocks = nn.Sequential(*blocks)

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            HSwish(),
        )

        # Expose the best Grad-CAM layer
        self.cam_layer = self.head[0]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(num_classes, pretrained=False, **kwargs):
    return CustomNetMBConv(num_classes=num_classes)
