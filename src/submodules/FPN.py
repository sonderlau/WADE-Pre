import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(
        self,
        num_blocks,
        block=Bottleneck,
        in_channels=3,
        feature_channels=256,
        layer_channels=None,
        input_sizes=None,
    ):
        """
        Multi-Scale Feature Pyramid Network (FPN) Implementation.

        Args:
            block: block type, default is Bottleneck
            num_blocks: block numer per layer
            in_channels: channel of input feature maps
            feature_channels: FPN feature channels
            layer_channels: each layer's base channels before expansion
                            e.g., [64, 128, 256] for 3 layers
            input_sizes: each layer's input sizes，e.g., [(23,23), (38,38), (68,68)], from low to high
        """
        super(FPN, self).__init__()

        if layer_channels is None:
            layer_channels = [64, 128, 256, 512]

        self.num_layers = len(num_blocks)
        self.feature_channels = feature_channels
        self.input_sizes = input_sizes

        self.input_adapters = nn.ModuleList()
        self.in_planes_list = []

        for i, (channels, num_block) in enumerate(zip(layer_channels, num_blocks)):

            adapter = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    channels * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * block.expansion),
                nn.ReLU(inplace=True),
            )
            self.input_adapters.append(adapter)

        self.bottom_up_layers = nn.ModuleList()
        for i, (channels, num_block) in enumerate(zip(layer_channels, num_blocks)):
            self.in_planes = channels * block.expansion

            layer = self._make_layer(block, channels, num_block, stride=1)
            self.bottom_up_layers.append(layer)

        last_channels = layer_channels[-1] * block.expansion
        self.toplayer = nn.Conv2d(
            last_channels, feature_channels, kernel_size=1, stride=1, padding=0
        )


        self.lateral_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            in_ch = layer_channels[self.num_layers - 2 - i] * block.expansion
            lateral = nn.Conv2d(
                in_ch, feature_channels, kernel_size=1, stride=1, padding=0
            )
            self.lateral_layers.append(lateral)


        self.smooth_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            smooth = nn.Conv2d(
                feature_channels, feature_channels, kernel_size=3, stride=1, padding=1
            )
            self.smooth_layers.append(smooth)


        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + y

    def forward(self, inputs):
        """
        Args:
            inputs: multi-layers input, shape like: [B,C,23,23], [B,C,38,38], [B,C,68,68]

        Returns:
            tuple: all output tensors, e.g., ([B,F,23,23], [B,F,38,38], [B,F,68,68]), f is the output feature channels

        """
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("Input should be a list or tuple of feature maps")

        if len(inputs) != self.num_layers:
            raise ValueError(
                f"Input layer number {len(inputs)} is not equal to {self.num_layers} "
            )

        # Bottom-up
        bottom_up_features = []
        for i, (inp, adapter, layer) in enumerate(
            zip(inputs, self.input_adapters, self.bottom_up_layers)
        ):

            feat = adapter(inp)
            # residual connection
            feat = layer(feat)
            bottom_up_features.append(feat)

        # Top-down
        top_down_features = []
        p = self.toplayer(bottom_up_features[-1])
        top_down_features.append(p)

        for i in range(self.num_layers - 1):
            lateral_idx = i
            bottom_up_idx = self.num_layers - 2 - i

            p = self._upsample_add(
                p, self.lateral_layers[lateral_idx](bottom_up_features[bottom_up_idx])
            )
            top_down_features.append(p)


        top_down_features = top_down_features[::-1]

        # Smooth
        output_features = []
        for i in range(self.num_layers):
            if i < self.num_layers - 1:

                smoothed = self.smooth_layers[i](top_down_features[i])
                output_features.append(smoothed)
            else:
                # the lowest resolution layer
                # no need to smooth
                output_features.append(top_down_features[i])
        
        output_features = [f.contiguous() for f in output_features]
        return tuple(output_features)
