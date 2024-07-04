class SplitByConv1d(nn.Module):
    """Focus width and height information into channel space."""
    def __init__(self, in_channels, nc=80):
        super(SplitBy1x1Conv, self).__init__()
        xy_conv_weights = torch.zeros(2, in_channels, 1)
        wh_conv_weights = torch.zeros(2, in_channels, 1)
        conf_conv_weights = torch.zeros(nc + 5 - 4, in_channels, 1)
        for i in range(0, 2):
            xy_conv_weights[i][i][0] = 1.
        for i in range(0, 2):
            wh_conv_weights[i][i + 2][0] = 1.
        for i in range(0, 81):
            conf_conv_weights[i][i + 4][0] = 1.
        self.conv1 = nn.Conv1d(in_channels, 2, 1, 1, padding=0, bias=False)
        self.conv1.weight = nn.Parameter(xy_conv_weights, requires_grad=False)
        self.conv2 = nn.Conv1d(in_channels, 2, 1, 1, padding=0, bias=False)
        self.conv2.weight = nn.Parameter(wh_conv_weights, requires_grad=False)
        self.conv3 = nn.Conv1d(in_channels, nc + 5 - 4, 1, 1, padding=0, bias=False)
        self.conv3.weight = nn.Parameter(conf_conv_weights, requires_grad=False)


    def forward(self, x):
        return (self.conv1(x), self.conv2(x), self.conv3(x))
