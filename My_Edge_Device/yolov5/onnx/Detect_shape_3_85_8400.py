import torch
import torch.nn as nn


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        # self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs * 3, 85, 20,20)
            # x[i] = x[i].view(bs * self.na, self.no, ny, nx).permute(0, 2, 3, 1).contiguous()
            x[i] = x[i].view(bs * self.na, self.no, ny, nx).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    # xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 3)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 1)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    # y = torch.cat((xy, wh, conf), 3)
                    y = torch.cat((xy, wh, conf), 1)
                # z.append(y.view(bs, self.na * nx * ny, self.no))
                z.append(y.view(bs * self.na, self.no, nx * ny))

        return x if self.training else (torch.cat(z, 2),) if self.export else (torch.cat(z, 2), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        # shape = 1 * self.na, ny, nx, 2  # grid shape
        shape = 1 * self.na, 2, ny, nx  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        # grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        grid = torch.stack((xv, yv), 0).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        # anchor_grid = (self.anchors[i] * self.stride[i]).view((1 * self.na, 1, 1, 2)).expand(shape)
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1 * self.na, 2, 1, 1)).expand(shape)
        return grid, anchor_grid
