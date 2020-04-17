import numpy as np
import cupy

import chainer
import chainer.functions as F
import chainer.links as L


class Shortcut(chainer.Chain):
    def __init__(self, n_in, n_out, stride, loc, skip_type='B'):
        w = chainer.initializers.HeNormal(0.01)
        super(Shortcut, self).__init__()
        self.skip_type = skip_type
        self.n_in = n_in
        self.n_out = n_out
        self.loc = loc
        self.stride = stride
        with self.init_scope():
            if (n_in != n_out) and (self.skip_type != 'A'):
                self.conv = L.Convolution2D(n_in, n_out, 1, self.stride, 0, True, w)
                self.bn = L.BatchNormalization(n_in)

    def __call__(self, x):
        if (self.n_in == self.n_out) and (self.skip_type != 'C'):
            self.output = x
        elif self.skip_type == 'A':
            h = F.max_pooling_2d(x, self.stride)
            shape_hout = (h.shape[0], self.n_out - self.n_in) + h.shape[2:]
            h = F.concat((h, cupy.zeros(shape_hout, dtype=cupy.float32)))
            self.output = h
        elif (self.skip_type == 'B') and (self.loc == 'first'):
            h = self.conv(x)
            self.output = h
        elif (self.skip_type == 'B') and (self.loc != 'first'):
            h = self.conv(F.relu(self.bn(x)))
            self.output = h
        elif (self.skip_type == 'C') and (self.loc == 'first'):
            h = self.conv(x)
            self.output = h
        elif (self.skip_type == 'C') and (self.loc != 'first'):
            h = self.conv(F.relu(self.bn(x)))
            self.output = h
        return self.output


class BottleNeck(chainer.Chain):

    def __init__(self, n_in, n_mid, n_out, stride,  loc=None, type='Residual'):
        w = chainer.initializers.HeNormal()
        super(BottleNeck, self).__init__()
        self.loc = loc
        self.type = type
        with self.init_scope():
            self.bn1 = L.BatchNormalization(n_in)
            self.conv1 = L.Convolution2D(n_in, n_mid, 1, stride, 0, True, w)
            self.bn2 = L.BatchNormalization(n_mid)
            self.conv2 = L.Convolution2D(n_mid, n_mid, 3, 1, 1, True, w)
            self.bn3 = L.BatchNormalization(n_mid)
            self.conv3 = L.Convolution2D(n_mid, n_out, 1, 1, 0, True, w)
            if self.type == 'Residual':
                self.skip = Shortcut(n_in, n_out, stride, loc=loc, skip_type='B')

    def __call__(self, x):
        # For the first Residual Unit, we adopt the first activation right after conv and before splitting into two paths
        if self.loc == 'first':
            h = self.conv1(x)
        else:
            h = self.conv1(F.relu(self.bn1(x)))

        h = self.conv2(F.relu(self.bn2(h)))

        if self.type == 'Plain':
            self.output = self.conv3(F.relu(self.bn3(h)))
        elif self.type == 'Residual':
            self.output = self.conv3(F.relu(self.bn3(h))) + self.skip(x)

        return self.output


class Block(chainer.ChainList):

    def __init__(self, n_in, n_mid, n_out, n_resblocks, stride=1, loc=None, type='Residual'):
        super(Block, self).__init__()
        self.add_link(BottleNeck(n_in, n_mid, n_out, stride,  loc=loc, type=type))
        for _ in range(n_resblocks - 1):
            self.add_link(BottleNeck(n_out, n_mid, n_out, 1, type=type))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class ResNet(chainer.Chain):

    def __init__(self, n_class=10, n=3, type='Residual'):
        super(ResNet, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 3, 1, 1, True, w)
            self.bn2 = L.BatchNormalization(16)
            self.block3 = Block(16, 16, 64, n, 1, loc='first', type=type)
            self.block4 = Block(64, 32, 128, n, 2, type=type)
            self.block5 = Block(128, 64, 256, n, 2, type=type)
            self.bn5 = L.BatchNormalization(256)
            self.fc7 = L.Linear(None, n_class, initialW=w)

    def __call__(self, x):
        h = chainer.Variable(x)
        h = F.relu(self.bn2(self.conv1(h)))
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        # For the last Residual Unit , we adopt an extra activation right after its element-wise addition
        h = F.relu(self.bn5(h))
        h = F.average_pooling_2d(h, h.shape[2:])

        h = self.fc7(h)
        return h


class ResNet164(ResNet):

    def __init__(self, n_class=10):
        super(ResNet164, self).__init__(n_class, 18, 'Residual')


class ResNet272(ResNet):

    def __init__(self, n_class=10):
        super(ResNet272, self).__init__(n_class, 30, 'Residual')


if __name__ == '__main__':
    import numpy as np
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    model = ResNet(10)
    y = model(x)