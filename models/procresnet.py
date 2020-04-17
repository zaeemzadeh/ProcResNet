from utilities import regularize_conv
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from models.resnet import Block


class ProcResNet(chainer.Chain):

    def __init__(self, n_class=10, n=3, type='Residual', widen=1):
        super(ProcResNet, self).__init__()
        w = chainer.initializers.HeNormal()
        w_no_relu = chainer.initializers.HeNormal(scale=np.sqrt(1./2))
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64*widen, 3, 1, 1, True, w_no_relu)
            self.block3 = Block(64*widen, 16*widen, 64*widen, n, 1, type=type)
            self.bn_trans_1 = L.BatchNormalization(64)
            self.trans_conv1 = L.Convolution2D(64*widen, 128*widen, 1, 2, 0, True, w_no_relu)
            self.block4 = Block(128*widen, 32*widen, 128*widen, n, 1, type=type)
            self.bn_trans_2 = L.BatchNormalization(128*widen)
            self.trans_conv2 = L.Convolution2D(128*widen, 256*widen, 1, 2, 0, True, w_no_relu)
            self.block5 = Block(256*widen, 64*widen, 256*widen, n, 1, type=type)
            self.bn5 = L.BatchNormalization(256*widen)
            self.fc7 = L.Linear(None, n_class, initialW=w)

    def __call__(self, x):
        h = chainer.Variable(x)

        h = self.conv1(h)
        self.conv1.out_size = h.shape[-2:]
        h = self.block3(h)

        h = self.trans_conv1(h)
        self.trans_conv1.out_size = h.shape[-2:]
        h = self.block4(h)

        h = self.trans_conv2(h)
        self.trans_conv2.out_size = h.shape[-2:]
        h = self.block5(h)

        h = F.relu(self.bn5(h))
        h = F.average_pooling_2d(h, h.shape[2:])

        h = self.fc7(h)

        return h

    def regularize_convs(self, p=0.5, devices=-1):
        if np.random.rand() > p:
            return

        n_out, n_in, _, _ = self.conv1.W.data.shape
        self.conv1.W.data = regularize_conv(self.conv1.W.data, self.conv1.out_size, np.sqrt(float(n_out)/n_in), devices)

        n_out, n_in, _, _ = self.trans_conv1.W.data.shape
        self.trans_conv1.W.data = regularize_conv(self.trans_conv1.W.data, self.trans_conv1.out_size, np.sqrt(float(n_out)/n_in), devices)

        n_out, n_in, _, _ = self.trans_conv2.W.data.shape
        self.trans_conv2.W.data = regularize_conv(self.trans_conv2.W.data, self.trans_conv2.out_size, np.sqrt(float(n_out)/n_in), devices)

        return


class ProcResNet274(ProcResNet):

    def __init__(self, n_class=10):
        super(ProcResNet274, self).__init__(n_class, 30, 'Residual')


class ProcResNet166(ProcResNet):

    def __init__(self, n_class=10):
        super(ProcResNet166, self).__init__(n_class, 18, 'Residual')


if __name__ == '__main__':
    import numpy as np
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    model = ProcResNet(10)
    y = model(x)