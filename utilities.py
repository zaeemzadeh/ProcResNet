import chainer
import cupy as cp


class ConvRegularization:
    name = 'normalize_grads'
    call_for_each_param = False

    def __init__(self, p=0.5, devices=-1):
        self.devices = devices
        self.p = p

    def __call__(self, opt):
        predictor_net = opt.target.predictor
        if callable(getattr(predictor_net, "regularize_convs", None)):
            predictor_net.regularize_convs(self.p, self.devices)


def regularize_conv(kernel, input_shape, clip_to, devices=-1):

    kernel = cp.transpose(kernel, (3, 2, 1, 0)) # tensor flow format

    kernel = cp.rot90(kernel, k=2, axes=(0, 1)) # normalizing the back prop
    kernel = cp.transpose(kernel, (0, 1, 3, 2))

    kernel = cp_regularize_conv(kernel, input_shape, clip_to, devices)

    kernel = cp.transpose(kernel, (0, 1, 3, 2))
    kernel = cp.rot90(kernel, k=2, axes=(0, 1))

    kernel = cp.transpose(kernel, (3, 2, 1, 0)) # back to chainer format
    return kernel


def cp_regularize_conv(kernel, input_shape, clip_to, devices=-1):

    A = cp.fft.fft2(kernel, input_shape, axes=(0, 1))
    # complex SVD using real formulation https://www.osti.gov/servlets/purl/756121
    A_resh = cp.reshape(A, (-1, A.shape[2], A.shape[3]))
    R_resh = A_resh.real
    I_resh = A_resh.imag

    upper_batch = cp.concatenate((R_resh, I_resh), axis=2)
    lower_batch = cp.concatenate((-I_resh, R_resh), axis=2)
    K_batch = cp.concatenate((upper_batch, lower_batch), axis=1)

    KTK = cp.matmul(cp.transpose(K_batch, axes=(0, 2, 1)), K_batch)

    if isinstance(devices, int):
        _, sKTKinv = batch_sqrtm(KTK, numIters=10, reg=1.)
    elif isinstance(devices, dict):
        if len(devices)==1:
            _, sKTKinv = batch_sqrtm(KTK, numIters=10, reg=1.)
        else:
            sKTKinv = batch_sqrtm_multi_gpu(KTK, devices, numIters=10, reg=1.)


    K_tran_batch = cp.squeeze(cp.matmul(K_batch, sKTKinv)) * clip_to #* np.sqrt(1./2.)

    if len(K_tran_batch.shape) == 2:
        K_tran_batch = K_tran_batch[cp.newaxis, :, :]

    _, M, N = K_tran_batch.shape

    A_tran = K_tran_batch[:, 0:M / 2, 0:N / 2] - 1j * K_tran_batch[:, 0:M / 2, -N / 2:]
    A_tran = cp.reshape(A_tran, A.shape).astype(cp.complex64)

    clipped_kernel = cp.fft.ifft2(A_tran, axes=(0, 1)).real
    return clipped_kernel[cp.ix_(*[range(d) for d in kernel.shape])]


def batch_sqrtm_multi_gpu(KTK, devices, numIters=10, reg=1.):
    #dividing the data between gpus
    KTK_d = []
    num_devices = len(devices)
    batch_size = len(KTK)//num_devices
    for b in range(num_devices):
        device = list(devices.values())[b]
        batch_start = b * batch_size
        batch_end = min((b+1) * batch_size, len(KTK))
        if b == (num_devices-1):
            batch_end = len(KTK)
        with cp.cuda.Device(chainer.backends.cuda.get_device_from_array(KTK)):
            KTK_d.append(chainer.backends.cuda.to_gpu(KTK[batch_start:batch_end], device))

    # doing sqrt in each device
    for K in KTK_d:
        with chainer.backends.cuda.get_device_from_array(K):
            _, K.real = batch_sqrtm(K, numIters, reg)

    # collecting data
    for d in range(num_devices):
        with chainer.backends.cuda.get_device_from_array(KTK_d[d]):
            KTK_d[d] = chainer.backends.cuda.to_gpu(KTK_d[d], devices['main'])
    sKTKinv = cp.concatenate(KTK_d)

    return sKTKinv


def batch_sqrtm(A, numIters = 20, reg = 2.0):
    """
    Batch matrix root via Newton-Schulz iterations
    from: https://github.com/BorisMuzellec/EllipticalEmbeddings/blob/master/utils.py
    """
    batchSize = A.shape[0]
    dim = A.shape[1]
    #Renormalize so that the each matrix has a norm lesser than 1/reg, but only normalize when necessary
    normA = reg * cp.linalg.norm(A, axis=(1, 2))
    renorm_factor = cp.ones_like(normA)
    renorm_factor[cp.where(normA > 1.0)] = normA[cp.where(normA > 1.0)]
    renorm_factor = renorm_factor.reshape(batchSize, 1, 1)

    Y = cp.divide(A, renorm_factor)
    I = cp.eye(dim).reshape(1, dim, dim).repeat(batchSize, axis=0)
    Z = cp.eye(dim).reshape(1, dim, dim).repeat(batchSize, axis=0)
    for i in range(numIters):
        T = 0.5 * (3.0 * I - cp.matmul(Z, Y))
        Y = cp.matmul(Y, T)
        Z = cp.matmul(T, Z)
    sA = Y * cp.sqrt(renorm_factor)
    sAinv = Z / cp.sqrt(renorm_factor)
    return sA, sAinv