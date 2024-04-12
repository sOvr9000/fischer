
import numpy as np
from typing import Union
from keras.layers import Conv1D, Conv2D, Conv3D, LeakyReLU, Dropout, Add, Layer, GlobalAvgPool1D, GlobalAvgPool2D, GlobalAvgPool3D, Reshape, Dense, Activation, multiply, BatchNormalization, Attention
from keras.initializers.initializers_v2 import TruncatedNormal, HeNormal
# from tensorflow.keras.initializers import TruncatedNormal
from keras.regularizers import L1L2



__all__ = ['residual_block']



def residual_block(input_layer: Layer, filters: int, kernel_size: Union[int, tuple] = None, weight_layers_per_block: int = 1, blocks: int = 1, dropout: float = 0, leaky_alpha: float = 0.3, squeeze_factor: int = None, use_batch_normalization: bool = False, initial_weight_scale: float = 1, prepend_layer: bool = True, activation_function: str = None, kernel_regularization: float = 0.001, bias_regularization: float = 0.001, use_he_initialization: bool = False, v2: bool = True) -> Layer:
    '''
    Residual block as defined by the ResNet model architecture boosted by a squeeze and excitation block.  View https://github.com/titu1994/keras-squeeze-excite-network#readme for a visual of the combined architectures.

    `octaves` is the depth of recursion in ResNet architecture.  For example, if `octaves = 1`, then the standard ResNet module is used per block.  If `octaves = 2`, then each internal weight layer of the top-level ResNet module is a ResNet module itself.  The result of this is that training is further stabilized (?)

    If `prepend_layer = True`, then a single convolutional layer is prepended to the sequence of `padded_iterations` blocks.  This makes it easier to deal with consistent tensor shapes for implementing residuality (final conv layer needs to have the same shape as input shape to the residual block in order for the input to be added to the output of the block).
    However, if the shape is already consistent, then it is better to use `prepend_layer = False` because it reduces or fully mitigates the number of non-residual layers/blocks in the model.

    If `squeeze_factor = None`, then squeeze and excitation layers are not used.  Otherwise, the factor of dimensionality reduction by the squeeze and excitation layers is `squeeze_factor`.

    If `use_transpose = True`, then `prepend_layer` must be `True`.

    If `v2 = True`, then the ResNetV2 architecture is used instead of ResNet.

    Supports Dense layers and Conv layers separately, depending on whether `kernel_size` is defined.  If `kernel_size = None`, then `filters` is interpreted as the number of units in each Dense layer.
    '''

    if squeeze_factor is not None and (not isinstance(squeeze_factor, int) or squeeze_factor <= 1):
        raise ValueError(f'Invalid value for argument: squeeze_factor={squeeze_factor}  (Expected an integer greater than 1)')

    use_squeeze_excitation = squeeze_factor is not None

    if kernel_size is not None:
        params = {
            'filters': filters,
            'kernel_size': kernel_size,
            'padding': 'same',
            'use_bias': True,
            'kernel_regularizer': L1L2(kernel_regularization, kernel_regularization),
            'bias_regularizer': L1L2(bias_regularization, bias_regularization),
        }
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        conv_dims = len(kernel_size)
        if conv_dims == 1:
            layer_cls = Conv1D
            pool_cls = GlobalAvgPool1D
        elif conv_dims == 2:
            layer_cls = Conv2D
            pool_cls = GlobalAvgPool2D
        elif conv_dims == 3:
            layer_cls = Conv3D
            pool_cls = GlobalAvgPool3D
        else:
            raise ValueError(f'kernel_size can only have 1, 2, or 3 integers.')
    else:
        layer_cls = Dense
        params = {
            'units': filters,
            'use_bias': True,
        }

    def se_block(x):
        inp = x
        if kernel_size is None:
            filters = x.shape[-1]
            se = inp
        else:
            filters = x.shape[-1]
            se = pool_cls()(inp)
            se_shape = np.ones((conv_dims+1,), dtype=int)
            se_shape[-1] = filters
            se_shape = tuple(se_shape)
            se = Reshape(se_shape)(se)
        se = Dense(filters // squeeze_factor, kernel_initializer=new_init(), bias_initializer=new_init(), use_bias=True)(se)
        se = LeakyReLU(leaky_alpha)(se)
        se = Dense(filters, kernel_initializer=new_init(), bias_initializer=new_init(), use_bias=True, activation='sigmoid')(se)
        # se = Activation('sigmoid')(se)
        x = multiply([inp, se])
        return x

    def new_truncated_normal():
        return TruncatedNormal(mean=0, stddev=initial_weight_scale, seed=None)

    def new_he_normal():
        return HeNormal(seed=None)

    def new_init():
        if use_he_initialization:
            return new_he_normal()
        return new_truncated_normal()

    if activation_function is None:
        act_params = [leaky_alpha]
        activation_function = LeakyReLU
    else:
        act_params = [activation_function]
        activation_function = Activation

    def res_module(x, do_first_activation: bool = True):
        inp = x
        if v2 and do_first_activation:
            x = activation_function(*act_params)(x)
        for k in range(weight_layers_per_block):
            x = layer_cls(**params, kernel_initializer=new_init(), bias_initializer=new_init())(x)
            if dropout > 0:
                x = Dropout(dropout)(x)
            if use_batch_normalization:
                x = BatchNormalization()(x)
            if not v2:
                if k < weight_layers_per_block - 1:
                    # don't apply activation on the final layer in the block so that the residual can be added to the input before the activation is applied
                    x = activation_function(*act_params)(x)
        if use_squeeze_excitation:
            x = se_block(x)
        x = Add()((x, inp))
        if use_batch_normalization:
            x = BatchNormalization()(x)
        if not v2:
            x = activation_function(*act_params)(x)
        return x

    x = input_layer
    if prepend_layer:
        x = layer_cls(**params, kernel_initializer=new_init(), bias_initializer=new_init())(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        x = activation_function(*act_params)(x)
    for b in range(blocks):
        x = res_module(x, do_first_activation=b>0)
    if v2:
        x = activation_function(*act_params)(x)

    return x


