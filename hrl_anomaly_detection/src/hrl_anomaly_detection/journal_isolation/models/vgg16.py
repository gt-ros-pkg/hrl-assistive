# -*- coding: utf-8 -*-
"""VGG16 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

from keras.models import Sequential, Model
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Concatenate, concatenate
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras import regularizers
## from .imagenet_utils import decode_predictions
## from .imagenet_utils import preprocess_input
#from .imagenet_utils import _obtain_input_shape


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG16(include_top=True, include_multi_top=False, weights='imagenet', weights_file=None,
          input_tensor=None, input_shape=None, input_shape2=None,
          pooling=None, classes=1000):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    ## if not (weights in {'imagenet', None} or os.path.exists(weights)):
    ##     raise ValueError('The `weights` argument should be either '
    ##                      '`None` (random initialization), `imagenet` '
    ##                      '(pre-training on ImageNet), '
    ##                      'or the path to the weights file to be loaded.')

    ## if weights == 'imagenet' and include_top and classes != 1000:
    ##     raise ValueError('If using `weights` as imagenet with `include_top`'
    ##                      ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    model = Sequential()
    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models',
                            file_hash='6d6bbae143d832006294945121d1f1fc')
    model.load_weights(weights_path)
    sys.exit()


    # total 18 layers
    for layer in model.layers[:-2]:
        layer.trainable = False
    ## for layer in model.layers:
    ##     layer.trainable = False

    # Create model.
    print(model.summary())


    if include_multi_top:
        model.add(Flatten(name='flatten'))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu', name='fc1',
                  kernel_regularizer=regularizers.l2(0.03)))
        model.add(Dense(16, activation='relu', name='fc2',
                  kernel_regularizer=regularizers.l2(0.01)))

        sig_model = Sequential()
        sig_model.add(Dense(16, kernel_initializer='random_uniform', input_shape=input_shape2,
                            activation='tanh', name='sig_1'))#(sig_input)
        sig_model.add(Dropout(0.3))
        sig_model.add(Dense(16, kernel_initializer='random_uniform',
                            activation='tanh', name='sig_2'))
        sig_model.add(Dropout(0.3))

        merged = Concatenate()([model.output, sig_model.output]) 
        
        out = Dense(16, activation='tanh', kernel_initializer='random_uniform', name='fc3_1',
                       kernel_regularizer=regularizers.l2(0.05))(merged)
        out = Dense(classes, activation='softmax', name='fc_out')(out)
        multi_model = Model([model.input, sig_model.input], out)
        
    elif include_top:
        model.add(Flatten(name='flatten'))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu', name='fc1',
                  kernel_regularizer=regularizers.l2(0.03)))
        model.add(Dense(16, activation='relu', name='fc2',
                  kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(classes, activation='softmax', name='predictions'))
        multi_model = model
    else:
        if pooling == 'avg':
            model.add(GlobalAveragePooling2D())
        elif pooling == 'max':
            model.add(GlobalMaxPooling2D())
        multi_model = model
        ## model = Model(img_input, x, name='vgg16')


    if weights is not None:
        if include_multi_top and weights_file is not None:
            if weights_file[2] is not None:
                multi_model.load_weights(weights_file[2])
            else:
                multi_model.load_weights(weights_file[0], by_name=True)
                multi_model.load_weights(weights_file[1], by_name=True)
        elif include_top and weights_file is not None:
            multi_model.load_weights(weights_file[1], by_name=True)
            
    return multi_model


def vgg_image_top_net(input_shape, classes):

    model = Sequential()
    model.add(Flatten(name='flatten', input_shape=input_shape))
    ## model.add(Dense(1024, activation='relu', name='fc1',
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu', name='fc1',
                    kernel_regularizer=regularizers.l2(0.03)))
    #model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu', name='fc2',
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(classes, activation='softmax', name='predictions'))
    print(model.summary())

    return model
    



def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate a model's input shape.
    
    # Arguments
    input_shape: Either None (will return the default network input shape),
    or a user-provided shape to be validated.
    default_size: Default input width/height for the model.
    min_size: Minimum input width/height accepted by the model.
    data_format: Image data format to use.
    require_flatten: Whether the model is expected to
    be linked to a classifier via a Flatten layer.
    weights: One of `None` (random initialization)
    or 'imagenet' (pre-training on ImageNet).
    If weights='imagenet' input channels must be equal to 3.
    
    # Returns
    An integer shape tuple (may include None entries).
    
    # Raises
    ValueError: In case of invalid argument values.
    """
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                    (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                    (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape


        
