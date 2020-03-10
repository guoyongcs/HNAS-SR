from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat upsampling upsampling_concat')
Genotype_multi = namedtuple('Genotype', 'normal_bottom normal_concat_bottom upsampling_bottom upsampling_concat_bottom \
                                         normal_mid normal_concat_mid upsampling_mid upsampling_concat_mid \
                                         normal_top normal_concat_top')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

LSTM_PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

COMPACT_PRIMITIVES = [
   # 'up_and_down',
    'rcab',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'

]

COMPACT_PRIMITIVES_UPSAMPLING = [
    'sub_pixel',
    'deconvolution',
    'bilinear',
    'nearest',
    'area',
]


PRUNER_PRIMITIVES = [
    'none',
    'same',
    'skip_connect',
]


SR_RCAN_MODEL = Genotype(
    normal=[('none', 0, 2), ('rcab', 1, 2), ('none', 0, 3), ('rcab', 2, 3), ('none', 0, 4),
            ('rcab', 3, 4), ('rcab', 4, 5), ('skip_connect', 1, 5)], normal_concat=[5],
    upsampling=[('none', 0, 2), ('skip_connect', 1, 2), ('none',0, 3), ('skip_connect', 2, 3), ('none', 0, 4),
                ('skip_connect', 3, 4),('upprojectnone', 0, 5), ('manual_sub_pixel', 4, 5)], upsampling_concat=[5])

SR_PLAIN_MODEL = Genotype(
    normal=[('none', 0, 2), ('conv_3x3_no_relu', 1, 2), ('conv_3x3', 2, 3), ('none', 0, 3), ('conv_3x3_no_relu', 3, 4),
            ('none', 0, 4), ('conv_3x3', 4, 5), ('none', 0, 5)], normal_concat=[5],
    upsampling=[('none', 0, 2), ('skip_connect', 1, 2), ('none',0, 3), ('skip_connect', 2, 3), ('none', 0, 4),
                ('skip_connect', 3, 4),('upprojectnone', 0, 5), ('manual_sub_pixel', 4, 5) ], upsampling_concat=[5])

SR_EDSR_MODEL = Genotype(
    normal=[('none', 0, 2), ('conv_3x3_no_relu', 1, 2), ('conv_3x3', 2, 3), ('skip_connect', 1, 3), ('conv_3x3_no_relu', 3, 4),
            ('none', 0, 4), ('conv_3x3', 4, 5), ('skip_connect', 3, 5)], normal_concat=[5],
    upsampling=[('none', 0, 2), ('skip_connect', 1, 2), ('none',0, 3), ('skip_connect', 2, 3), ('none', 0, 4),
                ('skip_connect', 3, 4),('upprojectnone', 0, 5), ('manual_sub_pixel', 4, 5) ], upsampling_concat=[5])



# # upsampling_position:10
HNAS_A = Genotype(normal=[('skip_connect', 1, 2), ('sep_conv_5x5', 1, 2), ('dil_conv_3x3', 0, 3), ('dil_conv_5x5', 1, 3), ('skip_connect', 1, 4), ('dil_conv_3x3', 1, 4), ('dil_conv_5x5', 1, 5), ('sep_conv_3x3', 4, 5)], normal_concat=range(2, 6), upsampling=[('deconvolution', 0, 2), ('bilinear', 0, 2), ('area', 1, 3), ('deconvolution', 0, 3), ('area', 3, 4), ('bilinear', 0, 4), ('bilinear', 1, 5), ('deconvolution', 3, 5)], upsampling_concat=range(2, 6))

# upsampling_position:12
HNAS_B = Genotype(normal=[('dil_conv_3x3', 0, 2), ('rcab', 1, 2), ('dil_conv_5x5', 0, 3), ('sep_conv_5x5', 1, 3), ('sep_conv_3x3', 1, 4), ('dil_conv_3x3', 0, 4), ('sep_conv_3x3', 2, 5), ('dil_conv_5x5', 3, 5)], normal_concat=range(2, 6), upsampling=[('sub_pixel', 1, 2), ('bilinear', 1, 2), ('bilinear', 1, 3), ('sub_pixel', 0, 3), ('sub_pixel', 3, 4), ('sub_pixel', 3, 4), ('deconvolution', 0, 5), ('nearest', 1, 5)], upsampling_concat=range(2, 6))

# upsampling_position:12
# HNAS_C = Genotype(normal=[('skip_connect', 0, 2), ('rcab', 1, 2), ('dil_conv_5x5', 0, 3), ('sep_conv_3x3', 0, 3), ('sep_conv_5x5', 1, 4), ('rcab', 2, 4), ('dil_conv_5x5', 0, 5), ('sep_conv_5x5', 2, 5)], normal_concat=range(2, 6), upsampling=[('nearest', 1, 2), ('deconvolution', 0, 2), ('bilinear', 0, 3), ('deconvolution', 0, 3), ('area', 1, 4), ('sub_pixel', 3, 4), ('area', 2, 5), ('bilinear', 0, 5)], upsampling_concat=range(2, 6))
HNAS_C = Genotype(normal=[('rcab', 1, 2), ('skip_connect', 1, 2), ('dil_conv_3x3', 1, 3), ('skip_connect', 0, 3), ('dil_conv_5x5', 0, 4), ('rcab', 3, 4), ('dil_conv_5x5', 4, 5), ('dil_conv_3x3', 0, 5)], normal_concat=range(2, 6), upsampling=[('sub_pixel', 0, 2), ('nearest', 1, 2), ('deconvolution', 2, 3), ('nearest', 0, 3), ('deconvolution', 3, 4), ('sub_pixel', 2, 4), ('bilinear', 0, 5), ('area', 0, 5)], upsampling_concat=range(2, 6))





