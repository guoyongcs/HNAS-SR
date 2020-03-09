import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/home/young/sr/srdata',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-320/321-800',
                    help='train_w/train_controller data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
# parser.add_argument('--model', default='EDSR',
parser.add_argument('--model', default='ENAS',
                    help='model name')

parser.add_argument('--upsampling_Pos', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
# parser.add_argument('--lr', type=float, default=1e-4,
parser.add_argument('--lr', type=float, default=1e-2,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='100',
                    help='learning rate decay type')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type: MultiStep | cosine')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--controller_optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=20,
                    help='how many batches to wait before logging training status')
parser.add_argument('--controller_print_every', type=int, default=30,
                    help='how many batches to wait before logging training status')
parser.add_argument('--repeat_data_time', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')



####### controller
parser.add_argument('--controller_learning_rate', type=float, default=3e-4,
                    help='learning rate for controller')
parser.add_argument('--controller_weight_decay', type=float, default=5e-4,
                    help='learning rate for controller')
parser.add_argument('--controller_betas', type=tuple, default=(0.5, 0.999),
                    help='ADAM beta')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.1, 0.1], help='coefficient for entropy: [normal, upsampling]')
parser.add_argument('--flops_scale', type=float, default=0.5,
                    help='flops_scale for reward')
parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--controller_start_training', type=int, default=0, help='Epoch that the training of controller starts')


parser.add_argument('--controller_type', type=str, default='SAMPLE', help='SAMPLE | LSTM')
parser.add_argument('--controller_hid', type=int, default=100, help='temperature for lstm')
parser.add_argument('--controller_temperature', type=float, default=None, help='temperature for lstm')
parser.add_argument('--controller_tanh_constant', type=float, default=None, help='tanh constant for lstm')

parser.add_argument('--lstm_num_layers', type=int, default=1, help='number of layers in lstm')
parser.add_argument('--controller_op_tanh_upsampling', type=float, default=2.5, help='coefficient for entropy')
# scheduler restart
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--T_mul', type=float, default=2.0, help='multiplier for cycle')
parser.add_argument('--T0', type=int, default=10, help='The maximum number of epochs within the first cycle')
parser.add_argument('--store', type=int, default=1, help='Whether to store the model')
parser.add_argument('--pruner_learning_rate', type=float, default=3e-4, help='learning rate for pruner')
parser.add_argument('--pruner_weight_decay', type=float, default=5e-4, help='learning rate for pruner')
parser.add_argument('--edge_hid', type=int, default=100, help='edge hidden dimension')
parser.add_argument('--pruner_nfeat', type=int, default=1024, help='feature dimension of each node')
parser.add_argument('--pruner_nhid', type=int, default=100, help='hidden dimension')
parser.add_argument('--pruner_dropout', type=float, default=0, help='dropout rate for pruner')
parser.add_argument('--pruner_normalize', action='store_true', default=False, help='use normalize in GCN')
parser.add_argument('--loose_end', action='store_true', default=False, help='loose_end')
parser.add_argument('--split_fc', action='store_true', default=False, help='split_fc')
parser.add_argument('--train_controller', default=True, help='train_controller')
parser.add_argument('--init_channels', type=int, default=32, help='number of init channels')
parser.add_argument('--layers', type=int, default=16, help='total number of layers')
parser.add_argument('--sampling', type=int, default=10, help='when test and save the arch, the number to sample')
parser.add_argument('--sampling_epoch_margin', type=int, default=5, help='the margin of epoch to sample')
parser.add_argument('--genotype', type=str, default='DARTS', help='which architecture to use')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

