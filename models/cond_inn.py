import torch
import numpy as np
from torch import optim
from torch import nn
from models.flow import get_latent_cnf
from models.flow import get_hyper_cnf
from utils import truncated_normal, standard_normal_logprob, standard_laplace_logprob
from torch.nn import init
from torch.distributions.laplace import Laplace
from torchvision.models.resnet import resnet50, resnet101
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import mmfp_utils

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from models.reshapes import haar_multiplex_layer
import models.subnet_coupling as subnet_coupling
import models.config as c


# the reason the subnet init is needed, is that with uninitalized
# weights, the numerical jacobian check gives inf, nan, etc,

# https://github.com/VLL-HD/FrEIA/blob/451286ffae2bfc42f6b0baaba47f3d4583258599/tests/test_reversible_graph_net.py


def subnet_initialization(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        m.weight.data *= 0.3
        m.bias.data *= 0.1


def F_conv(cin, cout):
    '''Simple convolutional subnetwork'''
    print(cin, cout)
    net = nn.Sequential(nn.Conv2d(cin, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, cout, 3, padding=1))
    net.apply(subnet_initialization)
    return net


def F_fully_connected(cin, cout):
    '''Simple fully connected subnetwork'''
    print(cin, cout)
    net = nn.Sequential(nn.Linear(cin, 128),
                        nn.ReLU(),
                        nn.Linear(128, cout))
    net.apply(subnet_initialization)
    return net

def subnet_fc(c_in, c_out):
    print(c_in, c_out)
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                        nn.Linear(512,  c_out))

def subnet_conv(c_in, c_out):
    print(c_in, c_out)
    return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
                        nn.Conv2d(256,  c_out, 3, padding=1))

def subnet_conv_1x1(c_in, c_out):
    print(c_in, c_out)
    return nn.Sequential(nn.Conv2d(c_in, 256,   1), nn.ReLU(),
                        nn.Conv2d(256,  c_out, 1))

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=None):
    if padding is None:
        padding_inside = (kernel_size-1)//2
    else:
        padding_inside = padding
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=padding_inside, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=padding_inside, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias=True):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size-1)//2, bias=bias),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size-1)//2, bias=bias),
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                           stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):
    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


def init_deconv_bilinear(weight):
    f_shape = weight.size()
    heigh, width = f_shape[-2], f_shape[-1]
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weight.data.fill_(0.)
    for i in range(f_shape[0]):
        for j in range(f_shape[1]):
            weight.data[i, j, :, :] = torch.from_numpy(bilinear)


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class CondINN(nn.Module):
    def __init__(self, args, img_dims=None):
        super(CondINN, self).__init__()
        self.args = args
        self.model_dims = img_dims
        self.feature_channels = 256
        self.num_classes = args.num_classes + 1
        self.fc_cond_length = 256
        self.ndim_x = 3 * 64 * 64

        self.input_node = Ff.InputNode(3, 64, 64, name='inp_points')
        self.conditions = [Ff.ConditionNode(self.fc_cond_length + self.num_classes, 64, 64, name='cond-0'),
                        Ff.ConditionNode(self.fc_cond_length, name='cond-1')]

        self.nodes = []

        # input nodes
        self.nodes.append(self.input_node)

        for k in range(4):
            conv = Ff.Node(self.nodes[-1],
                        Fm.GLOWCouplingBlock,
                        {'subnet_constructor': subnet_conv, 'clamp': 1.2},
                        conditions=self.conditions[0],
                        name=F'conv{k}::c1')
            self.nodes.append(conv)
            permute = Ff.Node(self.nodes[-1], Fm.PermuteRandom,
                        {'seed': k}, name=F'permute_{k}')
            self.nodes.append(permute)

        self.nodes.append(Ff.Node(self.nodes[-1], Fm.IRevNetDownsampling, {}))

        for k in range(2):
            if k % 2 == 0:
                subnet = subnet_conv_1x1
            else:
                subnet = subnet_conv

            linear = Ff.Node(self.nodes[-1],
                            Fm.GLOWCouplingBlock,
                            {'subnet_constructor': subnet, 'clamp': 1.2},
                            #conditions=self.conditions[0],
                            name=F'conv_low_res_{k}')
            self.nodes.append(linear)
            permute = Ff.Node(self.nodes[-1], Fm.PermuteRandom,
                        {'seed': k}, name=F'permute_low_res_{k}')
            self.nodes.append(permute)

        self.nodes.append(Ff.Node(self.nodes[-1], Fm.Flatten, {}, name='flatten'))
        split_node = Ff.Node(self.nodes[-1],
                            Fm.Split,
                            {'section_sizes':(self.ndim_x // 4, 3 * self.ndim_x // 4), 'dim':0},
                            name='split')
        self.nodes.append(split_node)

        # Fully connected part
        for k in range(3):
            self.nodes.append(Ff.Node(self.nodes[-1],
                                Fm.GLOWCouplingBlock,
                                {'subnet_constructor':subnet_fc, 'clamp':2.0},
                                conditions=self.conditions[1],
                                name=F'fully_connected_{k}'))
            self.nodes.append(Ff.Node(self.nodes[-1],
                                Fm.PermuteRandom,
                                {'seed':k},
                                name=F'permute_{k}'))

        # Concatenate the fully connected part and the skip connection to get a single output
        self.nodes.append(Ff.Node([self.nodes[-1].out0, split_node.out1],
                            Fm.Concat1d, {'dim':0}, name='concat'))
        self.nodes.append(Ff.OutputNode(self.nodes[-1], name='output'))

        """
        # 2x64x64 px
        self._add_conditioned_section(self.nodes, depth=4, channels_in=2,
                                      channels=32, cond_level=0, condition=self.condition)
        self._add_split_downsample(self.nodes, split=False,
                                   downsample='reshape', channels_in=2, channels=64)


        # 8x32x32 px
        self._add_conditioned_section(self.nodes, depth=6, channels_in=8,
                                      channels=64, cond_level=1, condition=self.conditions[0])
        self._add_split_downsample(self.nodes, split=(
            16, 16), downsample='reshape', channels_in=8, channels=128)

        # 16x16x16 px
        self._add_conditioned_section(self.nodes, depth=6, channels_in=16,
                                      channels=128, cond_level=2, condition=self.conditions[0])
        self._add_split_downsample(self.nodes, split=(
            32, 32), downsample='reshape', channels_in=16, channels=256)

        # 32x8x8 px
        self._add_conditioned_section(self.nodes, depth=6, channels_in=32,
                                      channels=256, cond_level=3, condition=self.conditions[0])
        self._add_split_downsample(self.nodes, split=(32, 3*32),
                                   downsample='haar', channels_in=32, channels=256)

        # 32x4x4 = 512 px
        self._add_fc_section(self.nodes, self.fc_cond_length,
                             condition=self.conditions[1])
        """

        #self.output_node = Ff.OutputNode(self.nodes[-1], name='out')

        # output nodes
        #self.nodes.append(self.output_node)

        self.cinn = Ff.GraphINN(self.nodes + self.conditions, verbose=False)
        self.cinn = self.cinn.cuda()

        def init_model(model):
            for key, param in model.named_parameters():
                split = key.split('.')
                if param.requires_grad:
                    param.data = c.init_scale * \
                        torch.randn(param.data.shape).cuda()
                    # last convolution in the coeff func
                    if len(split) > 3 and split[3][-1] == '3':
                        param.data.fill_(0.)

        init_model(self.cinn)


    def _cond_subnet(self, level, c_out, extra_conv=False):
        c_intern = [self.feature_channels, 128, 128, 256]
        modules = []

        for i in range(level):
            modules.extend([nn.Conv2d(c_intern[i], c_intern[i+1], 3, stride=2, padding=1),
                            nn.LeakyReLU()])

        if extra_conv:
            modules.extend([
                nn.Conv2d(c_intern[level], 128, 3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(128, 2*c_out, 3, padding=1),
            ])
        else:
            modules.append(nn.Conv2d(c_intern[level], 2*c_out, 3, padding=1))

        modules.append(nn.BatchNorm2d(2*c_out))

        return nn.Sequential(*modules)

    def _add_conditioned_section(self, nodes, depth, channels_in, channels, cond_level, condition=None):

        for k in range(depth):
            nodes.append(Ff.Node([nodes[-1].out0],
                                 subnet_coupling.subnet_coupling_layer,
                                 {'clamp': c.clamping, 'F_class': F_conv,
                                  'subnet': self._cond_subnet(cond_level, channels//2), 'sub_len': channels,
                                  'F_args': {'leaky_slope': 5e-2, 'channels_hidden': channels}},
                                 conditions=[condition], name=F'conv_{k}'))

            nodes.append(Ff.Node([nodes[-1].out0], Fm.Fixed1x1Conv,
                                 {'M': random_orthog(channels_in)}))

    def _add_split_downsample(self, nodes, split, downsample, channels_in, channels):
        if downsample == 'haar':
            nodes.append(Ff.Node([nodes[-1].out0], haar_multiplex_layer,
                                 {'rebalance': 0.5, 'order_by_wavelet': True}, name='haar'))
        if downsample == 'reshape':
            nodes.append(
                Ff.Node([nodes[-1].out0], Fm.IRevNetDownSampling, {}, name='reshape'))

        for i in range(2):
            nodes.append(Ff.Node([nodes[-1].out0], Fm.Fixed1x1Conv,
                                 {'M': random_orthog(channels_in*4)}))
            nodes.append(Ff.Node([nodes[-1].out0],
                                 Fm.GLOWCouplingBlock,
                                 {'clamp': c.clamping, 'F_class': F_conv,
                                  'F_args': {'kernel_size': 1, 'leaky_slope': 1e-2, 'channels_hidden': channels}},
                                 conditions=[]))

        if split:
            nodes.append(Ff.Node([nodes[-1].out0], Fm.Split,
                                 {'split_size_or_sections': split, 'dim': 0}, name='split'))

            output = Ff.Node([nodes[-1].out1], Fm.Flatten, {}, name='flatten')
            nodes.insert(-2, output)
            nodes.insert(-2, Fm.OutputNode([output.out0], name='out'))

    def _add_fc_section(self, nodes, fc_cond_length, n_blocks_fc=8, condition=None):
        nodes.append(Ff.Node([nodes[-1].out0], Fm.Flatten, {}, name='flatten'))
        for k in range(n_blocks_fc):
            nodes.append(Ff.Node([nodes[-1].out0], Fm.PermuteRandom,
                                 {'seed': k}, name=F'permute_{k}'))
            nodes.append(Ff.Node([nodes[-1].out0], Fm.GLOWCouplingBlock,
                                 {'clamp': c.clamping, 'F_class': F_fully_connected,
                                  'F_args': {'internal_size': fc_cond_length}},
                                 conditions=[condition], name=F'fc_{k}'))

    def forward(self, x, cond, rev=False):
        # if load_inn_only:
        #    self.cinn.load_state_dict(torch.load(load_inn_only)['net'])

        if rev is False:
            z, log_jac_det = self.cinn(x, c=cond, rev=rev)
        else:
            z, log_jac_det = self.cinn(x, c=cond, rev=rev)

        return z, log_jac_det


class CondINNWrapper(nn.Module):
    def __init__(self, args, img_dims=None):
        super(CondINNWrapper, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.backbone = ConditionalBackbone(args)
        self.cinn = CondINN(args)
        self.gpu = args.gpu
        self.logprob_type = args.logprob_type
        self.ce_loss = nn.CrossEntropyLoss()
        self.fc_cond_length = 256

        upsample_layers = []
        for i in range(4):
            upsample_layers.append(nn.ConvTranspose2d(self.fc_cond_length + args.num_classes + 1, self.fc_cond_length + args.num_classes + 1, 3, stride=2, padding=1))
        self.upsample_layers = ListModule(*upsample_layers)

        self.fc_cond_net = nn.Sequential(*[nn.Conv2d(337, 128, 3, stride=2, padding=1), # 256x64x64
                                        nn.LeakyReLU(),
                                        nn.Conv2d(128, 64, 3, stride=2, padding=1), # 128x32x32
                                        nn.LeakyReLU(),
                                        nn.Conv2d(64, 256, 3, stride=2, padding=1), # 64x16x16
                                        nn.LeakyReLU(),
                                        nn.AvgPool2d(8)], # 256x16x16
                                        nn.BatchNorm2d(self.fc_cond_length)
                                        ) # 256x8x8

        self.optimizer = self.make_optimizer(args)

    def make_optimizer(self, args):
        def _get_opt_(params):
            if args.optimizer == 'adam':
                optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                                       weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    params, lr=args.lr, momentum=args.momentum)
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer
        opt = _get_opt_(list(self.backbone.parameters()) +
                        list(self.cinn.parameters()) +
                        list(self.upsample_layers.parameters()) +
                        list(self.fc_cond_net.parameters()))
        return opt

    def get_optimizer(self):
        return self.optimizer

    def conditioning(self, x, cond):
        cond = cond.cuda()

        net_outputs = self.backbone(x, cond)

        # 337x8x8
        net_outputs['3'] += self.upsample_layers[3](net_outputs['pool'], output_size=net_outputs['3'].size())
        # 337x16x16
        net_outputs['2'] += self.upsample_layers[2](net_outputs['3'], output_size=net_outputs['2'].size())
        # 337x32x32
        net_outputs['1'] += self.upsample_layers[1](net_outputs['2'], output_size=net_outputs['1'].size())
        # 337x64x64
        net_outputs['0'] += self.upsample_layers[0](net_outputs['1'], output_size=net_outputs['0'].size())
        
        cond0 = net_outputs['0']

        cond_net_outputs = self.fc_cond_net(cond0)

        cond1 = cond_net_outputs.view(cond_net_outputs.shape[0], -1)

        # 64x256
        return [cond0, cond1]

    def forward(self, x, y, cond, writer=None):

        self.optimizer.zero_grad()

        batch_size = x.size(0)

        conditions = self.conditioning(x, cond)

        y = y.unsqueeze(1).repeat(1, 3, 1, 1)

        z, log_jac_det = self.cinn(y, conditions)

        loss = 0.5*torch.sum(z**2, 1) - log_jac_det
        loss = loss.mean() / batch_size
        loss.backward()

        self.optimizer.step()

        #import pdb; pdb.set_trace()
        print('z-min: ', z.min())
        print('z-max: ', z.max())
        print('z-mean: ', z.mean())

        # z = z.mean() + torch.empty(z.shape).normal_(mean=0, std=0.005).cuda()
        sample, _ = self.cinn(z, conditions, rev=True)
        print(sample.shape)

        #z = z.view(z.size(0), 3, 64, 64)

        return sample, loss

    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    @staticmethod
    def sample_laplace(size, gpu=None):
        m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        y = m.sample(sample_shape=torch.Size(
            [size[0], size[1], size[2]])).float().squeeze(3)
        y = y if gpu is None else y.cuda(gpu)
        return y

    def decode(self, x, cond):
        conditions = self.conditioning(x, cond)
        #y = y.unsqueeze(1).repeat(1, 3, 1, 1)
        z = torch.randn((x.size(0), 3 * 64 * 64)).cuda()
        #conditions[0] = conditions[0].repeat(1, 1, 1, 1)
        #conditions[1] = conditions[1].repeat(1, 1, 1, 1)
        x, _ = self.cinn(z, conditions, rev=True)

        return x

    def get_logprob(self, x, y_in, class_conditional):
        batch_size = x.size(0)
        class_conditional = class_conditional.cuda()
        target_networks_weights = self.hyper(x, class_conditional)
        # 2*128 + 128 + 128 + 128 + 128
        # 128*2 + 2 + 2 + 2 + 2
        # + 81
        # = 1131

        # Loss
        y, delta_log_py = self.point_cnf(y_in, target_networks_weights, torch.zeros(
            batch_size, y_in.size(1), 1).to(y_in))
        if self.logprob_type == "Laplace":
            log_py = standard_laplace_logprob(y)
        if self.logprob_type == "Normal":
            log_py = standard_normal_logprob(y)

        batch_log_py = log_py.sum(dim=2)
        batch_log_px = batch_log_py - delta_log_py.sum(dim=2)
        log_py = log_py.view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, y.size(1), 1).sum(1)
        log_px = log_py - delta_log_py

        return log_py, log_px, (batch_log_py, batch_log_px)


class ConditionalBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.fpn = ResNet101FPN()

    def forward(self, x, cond):
        fpn_outputs = self.fpn(x)
        _, _, dim0, dim1 = fpn_outputs['0'].shape
        cond0 = cond.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dim0, dim1)
        fpn_outputs['0'] = torch.cat([fpn_outputs['0'], cond0], dim=1)

        _, _, dim0, dim1 = fpn_outputs['1'].shape
        cond1 = cond.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dim0, dim1)
        fpn_outputs['1'] = torch.cat([fpn_outputs['1'], cond1], dim=1)

        _, _, dim0, dim1 = fpn_outputs['2'].shape
        cond2 = cond.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dim0, dim1)
        fpn_outputs['2'] = torch.cat([fpn_outputs['2'], cond2], dim=1)

        _, _, dim0, dim1 = fpn_outputs['3'].shape
        cond3 = cond.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dim0, dim1)
        fpn_outputs['3'] = torch.cat([fpn_outputs['3'], cond3], dim=1)

        _, _, dim0, dim1 = fpn_outputs['pool'].shape
        cond_pool = cond.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dim0, dim1)
        fpn_outputs['pool'] = torch.cat([fpn_outputs['pool'], cond_pool], dim=1)

        return fpn_outputs

class ResNet101FPN(nn.Module):
    def __init__(self):
        super(ResNet101FPN, self).__init__()

        self.backbone = resnet_fpn_backbone(
            'resnet101', pretrained=True, trainable_layers=3)
        # num_features = self.net.fc.in_features
        # self.net.fc = nn.Linear(num_features, 1024)
        # self.net = self.net.cuda()

    def forward(self, x):
        fpn_outputs = self.backbone(x)
        return fpn_outputs