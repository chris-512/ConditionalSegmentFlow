import warnings

import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from reshapes import haar_multiplex_layer
import subnet_coupling
import config as c
import data

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
    net = nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, cout, 3, padding=1))
    net.apply(subnet_initialization)
    return net


def F_fully_connected(cin, cout):
    '''Simple fully connected subnetwork'''
    net = nn.Sequential(nn.Linear(cin, 128),
                        nn.ReLU(),
                        nn.Linear(128, cout))
    net.apply(subnet_initialization)
    return net


def random_orthog(n):
    w = np.random.randn(n, n)
    w = w + w.T
    w, S, V = np.linalg.svd(w)
    return torch.FloatTensor(w)


class CondINN(nn.Module):
    def __init__(self, args, img_dims=None):
        self.args = args
        self.model_dims = img_dims
        self.feature_channels = 256
        self.num_classes = 81
        self.fc_cond_length = 512 + self.num_classes

        self.input_node = Ff.InputNode(2000, 2, name='inp_points')
        self.conditions = [Ff.ConditionNode(self.feature_channels, self.model_dims[0], self.model_dims[1], name='cond-0'),
                           Ff.ConditionNode(self.fc_cond_length, name='cond-1')]

        self.nodes = []

        # input nodes
        self.nodes.append(self.input_node)

        # 2x64x64 px
        self._add_conditioned_section(self.nodes, depth=4, channels_in=2,
                                      channels=32, cond_level=0, condition=self.conditions[0])
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

        self.output_node = Ff.OutputNode([self.nodes[-1].out0], name='out')

        # output nodes
        self.nodes.append(self.output_node)

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

    def _fc_cond_net(self):

        return nn.Sequential(*[
            nn.Conv2d(self.feature_channels, 128, 3,
                      stride=2, padding=1),  # 32 x 32
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 16 x 16
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # 8 x 8
            nn.LeakyReLU(),
            nn.Conv2d(256, self.fc_cond_length, 3,
                      stride=2, padding=1),  # 4 x 4
            nn.AvgPool2d(4),
            nn.BatchNorm2d(self.fc_cond_length),
        ])

    def forward(self, x, cond):
        self.cinn = Ff.GraphINN(self.nodes + self.conditions, verbose=False)
        self.cinn = self.net.cuda()

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

        # if load_inn_only:
        #    self.cinn.load_state_dict(torch.load(load_inn_only)['net'])

        return self.cinn(x, c=[cond])


class CondINNWrapper(nn.Module):
    def __init__(self, args, feature_network, img_dims=None):
        super().__init__()

        self.img_dims = img_dims
        self.inn = CondINN(args, img_dims=img_dims)
        self.feature_network = feature_network
        self.fc_cond_network = self.inn._fc_cond_net

        self._make_optim()

    def forward(self, x, points, class_cond):

        self.optim.zero_grad()
        self.feature_optim.zero_grad()

        x = F.interpolate(x, size=self.img_dims)
        points += 5e-2 * torch.cuda.FloatTensor(points.shape).normal_()

        self.feature_network = self.feature_network.cuda()
        self.fc_cond_network = self.fc_cond_network.cuda()

        if c.end_to_end:
            features = self.feature_network.features(x)
            features = features[:, :, 1:-1, 1:-1]
        else:
            with torch.no_grad():
                features = self.feature_network.features(x)
                features = features[:, :, 1:-1, 1:-1]

        cond_with_class = torch.cat(
            [self.fc_cond_network(features).squeeze(), class_cond], dim=1)
        cond = [features, cond_with_class]

        z = self.inn(points, cond)
        zz = sum(torch.sum(o**2, dim=1) for o in z)
        jac = self.inn.jacobian(run_forward=False)

        neglog_likelihood = 0.5 * zz - jac
        loss = torch.mean(neglog_likelihood)
        loss.backward()

        self.optim.step()
        # self.weight_scheduler.step()
        self.feature_optim.step()
        # self.feature_scheduler.step()

        return zz, jac

    def reverse_sample(self, z, cond):
        return self.inn(z, cond, rev=True)

    def _make_optim(self):

        sched_factor = 0.2
        sched_patience = 8
        sched_trehsh = 0.001
        sched_cooldown = 2

        trainable_params = (list(filter(lambda p: p.requires_grad,  self.inn.parameters()))
                            + list(self.fc_cond_network.parameters()))

        feature_params = list(self.feature_network.parameters())

        self.optim = torch.optim.Adam(
            trainable_params, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
        self.weight_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim,
                                                                           factor=sched_factor,
                                                                           patience=sched_patience,
                                                                           threshold=sched_trehsh,
                                                                           min_lr=0, eps=1e-08,
                                                                           cooldown=sched_cooldown,
                                                                           verbose=True)

        self.feature_optim = torch.optim.Adam(
            feature_params, lr=c.lr_feature_net, betas=c.betas, eps=1e-4)

        self.feature_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.feature_optim,
                                                                            factor=sched_factor,
                                                                            patience=sched_patience,
                                                                            threshold=sched_trehsh,
                                                                            min_lr=0, eps=1e-08,
                                                                            cooldown=sched_cooldown,
                                                                            verbose=True)
