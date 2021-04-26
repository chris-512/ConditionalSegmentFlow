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
import models.modules as modules

# the reason the subnet init is needed, is that with uninitalized
# weights, the numerical jacobian check gives inf, nan, etc,

# https://github.com/VLL-HD/FrEIA/blob/451286ffae2bfc42f6b0baaba47f3d4583258599/tests/test_reversible_graph_net.py

def subnet_initialization(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.weight.data *= 0.3
        m.bias.data *= 0.1
        #m.weight.data.fill_(0.)
        #m.bias.data.fill_(0.)

def subnet_fc(c_in, c_out):
    print('subnet_fc: ', c_in, c_out)
    net = nn.Sequential(nn.Linear(c_in, 32), nn.ReLU(),
                         nn.Linear(32,  c_out))
    net.apply(subnet_initialization)
    return net

def subnet_conv(c_in, c_out):
    print('subnet_cont: ', c_in, c_out)
    net = nn.Sequential(nn.Conv2d(c_in, 32,   3, padding=1), nn.ReLU(),
                         nn.Conv2d(32,  c_out, 3, padding=1))
    net.apply(subnet_initialization)
    return net

def subnet_conv2(c_in, c_out):
    print('subnet_cont: ', c_in, c_out)
    net = nn.Sequential(nn.Conv2d(c_in, 64,   3, padding=1), nn.ReLU(),
                         nn.Conv2d(64,  c_out, 3, padding=1))
    net.apply(subnet_initialization)
    return net

def subnet_conv_1x1(c_in, c_out):
    print('subnet_conv_1x1: ', c_in, c_out)
    net = nn.Sequential(nn.Conv2d(c_in, 64,   1), nn.ReLU(),
                         nn.Conv2d(64,  c_out, 1))
    net.apply(subnet_initialization)
    return net

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
        self.ndim_x = 4 * 128 * 128

        self.input_node = Ff.InputNode(4, 128, 128, name='inp_points')
        self.conditions = [Ff.ConditionNode(3 * 4, 256 // 2, 256 // 2, name='cond-0'),
                           Ff.ConditionNode(
                               3 * 16, 256 // 4, 256 // 4, name='cond-1'),
                            Ff.ConditionNode(self.num_classes, name='cond-2')]

        self.nodes = []

        # input nodes
        self.nodes.append(self.input_node)

        """
        for k in range(1):
            print(k)
            self.nodes.append(Ff.Node(self.nodes[-1], Fm.ActNorm, {},
                              name=f'actnorm_{k}'))
            print(self.nodes[-1].out0[0].output_dims)
            self.nodes.append(Ff.Node(self.nodes[-1], Fm.IResNetLayer,
                              {'hutchinson_samples': 20,
                               'internal_size': 100,
                               'n_internal_layers': 3},
                              conditions=[self.conditions[0]],
                              name=f'i_resnet_{k}'))
            print(self.nodes[-1].out0[0].output_dims)
        """
    
        block = Fm.GLOWCouplingBlock

        for k in range(13):
            print(k)
            conv = Ff.Node(self.nodes[-1],
                           block,
                           {'subnet_constructor': subnet_conv, 'clamp': 2.0},
                           conditions=self.conditions[0],
                           name=F'conv{k}::c1')
            self.nodes.append(conv)
            print(self.nodes[-1].out0[0].output_dims)
            permute = Ff.Node(self.nodes[-1], Fm.PermuteRandom,
                              {'seed': k}, name=F'permute_{k}')
            self.nodes.append(permute)
            print(self.nodes[-1].out0[0].output_dims)

        self.nodes.append(Ff.Node(self.nodes[-1], Fm.HaarDownsampling, {}))
        print(self.nodes[-1].out0[0].output_dims)

        """
        for k in range(4):
            print(k)
            if k % 2 == 0:
                subnet = subnet_conv_1x1
            else:
                subnet = subnet_conv2

            linear = Ff.Node(self.nodes[-1],
                             block,
                             {'subnet_constructor': subnet, 'clamp': 1.2},
                             # conditions=self.conditions[1],
                             name=F'conv_low_res_{k}')
            self.nodes.append(linear)
            print(self.nodes[-1].out0[0].output_dims)
            permute = Ff.Node(self.nodes[-1], Fm.PermuteRandom,
                              {'seed': k}, name=F'permute_low_res_{k}')
            self.nodes.append(permute)
            print(self.nodes[-1].out0[0].output_dims)
            if k % 2 != 0:
                self.nodes.append(
                    Ff.Node(self.nodes[-1], Fm.IRevNetDownsampling, {}))
                print(self.nodes[-1].out0[0].output_dims)
        print(self.nodes[-1].out0[0].output_dims)
        """

        self.nodes.append(
            Ff.Node(self.nodes[-1], Fm.Flatten, {}, name='flatten'))
        print(self.nodes[-1].out0[0].output_dims)

        split_node = Ff.Node(self.nodes[-1],
                             Fm.Split,
                             {'section_sizes': (
                                 self.ndim_x // 4, 3 * self.ndim_x // 4), 'dim': 0},
                             name='split')
        self.nodes.append(split_node)
        print(self.nodes[-1].out0[0].output_dims)

        # Fully connected part
        for k in range(12):
            self.nodes.append(Ff.Node(self.nodes[-1],
                                      block,
                                      {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                                      conditions=self.conditions[2],
                                      name=F'fully_connected_{k}'))
            print(self.nodes[-1].out0[0].output_dims)
            self.nodes.append(Ff.Node(self.nodes[-1],
                                      Fm.PermuteRandom,
                                      {'seed': k},
                                      name=F'permute_{k}'))
            print(self.nodes[-1].out0[0].output_dims)

        # Concatenate the fully connected part and the skip connection to get a single output
        self.nodes.append(Ff.Node([self.nodes[-1].out0, split_node.out1],
                                  Fm.Concat1d, {'dim': 0}, name='concat'))
        print(self.nodes[-1].out0[0].output_dims)


        self.nodes.append(Ff.OutputNode(self.nodes[-1], name='output'))

        self.cinn = Ff.GraphINN(self.nodes + self.conditions, verbose=False)
        self.cinn = self.cinn.cuda()

        def init_model(model):
            for key, param in model.named_parameters():
                print(key)
                split = key.split('.')
                if param.requires_grad:
                    # c.init_scale = 0.03 (nearly xavier initialization)
                    param.data = c.init_scale * \
                        torch.randn(param.data.shape).cuda()
                    # last convolution in the coeff func
                    if len(split) > 3 and split[3][-1] == '2':
                        param.data.fill_(0.)

        init_model(self.cinn)

    def forward(self, x, cond, rev=False):
        # if load_inn_only:
        #    self.cinn.load_state_dict(torch.load(load_inn_only)['net'])

        if rev is False:
            x = modules.squeeze2d(x, factor=2)
            z, log_jac_det = self.cinn(x, c=cond, rev=rev)
        else:
            z, log_jac_det = self.cinn(x, c=cond, rev=rev)
            z = modules.unsqueeze2d(z, factor=2)

        return z, log_jac_det


class CondINNWrapper(nn.Module):
    def __init__(self, args, img_dims=None):
        super(CondINNWrapper, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        """
        self.backbone = ConditionalBackbone(args)
        """
        self.cinn = CondINN(args)
        self.gpu = args.gpu
        self.logprob_type = args.logprob_type
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.fc_cond_length = 256

        """
        upsample_layers = []
        for i in range(4):
            upsample_layers.append(nn.ConvTranspose2d(
                self.fc_cond_length + 81, self.fc_cond_length + 81, 3, stride=2, padding=1))
        self.upsample_layers = ListModule(*upsample_layers)

        self.fc_cond_net = nn.Sequential(*[nn.Conv2d(256 + 81, 128, 3, stride=2, padding=1),  # 337x32x32
                                           nn.LeakyReLU(),
                                           # 128x16x16
                                           nn.Conv2d(
                                               128, 64, 3, stride=2, padding=1),
                                           nn.LeakyReLU(),
                                           # 64x8x8
                                           nn.Conv2d(
                                               64, 256, 3, stride=2, padding=1),
                                           nn.LeakyReLU(),
                                           nn.AvgPool2d(4)],  # 256x4x4
                                         nn.BatchNorm2d(self.fc_cond_length)
                                         )  # 256x8x8
        """

        C = 4*4
        N = args.num_classes + 1
        W, H = (256//4, 256//4)
        self.learn_top = modules.Conv2dZeros(C * 2, C * 2)
        self.project_ycond = modules.LinearZeros(N, C*2)
        self.project_class = modules.LinearZeros(C, N)

        self.register_parameter(
            "prior_h",
            nn.Parameter(torch.zeros([args.batch_size, 2 * C, H, W])))
        self.register_parameter(
            "test_prior_h",
            nn.Parameter(torch.zeros([1, 2 * C, H, W])))

        self.optimizer = self.make_optimizer(args)
        self.scheduler = self.make_scheduler(args, self.optimizer)

    def make_optimizer(self, args):
        def _get_opt_(params):
            print('optimizer: ', args.optimizer)
            if args.optimizer == 'adam':
                optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                                       weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    params, lr=args.lr, momentum=args.momentum)
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer
        """
        opt = _get_opt_(list(self.backbone.parameters()) +
                        list(self.cinn.parameters()) +
                        list(self.fc_cond_net.parameters()) +
                        list(self.learn_top.parameters()) +
                        list(self.project_ycond.parameters()) +
                        list(self.project_class.parameters()) +
                        list(self.upsample_layers.parameters()))
        """
        opt = _get_opt_(list(self.parameters()))
        return opt

    def get_optimizer(self):
        return self.optimizer

    def make_scheduler(self, args, optimizer):
        print('learning rate scheduler: ', args.scheduler)
        # initialize the learning rate scheduler
        if args.scheduler == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, args.exp_decay)
        elif args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=args.epochs // 2, gamma=0.1)
        elif args.scheduler == 'linear':
            def lambda_rule(ep):
                lr_l = 1.0 - max(0, ep - 0.5 * args.epochs) / \
                    float(0.5 * args.epochs)
                return lr_l
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda_rule)
        else:
            assert 0, "args.schedulers should be either 'exponential' or 'linear'"

        return scheduler

    def scheduler_step(self, epoch):
        self.scheduler.step(epoch=epoch)
        print('Adjust learning rate: ', self.scheduler.get_lr())

    def conditioning(self, x, cond):

        net_outputs = self.backbone(x, cond)

        # 337x8x8
        net_outputs['3'] = net_outputs['3'].clone(
        ) + self.upsample_layers[3](net_outputs['pool'], output_size=net_outputs['3'].size())
        # 337x16x16
        net_outputs['2'] = net_outputs['2'].clone(
        ) + self.upsample_layers[2](net_outputs['3'], output_size=net_outputs['2'].size())
        # 337x32x32
        net_outputs['1'] = net_outputs['1'].clone(
        ) + self.upsample_layers[1](net_outputs['2'], output_size=net_outputs['1'].size())
        # 337x64x64
        net_outputs['0'] = net_outputs['0'].clone(
        ) + self.upsample_layers[0](net_outputs['1'], output_size=net_outputs['0'].size())

        cond0 = net_outputs['1']

        cond_net_outputs = self.fc_cond_net(cond0)

        cond1 = cond_net_outputs.view(cond_net_outputs.shape[0], -1)

        # 64x256
        return [cond0, cond1]

    def prior(self, y_onehot=None):

        if self.training:
            print('training')
            B, C = self.prior_h.size(0), self.prior_h.size(1)
            hid = self.prior_h.detach().clone()
        else:
            B, C = self.test_prior_h.size(0), self.test_prior_h.size(1)
            hid = self.test_prior_h.detach().clone()

        assert torch.sum(hid) == 0.0
        # preserve # of input_channels == # of output_channels
        hid = self.learn_top(hid)
        # encode one-hot class-condition
        hid += self.project_ycond(y_onehot).view(B, C, 1, 1)
        C = hid.size(1)
        mean = hid[:, :C//2, ...]
        logs = hid[:, C//2:, ...]
        return mean, logs

    def forward(self, x, y, cond, writer=None):

        self.optimizer.zero_grad()

        batch_size = x.size(0)

        x = x.float().cuda()
        y = y.float().cuda()
        y = y.float().cuda() + torch.normal(mean=torch.zeros_like(y), std = torch.ones_like(y) * 0.001).cuda()
        cond = cond.cuda()

        # conditions = self.conditioning(x, cond)
        conditions = []
        x = modules.squeeze2d(x, factor=2)
        conditions.append(x)
        x = modules.squeeze2d(x, factor=2)
        conditions.append(x)
        conditions.append(cond)

        y = y.unsqueeze(1)

        z, log_jac_det = self.cinn(y, conditions)
        z_before = z
        z = z.view(-1, 1, y.size(2), y.size(3))
        z = modules.squeeze2d(z, factor=4)
        loss_norm = y.size(2) * y.size(3)

        # cond = [B, 81]
        # z = [B, C, W, H]
        # mean = [B, C, W, H]
        # logs = [B, C, W, H]
        mean, logs = self.prior(cond)
        print("Before: ", mean[0].mean(), logs[0].mean())
        prior_prob = modules.GaussianDiag.logp(mean, logs, z)

        # classification loss
        y_logits = self.project_class(z.mean(2).mean(2))
        bce_loss = self.bce_loss(y_logits, cond)

        z_perturb = z_before + \
            torch.normal(mean=torch.zeros_like(z_before),
                         std=torch.ones_like(z_before) * 0.1)
        sample, _ = self.cinn(z_perturb, conditions, rev=True)
        recons_loss = ((sample - y) ** 2).mean()

        loss = -log_jac_det - prior_prob
        print('CE loss: ', bce_loss)
        print('prior loss: ', -prior_prob.mean())
        print('log_jac_det: ', -log_jac_det.mean())
        # print(log_jac_det.mean())
        # print(gaussian_log_prob.mean())
        loss = loss.mean() / loss_norm
        print('mean loss: ', loss)
        loss += bce_loss
        # loss += 1000*recons_loss
        print('bce loss: ', bce_loss)
        loss.backward()

        """
        for key, params in self.named_parameters():
            print(key)
            if "prior_h" in key:
                print(params.mean())
            elif "learn_top" in key:
                print(params.mean())
            elif "project_ycond" in key:
                print(params.mean())
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5)

        self.optimizer.step()

        # import pdb; pdb.set_trace()
        # print('z-min: ', z.min())
        # print('z-max: ', z.max())
        # print('z-mean: ', z.mean())

        # z = z.mean() + torch.empty(z.shape).normal_(mean=0, std=0.005).cuda()
        mean, logs = self.prior(cond)
        print("After: ", mean[0].mean(), logs[0].mean())
        z = modules.GaussianDiag.sample(mean, logs, 0.01)
        z = modules.unsqueeze2d(z, factor=4)
        z = z.view(z.size(0), -1)

        print(((z - z_before) ** 2).mean().item())
        z_perturb = z_before + \
            torch.normal(mean=torch.zeros_like(z_before),
                         std=torch.ones_like(z_before) * 0.1)
        print(((z_perturb - z_before) ** 2).mean().item())
        sample, _ = self.cinn(z, conditions, rev=True)
        print(sample.shape)
        print(abs(sample - y).mean())

        losses = {
            'train_loss': loss,
            'prior_prob': prior_prob.mean() / loss_norm,
            'logdet': log_jac_det.mean() / loss_norm,
            'bce_loss': bce_loss,
            'recons_error': abs(sample - y).mean()
        }

        # z = z.view(z.size(0), 3, 64, 64)

        return sample, losses

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

        x = x.float().cuda()
        cond = cond.cuda()

        # conditions = self.conditioning(x, cond)
        conditions = []
        x = modules.squeeze2d(x, factor=2)
        conditions.append(x)
        x = modules.squeeze2d(x, factor=2)
        conditions.append(x)
        conditions.append(cond)

        mean, logs = self.prior(cond)
        z = modules.GaussianDiag.sample(mean, logs, 0.01)
        z = modules.unsqueeze2d(z, factor=4)
        z = z.view(z.size(0), -1)
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
        fpn_outputs['pool'] = torch.cat(
            [fpn_outputs['pool'], cond_pool], dim=1)

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
