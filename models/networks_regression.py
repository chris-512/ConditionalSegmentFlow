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


class HyperRegression(nn.Module):
    def __init__(self, args, input_width=320, input_height=576):
        super(HyperRegression, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.hyper = HyperFlowNetwork(
            args, input_height=input_height, input_width=input_width)
        self.point_cnf = get_hyper_cnf(args)
        self.gpu = args.gpu
        self.logprob_type = args.logprob_type
        self.ce_loss = nn.CrossEntropyLoss()

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
        opt = _get_opt_(list(self.hyper.parameters()) +
                        list(self.point_cnf.parameters()))
        return opt

    def forward(self, x, y, class_cond, opt, step, writer=None):
        opt.zero_grad()
        batch_size = x.size(0)
        class_cond = class_cond.cuda()
        target_networks_weights = self.hyper(x, class_cond)
        # print(target_networks_weights.shape)
        # print(class_cond.shape)

        # Normalizing Flow Loss
        # y = torch.exp(y-0.5)
        y, delta_log_py = self.point_cnf(
            y, target_networks_weights, torch.zeros(batch_size, y.size(1), 1).to(y))
        if self.logprob_type == "Laplace":
            log_py = standard_laplace_logprob(y).view(
                batch_size, -1).sum(1, keepdim=True)
        if self.logprob_type == "Normal":
            log_py = standard_normal_logprob(y).view(
                batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, y.size(1), 1).sum(1)

        log_px = log_py - delta_log_py

        loss = -log_px.mean()

        # Cross Entropy Loss
        # ex) num_classes = 80
        # background + classes = 81 classes (index: 0 ~ 80)
        # target_class_ind = (class_cond == 1).nonzero()[:, 1]
        # multi_ce_loss = self.ce_loss(pred_class_outputs, target_class_ind)

        # loss += multi_ce_loss

        loss.backward()
        opt.step()

        recon = -log_px.sum()
        recon_nats = recon / float(y.size(0))
        return recon_nats

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

    def decode(self, z, class_cond, num_points):
        # transform points from the prior to a point cloud, conditioned on a shape code
        class_cond = class_cond.cuda()
        target_networks_weights = self.hyper(z, class_cond)

        if self.logprob_type == "Laplace":
            y = self.sample_laplace(
                (z.size(0), num_points, self.input_dim), self.gpu)
        if self.logprob_type == "Normal":
            y = self.sample_gaussian(
                (z.size(0), num_points, self.input_dim), None, self.gpu)
        x = self.point_cnf(y, target_networks_weights,
                           reverse=True).view(*y.size())
        return y, x

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


class HyperFlowNetwork(nn.Module):
    def __init__(self, args, input_width=320, input_height=576):
        super().__init__()

        # self.encoder = FlowNetS(input_width=input_width,
        #                        input_height=input_height, input_channels=args.input_channels)

        self.encoder = ResNet101FPN(input_width=input_width,
                                    input_height=input_height)
        output = []
        self.n_out = self.encoder.n_out + 81

        # self.n_out = 46080
        dims = tuple(map(int, args.dims.split("-")))  # 128-128-128
        total_output_dims = 0
        for k in range(len(dims)):
            if k == 0:
                output.append(
                    nn.Linear(self.n_out, args.input_dim * dims[k], bias=True))
                total_output_dims += args.input_dim * dims[k]
            else:
                output.append(
                    nn.Linear(self.n_out, dims[k - 1] * dims[k], bias=True))
                total_output_dims += dims[k - 1] * dims[k]
            # bias
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            # scaling
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            # shift
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            total_output_dims += dims[k] * 4

        output.append(
            nn.Linear(self.n_out, dims[-1] * args.input_dim, bias=True))
        # bias
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))
        # scaling
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))
        # shift
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))

        total_output_dims += (dims[-1] * args.input_dim + args.input_dim * 4)

        self.output = ListModule(*output)

    def forward(self, x, conditional):
        encoder_outputs = self.encoder(x)
        encoder_outputs = torch.cat([encoder_outputs, conditional], dim=1)
        # output = output.view(output.size(0), -1)
        multi_outputs = []
        for j, target_network_layer in enumerate(self.output):
            multi_outputs.append(target_network_layer(encoder_outputs))
        multi_outputs = torch.cat(multi_outputs, dim=1)
        print('Output dims of HyperFlowNetwork: ', multi_outputs.size(1))
        # multi_outputs = nn.Sigmoid()(multi_outputs)  # Add sigmoid
        return multi_outputs


class FlowNetS(nn.Module):
    def __init__(self, input_channels=12, batchNorm=True, input_width=320, input_height=576):
        super(FlowNetS, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels,
                          64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.conv7 = conv(self.batchNorm, 1024, 1024,
                          kernel_size=1, stride=1, padding=0)
        self.conv8 = conv(self.batchNorm, 1024, 1024,
                          kernel_size=1, stride=1, padding=0)

        fc1_input_features = (input_height * input_width) // 4
        self.fc1 = nn.Linear(in_features=fc1_input_features,
                             out_features=1024, bias=True)

        self.fc2 = nn.Linear(in_features=1024, out_features=1024, bias=True)

        # self.predict_6 = predict_flow(1024)
        # self.fc1 = nn.Linear(in_features=90, out_features=512, bias=True)
        # self.fc2 = nn.Linear(in_features=512, out_features=1024, bias=True)
        # self.fc3 = nn.Linear(in_features=1024, out_features=2048, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        out_conv8 = self.conv8(self.conv7(out_conv6))  # SDD
        # [batch_size, 1024, 4, 4]
        # view(batch_size, -1) -> [batch_size, 1024 * 4 * 4]
        # out_fc1 = nn.functional.relu(
        #    self.fc1(out_conv8.view(out_conv6.size(0), -1)))
        kernel_size = out_conv8.size(2)
        out_gap = nn.AvgPool2d(kernel_size=kernel_size)(
            out_conv8).view(out_conv8.size(0), -1)
        # out_fc1 = [batch_size, 1024]
        # out_fc2 = nn.functional.relu(self.fc2(out_fc1))
        # out_fc2 = [batch_size, 1024]
        out_gap = nn.functional.relu(out_gap)
        return out_gap


class ResNet50(nn.Module):
    def __init__(self, input_width=224, input_height=224):
        super(ResNet50, self).__init__()

        self.net = resnet50(pretrained=True)
        num_features = self.net.fc.in_features
        self.net.fc = nn.Linear(num_features, 1024)
        self.net = self.net.cuda()
        self.n_out = 1024

    def forward(self, x):
        return self.net(x)


class ResNet101(nn.Module):
    def __init__(self, input_width=224, input_height=224):
        super(ResNet101, self).__init__()

        self.net = resnet101(pretrained=True)
        num_features = self.net.fc.in_features
        self.net.fc = nn.Linear(num_features, 1024)
        self.net = self.net.cuda()
        self.n_out = 1024

    def forward(self, x):
        return self.net(x)


class ResNet50FPN(nn.Module):
    def __init__(self, input_width=224, input_height=224):
        super(ResNet50FPN, self).__init__()

        self.backbone = resnet_fpn_backbone(
            'resnet50', pretrained=True, trainable_layers=3)
        # num_features = self.net.fc.in_features
        # self.net.fc = nn.Linear(num_features, 1024)
        # self.net = self.net.cuda()
        self.n_out = 1280

    def forward(self, x):
        outputs = self.backbone(x)
        batch_size = outputs['0'].size(0)

        net_outputs = []
        for key, output in outputs.items():
            net_outputs.append(nn.AvgPool2d(kernel_size=output.size(2))(
                output))
            # print(net_outputs[-1].shape)
        net_outputs = torch.cat(net_outputs, dim=1).view(
            batch_size, -1)
        return net_outputs


class ResNet101FPN(nn.Module):
    def __init__(self, input_width=224, input_height=224):
        super(ResNet101FPN, self).__init__()

        self.backbone = resnet_fpn_backbone(
            'resnet101', pretrained=True, trainable_layers=3)
        # num_features = self.net.fc.in_features
        # self.net.fc = nn.Linear(num_features, 1024)
        # self.net = self.net.cuda()
        self.n_out = (64 * 64 + 32 * 32 + 16 * 16 + 8 * 8 + 4 * 4)

        outputs = []
        for i in range(5):
            outputs.append(nn.Conv2d(256, 1, 1))
        self.conv1x1_layers = ListModule(*outputs)

    def forward(self, x):
        outputs = self.backbone(x)
        batch_size = outputs['0'].size(0)

        # outputs['0'].view(batch_size, -1)

        net_outputs = []
        for multi_scale_output, conv1x1_layer in zip(outputs.values(), self.conv1x1_layers):
            # net_outputs.append(nn.AvgPool2d(kernel_size=multi_scale_output.size(2))(
            #    multi_scale_output))
            net_outputs.append(conv1x1_layer(
                multi_scale_output).view(batch_size, -1))
            # print(net_outputs[-1].shape)
        net_outputs = torch.cat(net_outputs, dim=1).view(
            batch_size, -1)
        return net_outputs
