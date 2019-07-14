from __future__ import print_function
import configparser
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from tensorboardX import SummaryWriter
import random
import os
import math
import copy


# class CouplingLayer(nn.Module):
#     def __init__(self, num_inputs, num_hidden, mask, num_cond_inputs=None):
#         super(CouplingLayer, self).__init__()
#         self.num_inputs = num_inputs
#         self.mask = mask
#
#         if num_cond_inputs is not None:
#             total_inputs = num_inputs + num_cond_inputs
#         else:
#             total_inputs = num_inputs
#
#         self.scale_net = nn.Sequential(
#             nn.Linear(total_inputs, num_hidden),
#             nn.Tanh(),
#             nn.Linear(num_hidden, num_hidden),
#             nn.Tanh(),
#             nn.Linear(num_hidden, num_inputs))
#         self.translate_net = nn.Sequential(
#             nn.Linear(total_inputs, num_hidden),
#             nn.ReLU(),
#             nn.Linear(num_hidden, num_hidden),
#             nn.ReLU(),
#             nn.Linear(num_hidden, num_inputs))
#
#     def forward(self, inputs, cond_inputs=None, mode='direct'):
#         # input size rematch
#         inputs = inputs.view(-1, self.num_inputs)
#         masked_inputs = inputs * self.mask
#         if cond_inputs is not None:
#             masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)
#
#         if mode == 'direct':
#             log_s = self.scale_net(masked_inputs) * (1 - self.mask)
#             t = self.translate_net(masked_inputs) * (1 - self.mask)
#             s = torch.exp(log_s)
#             return inputs * s + t, log_s.sum(-1, keepdim=True)
#         else:
#             log_s = self.scale_net(masked_inputs) * (1 - self.mask)
#             t = self.translate_net(masked_inputs) * (1 - self.mask)
#             s = torch.exp(-log_s)
#             return (inputs - t) * s, -log_s.sum(-1, keepdim=True)
#
#
# class BatchNormFlow(nn.Module):
#     def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
#         super(BatchNormFlow, self).__init__()
#         self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
#         self.beta = nn.Parameter(torch.zeros(num_inputs))
#         self.momentum = momentum
#         self.eps = eps
#         self.num_inputs = num_inputs
#
#         self.register_buffer('running_mean', torch.zeros(num_inputs))
#         self.register_buffer('running_var', torch.ones(num_inputs))
#
#     def forward(self, inputs, cond_inputs=None, mode='direct'):
#         # input size rematch
#         inputs = inputs.view(-1, self.num_inputs)
#         if mode == 'direct':
#             if self.training:
#                 self.batch_mean = inputs.mean(0)
#                 self.batch_var = (
#                     inputs - self.batch_mean).pow(2).mean(0) + self.eps
#
#                 self.running_mean.mul_(self.momentum)
#                 self.running_var.mul_(self.momentum)
#
#                 self.running_mean.add_(self.batch_mean.data *
#                                        (1 - self.momentum))
#                 self.running_var.add_(self.batch_var.data *
#                                       (1 - self.momentum))
#
#                 mean = self.batch_mean
#                 var = self.batch_var
#             else:
#                 mean = self.running_mean
#                 var = self.running_var
#
#             x_hat = (inputs - mean) / var.sqrt()
#             y = torch.exp(self.log_gamma) * x_hat + self.beta
#             return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
#                 -1, keepdim=True)
#         else:
#             if self.training:
#                 mean = self.batch_mean
#                 var = self.batch_var
#             else:
#                 mean = self.running_mean
#                 var = self.running_var
#
#             x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)
#
#             y = x_hat * var.sqrt() + mean
#
#             return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
#                 -1, keepdim=True)
#
#
# class FlowSequential(nn.Sequential):
#     def __init__(self, num_inputs, num_blocks, num_hidden, num_cond_inputs=None):
#         modules = self._get_modules(num_inputs, num_blocks, num_hidden, num_cond_inputs)
#         super(FlowSequential, self).__init__(*modules)
#
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.orthogonal_(module.weight)
#                 if hasattr(module, 'bias') and module.bias is not None:
#                     module.bias.data.fill_(0)
#
#     def _get_modules(self, num_inputs, num_blocks, num_hidden, num_cond_inputs):
#         mask = torch.arange(0, num_inputs) % 2
#         mask = mask.to(device).float()
#         modules = []
#         for _ in range(num_blocks):
#             modules += [
#                 CouplingLayer(num_inputs, num_hidden, mask, num_cond_inputs),
#                 BatchNormFlow(num_inputs)
#             ]
#             mask = 1 - mask
#         return modules
#
#     def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
#         # self.num_inputs = inputs.size(-1)
#
#         if logdets is None:
#             logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)
#
#         assert mode in ['direct', 'inverse']
#         if mode == 'direct':
#             for module in self._modules.values():
#                 inputs, logdet = module(inputs, cond_inputs, mode)
#                 logdets += logdet
#         else:
#             for module in reversed(self._modules.values()):
#                 inputs, logdet = module(inputs, cond_inputs, mode)
#                 logdets += logdet
#
#         return inputs, logdets
#
#     def log_probs(self, inputs, cond_inputs=None):
#         # u, log_jacob = self(inputs, cond_inputs)
#         # log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
#         #     -1, keepdim=True)
#         # return (log_probs + log_jacob).sum(-1, keepdim=True)
#         z, log_diag_J = self(inputs, cond_inputs)
#         log_det_J = torch.sum(log_diag_J, dim=(1, 2, 3))
#         log_prior_prob = torch.sum(self.prior.log_prob(z), dim=(1, 2, 3))
#         return log_prior_prob + log_det_J
#
#     def sample(self, num_samples, num_inputs, noise=None, cond_inputs=None):
#         if noise is None:
#             noise = torch.Tensor(num_samples, num_inputs).normal_()
#         device = next(self.parameters()).device
#         noise = noise.to(device)
#         if cond_inputs is not None:
#             cond_inputs = cond_inputs.to(device)
#         samples = self.forward(noise, cond_inputs, mode='inverse')[0]
#         return samples


class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """
    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        self.register_buffer('mask', mask)

        # scale function
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear):
                self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=1))
        u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)  # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)

        log_abs_det_jacobian = - (1 - self.mask) * s  # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob

        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=1))
        x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du

        return x, log_abs_det_jacobian


class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0) # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians


class RealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        mask = torch.arange(input_size) % 2
        mask = mask.to(device).float()
        for i in range(n_blocks):
            modules += [
                LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size),
                BatchNorm(input_size)
            ]
            mask = 1 - mask

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return torch.distributions.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        x = x.view(x.shape[0], -1)
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)

    def sample(self, num_samples, num_inputs, noise=None):
        if noise is None:
            noise = torch.Tensor(num_samples, num_inputs).normal_()
        noise = noise.to(device)
        samples = self.forward(noise)[0]
        samples = samples.view(samples.shape[0], data_side, data_side)
        samples = torch.unsqueeze(samples, 1)
        print(samples.shape)
        return samples


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = -model.log_prob(data).mean(0)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        # log
        batch_count = (epoch - 1) * len(train_loader) + batch_idx + 1
        writer.add_scalar('train_batch_loss', loss.item(), batch_count)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))

    train_loss /= len(train_loader)
    writer.add_scalar('train_epoch_loss', train_loss, epoch)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))


def dev(epoch):
    global best_epoch, best_logprob
    model.eval()
    # dev_loss = 0
    # dev_loss = []
    dev_logprob = []
    with torch.no_grad():
        for i, (data, _) in enumerate(dev_loader):
            data = data.to(device)
            # loss = model.log_prob(data).mean(0)
            # logprobs.append(-model.log_prob(data))
            # dev_loss += loss.item()
            dev_logprob.append(model.log_prob(data))

    # dev_loss /= len(dev_loader)
    # writer.add_scalar('dev_epoch_loss', dev_loss, epoch)
    # print('====> Dev set loss: {:.4f}'.format(dev_loss))
    logprob = torch.cat(dev_logprob, dim=0).mean(0)
    writer.add_scalar('dev_logprob', logprob, epoch)
    print('====> Dev set logprob: {:.4f}'.format(logprob))

    # save model
    # if dev_loss < best_loss:
    #     print('Better loss! Saving model!')
    #     torch.save(model.state_dict(), config['realnvp']['model_path'] + 'realnvp.pth')
    #     best_epoch, best_loss = epoch, dev_loss
    if logprob > best_logprob:
        print('Better logprob! Saving model!')
        torch.save(model.state_dict(), config['realnvp']['model_path'] + 'realnvp.pth')
        best_epoch, best_logprob = epoch, logprob


def test():
    model.eval()
    # test_loss = 0
    test_logprob = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            # loss = -model.log_prob(data).mean(0)
            # test_loss += loss.item()
            test_logprob.append(model.log_prob(data))

    # test_loss /= len(test_loader)
    # writer.add_scalar('test_loss', test_loss)
    # print('====> Test set loss: {:.4f}'.format(test_loss))
    logprob = torch.cat(test_logprob, dim=0).mean(0)
    writer.add_scalar('test_logprob', logprob)
    print('====> Test set logprob: {:.4f}'.format(logprob))


def main():
    epochs = int(config['realnvp']['epochs'])
    for epoch in range(1, epochs + 1):
        train(epoch)
        dev(epoch)
        # generate result every epoch
        with torch.no_grad():
            # sample = model.sample(num_samples=64, num_inputs=sample_dim)
            # save_image(sample, config['realnvp']['result_path'] + 'sample_' + str(epoch) + '.png')

            u = model.base_dist.sample((64, 1)).squeeze()
            samples, _ = model.inverse(u)
            log_probs = model.log_prob(samples).sort(0)[1].flip(0)  # sort by log_prob; take argsort idxs; flip high to low
            samples = samples[log_probs]

            # convert and save images
            samples = samples.view(samples.shape[0], sample_dim)
            samples = (torch.sigmoid(samples) - 1e-6) / (1 - 2 * 1e-6)
            samples = samples.view(samples.shape[0], data_side, data_side)
            samples = torch.unsqueeze(samples, 1)
            save_image(samples, config['realnvp']['result_path'] + 'sample_' + str(epoch) + '.png')

    print('Reload the best model on epoch', str(best_epoch), 'with min loss', str(best_logprob))
    ckpt = torch.load(config['realnvp']['model_path'] + 'realnvp.pth')
    model.load_state_dict(ckpt)
    test()


if __name__ == "__main__":
    # config
    config = configparser.ConfigParser()
    config.read("config.ini")

    # seed
    seed = int(config['realnvp']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # log_interval
    log_interval = int(config['realnvp']['log_interval'])

    # data set
    os.makedirs(config['data']['path'], exist_ok=True)
    full_set = datasets.MNIST(config['data']['path'], train=True, download=True, transform=transforms.ToTensor())
    train_amount = int(len(full_set) * (1. - float(config['data']['dev_ratio'])))
    train_set = torch.utils.data.dataset.Subset(full_set, np.arange(train_amount))
    dev_set = torch.utils.data.dataset.Subset(full_set, np.arange(train_amount, len(full_set)))
    test_set = datasets.MNIST(config['data']['path'], train=False, download=True, transform=transforms.ToTensor())
    print('dataset size', len(train_set), len(dev_set), len(test_set))
    print('data size', train_set[0][0].shape)

    # dim
    data_side = train_set[0][0].shape[1]
    sample_dim = train_set[0][0].shape[1] * train_set[0][0].shape[2]
    hidden_dim = int(config['realnvp']['hidden_dim'])
    print('sample_dim', sample_dim, 'hidden_dim', hidden_dim)

    # data loader
    batch_size = int(config['realnvp']['batch_size'])
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

    # model & optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = FlowSequential(num_inputs=sample_dim,
    #                        num_blocks=int(config['realnvp']['blocks']),
    #                        num_hidden=int(config['realnvp']['hidden_dim'])).to(device)
    model = RealNVP(n_blocks=int(config['realnvp']['blocks']),
                    input_size=sample_dim,
                    hidden_size=int(config['realnvp']['hidden_dim']),
                    n_hidden=int(config['realnvp']['hidden_layers'])).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=float(config['realnvp']['lr']),
                           weight_decay=float(config['realnvp']['weight_decay']))

    # writer
    os.makedirs(config['realnvp']['log_path'], exist_ok=True)
    os.makedirs(config['realnvp']['model_path'], exist_ok=True)
    os.makedirs(config['realnvp']['result_path'], exist_ok=True)
    writer = SummaryWriter(config['realnvp']['log_path'])
    best_epoch, best_logprob = 0, float('-inf')
    main()
    writer.close()
