from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel

import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
import configparser
import numpy as np
from tensorboardX import SummaryWriter

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def weights_init(m):
    # custom weights initialization called on Generator and Discriminator
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


def train(epoch):
    for i, data in enumerate(train_loader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real data
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        output = netD(real_cpu)
        errD_real = criterion(output, label)  # loss
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake data
        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)  # loss
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # log
        batch_count = (epoch - 1) * len(train_loader) + i + 1
        writer.add_scalar('Loss_D', errD.item(), batch_count)
        writer.add_scalar('Loss_G', errG.item(), batch_count)
        writer.add_scalar('D_x', D_x, batch_count)
        writer.add_scalar('D_G_z1', D_G_z1, batch_count)
        writer.add_scalar('D_G_z2', D_G_z2, batch_count)
        if i % log_interval == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(train_loader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    # real samples
    n = min(real_cpu.size(0), 64)
    save_image(real_cpu[:n], config['dcgan']['result_path'] + 'real_samples.png', normalize=True)
    # generated samples
    with torch.no_grad():
        fake = netG(fixed_noise)
    save_image(fake.detach()[:n], config['dcgan']['result_path'] + 'fake_samples_epoch_%03d.png' % epoch, normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), config['dcgan']['model_path'] + 'netG_epoch_%03d.pth' % epoch)
    torch.save(netD.state_dict(), config['dcgan']['model_path'] + 'netD_epoch_%03d.pth' % epoch)


def main():
    for epoch in range(1, epochs + 1):
        train(epoch)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    args = parser.parse_args()
    ngpu = int(args.ngpu)

    # config
    config = configparser.ConfigParser()
    config.read("config.ini")

    # seed
    seed = int(config['dcgan']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # log_interval
    log_interval = int(config['dcgan']['log_interval'])

    # data set
    os.makedirs(config['data']['path'], exist_ok=True)
    data_transforms = transforms.Compose([
        transforms.Resize(int(config['dcgan']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
    full_set = datasets.MNIST(config['data']['path'], train=True, download=True, transform=data_transforms)
    train_amount = int(len(full_set) * (1. - float(config['data']['dev_ratio'])))
    train_set = torch.utils.data.dataset.Subset(full_set, np.arange(train_amount))
    dev_set = torch.utils.data.dataset.Subset(full_set, np.arange(train_amount, len(full_set)))
    test_set = datasets.MNIST(config['data']['path'], train=False, download=True, transform=data_transforms)
    print('dataset size', len(train_set), len(dev_set), len(test_set))
    print('data size', train_set[0][0].shape)

    # data loader
    batch_size = int(config['dcgan']['batch_size'])
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

    # parameters
    color_channels = train_set[0][0].shape[0]
    z_dim = int(config['dcgan']['z_dim'])
    g_feature_map = int(config['dcgan']['g_feature_map'])
    d_feature_map = int(config['dcgan']['d_feature_map'])

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator(ngpu, nz=z_dim, ngf=g_feature_map, nc=color_channels).to(device)
    netG.apply(weights_init)
    print(netG)
    netD = Discriminator(ngpu, nc=color_channels, ndf=d_feature_map).to(device)
    netD.apply(weights_init)
    print(netD)

    # optimizer
    lr = float(config['dcgan']['lr'])
    beta1 = float(config['dcgan']['beta1'])
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # loss
    criterion = nn.BCELoss()
    # fixed noise
    fixed_noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
    # label
    real_label = 1
    fake_label = 0

    # writer
    os.makedirs(config['dcgan']['log_path'], exist_ok=True)
    os.makedirs(config['dcgan']['model_path'], exist_ok=True)
    os.makedirs(config['dcgan']['result_path'], exist_ok=True)
    writer = SummaryWriter(config['dcgan']['log_path'])
    epochs = int(config['dcgan']['epochs'])
    main()
    writer.close()
