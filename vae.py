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


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, output_dim)
        self.fc22 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        h1 = F.relu(self.fc1(x))
        mu, logvar = self.fc21(h1), self.fc22(h1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(input_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = torch.sigmoid(self.fc4(h3))
        return h4


class VAE(nn.Module):
    def __init__(self, sample_dim=784, hidden_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.encode = Encoder(input_dim=sample_dim, hidden_dim=hidden_dim, output_dim=z_dim)
        self.decode = Decoder(input_dim=z_dim, hidden_dim=hidden_dim, output_dim=sample_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    # ELBO
    # Reconstruction + KL divergence losses summed over all elements and batch
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, sample_dim), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE - KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # log
        batch_count = (epoch - 1) * len(train_loader) + batch_idx + 1
        writer.add_scalar('train_batch_loss', loss.item() / len(data), batch_count)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    train_loss /= len(train_loader.dataset)
    writer.add_scalar('train_epoch_loss', train_loss, epoch)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))

    # save model
    # torch.save(model.state_dict(), config['vae']['model_path'] + 'vae.pth')


def dev(epoch):
    global best_epoch, best_loss
    model.eval()
    dev_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(dev_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            dev_loss += loss_function(recon_batch, data, mu, logvar).item()
            # comparison between original data and reconstructed data
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(batch_size, 1, data_side, data_side)[:n]])
                save_image(comparison.cpu(),
                         config['vae']['result_path'] + 'reconstruction_' + str(epoch) + '.png', nrow=n)

    dev_loss /= len(dev_loader.dataset)
    writer.add_scalar('dev_epoch_loss', dev_loss, epoch)
    print('====> Dev set loss: {:.4f}'.format(dev_loss))

    # save model
    if dev_loss < best_loss:
        print('Better loss! Saving model!')
        torch.save(model.state_dict(), config['vae']['model_path'] + 'vae.pth')
        best_epoch, best_loss = epoch, dev_loss


def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            # comparison between original data and reconstructed data
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(batch_size, 1, data_side, data_side)[:n]])
                save_image(comparison.cpu(),
                         config['vae']['result_path'] + 'reconstruction_test.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    writer.add_scalar('test_loss', test_loss)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def main():
    epochs = int(config['vae']['epochs'])
    for epoch in range(1, epochs + 1):
        train(epoch)
        dev(epoch)
        # generate result every epoch
        with torch.no_grad():
            sample = torch.randn(64, z_dim).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, data_side, data_side),
                       config['vae']['result_path'] + 'sample_' + str(epoch) + '.png')

    print('Reload the best model on epoch', str(best_epoch), 'with min loss', str(best_loss))
    ckpt = torch.load(config['vae']['model_path'] + 'vae.pth')
    model.load_state_dict(ckpt)
    test()


if __name__ == "__main__":
    # config
    config = configparser.ConfigParser()
    config.read("config.ini")

    # seed
    seed = int(config['vae']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # log_interval
    log_interval = int(config['vae']['log_interval'])

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
    print('side', data_side)
    sample_dim = train_set[0][0].shape[1] * train_set[0][0].shape[2]
    hidden_dim = int(config['vae']['hidden_dim'])
    z_dim = int(config['vae']['z_dim'])
    print('sample_dim', sample_dim, 'hidden_dim', hidden_dim, 'z_dim', z_dim)

    # data loader
    batch_size = int(config['vae']['batch_size'])
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

    # model & optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(sample_dim=sample_dim, hidden_dim=hidden_dim, z_dim=z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config['vae']['lr']))

    # writer
    os.makedirs(config['vae']['log_path'], exist_ok=True)
    os.makedirs(config['vae']['model_path'], exist_ok=True)
    os.makedirs(config['vae']['result_path'], exist_ok=True)
    writer = SummaryWriter(config['vae']['log_path'])
    best_epoch, best_loss = 0, float('inf')
    main()
    writer.close()
