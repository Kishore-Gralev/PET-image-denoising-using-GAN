import argparse
import os, shutil
import numpy as np
import math, sys
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import torch.optim.lr_scheduler as lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument("--no", type=int, help="number of training")
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
#parser.add_argument("--lrD", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--lambda_gp", type=float, default=10, help="wgan gp parameter")
parser.add_argument("--early_stop", type=int, default=30, help="early stop")
parser.add_argument("--weight_decay", type=float,default=0.0001, help="L2 regularization")
parser.add_argument("--save_path", type=str, default="1_NoAug", help="results path")
opt = parser.parse_args()
print(opt)

# Save path
if not os.path.exists(opt.save_path):
    os.mkdir(opt.save_path)
save_path = os.path.join(opt.save_path, str(opt.no))
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(save_path)

# Define dataset
class MyDataset(Dataset):
    def __init__(self, x_root, y_root):
        x_list = []
        y_list = []
        tf = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # for x root
        file_list = [file for _, _, file in os.walk(x_root)][0]
        file_list.sort(key=lambda x: float(x.split(".")[0]))
        
        for f in file_list:
            f_name = os.path.join(x_root, f)
            im = cv2.imread(f_name, cv2.IMREAD_GRAYSCALE)
            im = np.expand_dims(im, axis=-1)
            im = tf(im)
            x_list.append(im)

        # for y root
        file_list = [file for _, _, file in os.walk(y_root)][0]
        file_list.sort(key=lambda x: float(x.split(".")[0]))
  
        for f in file_list:
            f_name = os.path.join(y_root, f)
            im = cv2.imread(f_name, cv2.IMREAD_GRAYSCALE)
            im = np.expand_dims(im, axis=-1)
            im = tf(im)
            y_list.append(im)
        
        self.x_list = x_list
        self.y_list = y_list
        
    def __getitem__(self, index):
        return self.x_list[index], self.y_list[index]

    def __len__(self):
        return len(self.x_list)
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        layers = [
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(32, 1, 3, stride=1, padding=1), 
            nn.Tanh(),
            ]   
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.net(x)
        return out  

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        layers = [
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True), 
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.AdaptiveAvgPool2d(1)
            ] 
        self.net = nn.Sequential(*layers)

    def forward(self, img):
        out = self.net(img)
        return out

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
    # Get random interpolation between real and fake samples
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    d_interpolates = D(interpolates)
    fake_ones = Variable(torch.ones(d_interpolates.size()), requires_grad=False).cuda()
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

cuda = True if torch.cuda.is_available() else False

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
mse = nn.MSELoss()

if cuda:
    generator.cuda()
    discriminator.cuda()
    mse.cuda()

# Configure data loader
train_dataset = MyDataset("low", "high")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu, drop_last=False)
#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu, drop_last=False)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.n_cpu, drop_last=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

best_epoch =  {"epoch": -1,  "mse": 1234567890 }
# best_test =  {"epoch": -1,  "mae": 1234567890 }

start_time = time.time()

total_real = 0
total_fake = 0
correct_real = 0
correct_fake = 0


# Training loop

for epoch in range(opt.n_epochs):
    mean_mse = 0
    #mean_mae=0
    mean_d = 0
    mean_g = 0
    cnt = 0
    generator.train()
    discriminator.train()
    for i, data in enumerate(train_dataloader):

        # Configure input
        x, y = data

        x = Variable(x).cuda()
        y = Variable(y).cuda()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Generate a batch of images
        fake_imgs = generator(x)

        # Real images
        y_validity = discriminator(y)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, y.data, fake_imgs.data)
        # Adversarial loss
        real_loss = torch.mean(F.binary_cross_entropy_with_logits(y_validity, torch.ones_like(y_validity)))
        fake_loss = torch.mean(F.binary_cross_entropy_with_logits(fake_validity, torch.zeros_like(fake_validity)))
        d_loss = real_loss + fake_loss + opt.lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        mean_d += d_loss.item()

        optimizer_G.zero_grad()

                # Compute predictions for real and fake images
        real_predictions = torch.round(torch.sigmoid(y_validity)).squeeze()
        fake_predictions = torch.round(torch.sigmoid(fake_validity)).squeeze()
        
                # Update counts for real and fake images
        total_real += real_predictions.size(0)
        total_fake += fake_predictions.size(0)
        correct_real += torch.sum(real_predictions == 1).item()
        correct_fake += torch.sum(fake_predictions == 0).item()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(x)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            gan_loss = torch.mean(F.binary_cross_entropy_with_logits(fake_validity, torch.ones_like(fake_validity)))
            mse_loss = mse(fake_imgs, y)
            
            g_loss = mse_loss + 1e-3 * gan_loss
            #g_loss = mae_loss + 1e-3 * gan_loss
            g_loss.backward()
            optimizer_G.step()

            mean_mse += mse_loss.item()
            #mean_mae += mae_loss.item()
            mean_g += gan_loss.item()
            cnt += 1
            
            # Calculate accuracy for real and fake images
            accuracy_real = correct_real / total_real
            accuracy_fake = correct_fake / total_fake

    # epoch done 
    mean_mse /= cnt
    #mean_mae /= cnt
    mean_g /= cnt
    mean_d /= len(train_dataloader)
    
    
    if best_epoch["mse"] > mean_mse:
        best_epoch =  {"epoch": epoch,  "mse": mean_mse }
        print("Saving results ......")        
    
        # Save results
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            for jdx, data in enumerate(train_dataloader):
                x, y = data

                x = Variable(x).cuda()
                y = Variable(y).cuda()

                fake_hr = generator(x) 

                # save image
                fake_hr = fake_hr.cpu()
                for idx, fake in enumerate(fake_hr):
                    fake = fake.numpy().transpose((1, 2, 0))
                    fake = np.clip(fake, 0, 1)
                    fake = fake[:,:,0]
                    new_path=str(save_path) +"/"+"fake_{}".format(jdx * opt.batch_size + idx)+'.png'
                    plt.imsave(new_path,fake, cmap=plt.cm.gray)
                    plt.close()
                    
                 
    print(
        "[Epoch {}/{}] [MSE loss: {:.3e}] [D loss: {:.3e}] [G loss: {:.3e}] [Acc Real: {:.2f}%] [Acc Fake: {:.2f}%]".format(
            epoch, opt.n_epochs, mean_mse, mean_d, mean_g, accuracy_real * 100, accuracy_fake * 100))
    
    print(
        "best_epoch: [epoch: {}] [MSE loss: {:.3e}]\n".format(
            best_epoch["epoch"], best_epoch["mse"]
        )
    )
    
    if epoch - best_epoch["epoch"] > opt.early_stop:
        print("Early stop!")
        exit()

end_time = time.time()
elapsed_time = end_time - start_time

print("Training completed in {:.2f} seconds." .format(elapsed_time))
