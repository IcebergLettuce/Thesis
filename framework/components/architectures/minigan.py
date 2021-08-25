import torch
import torch.nn as nn
import torchcsprng as prng
import numpy as np
import pandas as p
import matplotlib.pyplot as plt
import os
import random
import time

import torch.utils.data as data_utils
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
import logging

logger = logging.getLogger(__name__)

ngf = 96
ndf = 96

def rescale_unet(x):
    return 255 * (x - x.min()) / (x.max() - x.min())

def weights_init(m):

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("SpectralNorm") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)


class Generator(nn.Module):

    def __init__(self, ngf=96):
        super(Generator, self).__init__()
        self.ngf = ngf
        
        self.convtrans1 = nn.Sequential(
            nn.ConvTranspose2d(128, 2 * 1, 98, 1, 1, bias=False),
            nn.BatchNorm2d(2 * 1),
            nn.ReLU(True),
        )

        self.convtrans2 = nn.Sequential(
            nn.ConvTranspose2d(10, 2, 1,1, bias=False)
        )
        self.activationG = nn.Tanh()

    def forward(self, inp):
        x = self.convtrans1(inp)
        #x = self.convtrans2(x)
        tanh_out = self.activationG(x)
        return tanh_out

class Discriminator(nn.Module):
    def __init__(self, ndf=96):
        super(Discriminator, self).__init__()
        self.ndf = ndf

        self.conv1 = nn.Sequential(

            nn.Conv2d(2, 1, 5, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 1, 16, 20,1,dilation=2,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, inp):

        x = self.conv1(inp)
        x = self.conv2(x)
        return x

class MINIGAN:

    def __init__(self, hp):
        logger.info(f'Initializing {__class__.__name__}')
        self.hp = hp
        self.training_log = {'epoch':[],'d_loss':[],'g_loss':[],'epsilon':[]}

    def generate(self, run_dir,n):

        manualSeed = 999
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        device = torch.device("cpu")

        netG = Generator(96)
        netG = convert_batchnorm_modules(netG).to(device)

        optimizerG = optim.Adam(netG.parameters(), lr=self.hp['lr_g'], betas=(self.hp['beta_1_g'], self.hp['beta_2_g']))

        params = torch.load(os.path.join(run_dir,'torch_weights'))
        netG.load_state_dict(params["generator"], strict=False)
        optimizerG.load_state_dict(params["optimizer_g"])

        test_noise = torch.randn(n, 128, 1, 1)

        test_fake = torch.empty(n, 2, 96, 96)
        test_fake = netG(test_noise).detach().cpu()
               
        for i, fake in enumerate(test_fake):
            sample = fake.clone()
            sample[1][sample[1] > 0.5] = 1
            sample[1][sample[1] <= 0.5] = 0
            sample[0] = rescale_unet(sample[0])  # rescaling back to 0-255
            test_fake[i] = sample

        gan_img = test_fake[:, 0, :, :].cpu().numpy()
        gan_label = test_fake[:, 1, :, :].cpu().numpy()
        return gan_img, gan_label


    def fit(self,dataset,epochs, run_dir, image_gen_callback):

        logger.info(f'Start fitting GAN for {epochs}')
        cardinality = len(dataset)
        logger.info(f'Training data has cardinality of {cardinality}')

        manualSeed = 999
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        #torch.set_default_tensor_type("torch.cuda.FloatTensor")

        sample_gen = prng.create_random_device_generator("/dev/urandom")
        dataloader = data_utils.DataLoader(
            dataset
        )

        device = torch.device("cpu")

        netG = Generator(96).to(device)
        netG.apply(weights_init)

        netD = Discriminator(96)

        netD = convert_batchnorm_modules(netD)
        netG = convert_batchnorm_modules(netG)

        netD = netD.to(device)

        netD.apply(weights_init)

        fixed_noise = torch.randn(self.hp['batch_size'], 128, 1, 1, device=device)

        optimizerD = optim.Adam(
        netD.parameters(), lr=self.hp['lr_d'], betas=(self.hp['beta_1_d'], self.hp['beta_2_d'])
        )

        privacy_engine = PrivacyEngine(
            netD,
            sample_rate=self.hp['batch_size'] / cardinality, #attention
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=self.hp['noise_multiplier'],
            max_grad_norm=self.hp['l2_norm_clip'],
            secure_rng=True,
            target_delta=1.0 / cardinality, # attention
        )
        privacy_engine.attach(optimizerD)
        torch.set_default_tensor_type("torch.FloatTensor")
        epsilon, best_alpha = optimizerD.privacy_engine.get_privacy_spent(1.0 / cardinality)
        logger.info(
            "(epsilon = %.2f, delta = %.2f) for alpha = %.2f"
            % (epsilon, 1.0 / cardinality, best_alpha)
        )
       # torch.set_default_tensor_type("torch.cuda.FloatTensor")

        optimizerG = optim.Adam(netG.parameters(), lr=self.hp['lr_g'], betas=(self.hp['beta_1_g'], self.hp['beta_2_g']))

        nr_params_g = sum(p.numel() for p in netG.parameters())
        nr_params_d = sum(p.numel() for p in netD.parameters())

        cd_1 = []
        cd_2 = []
        cd_3 = []

        cg_1 = []
        for epoch in range(epochs):

            if (epoch == 0):
                first_time = time.time()
                second_time = time.time()

            if (epoch == 1):
                second_time = time.time()    

            diff = second_time - first_time  

            logger.info(f'Epoch: {epoch} of {epochs} [h]')  
            
            for i, data in enumerate(dataloader, 0):
                
                numb_batches = cardinality // self.hp["batch_size"]
                #logger.info(f'Epoch: {epoch} Batch: {i+1} of {numb_batches}')

                for _ in range(self.hp['n_critic']):

                    optimizerD.zero_grad()
                    # Format batch
                    real_cpu = data[0].to(device)
                    b_size = real_cpu.size(0)

                    # Forward pass real batch through D
                    errD_real = netD(real_cpu)

                    # Calculate loss on all-real batch
                    errD_real = errD_real.view(-1).mean() * -1

                    errD_real.backward()
                    optimizerD.step()

                    noise = torch.randn(b_size, 128, 1, 1, device=device)

                    fake = netG(noise)

                    # Classify all fake batch with D
                    errD_fake = netD(fake.detach())
        
                    # Calculate D's loss on the all-fake batch
                    errD_fake = errD_fake.view(-1).mean()

                    errD_fake.backward()
                    optimizerD.step()

                    for parameter in netD.parameters():
                        parameter.data.clamp_(-self.hp['clip_value'], self.hp['clip_value'])
                    
                    
                    errD = errD_fake + errD_real # TODO: Collect this one
                    cd_1.append(errD.item())

               
                ############################
                # (2) Update G network
                ###########################

                # netG.zero_grad()
                optimizerG.zero_grad()

                # Since we just updated D, perform another forward pass of all-fake batch through D
                output_fake = netD(fake)

                errG = -output_fake.view(-1).mean()
                
                # Calculate gradients for G
                errG.backward() # TODO: Collect

                optimizerG.step()

                cd_2.append(np.mean(cd_1))
                cg_1.append(np.mean(errG.item()))
                            
            torch.set_default_tensor_type("torch.FloatTensor")
            epsilon, best_alpha = optimizerD.privacy_engine.get_privacy_spent(1.0 / cardinality)
            logger.info(
                "(epsilon = %.2f, delta = %.2f) for alpha = %.2f"
                % (epsilon, 1.0 / cardinality, best_alpha)
            )

            self.training_log['epoch'].append(epoch)
            self.training_log['d_loss'].append(np.mean(cd_2))
            self.training_log['g_loss'].append(np.mean(cg_1))
            self.training_log['epsilon'].append(epsilon)
            
            p.DataFrame(self.training_log).to_csv(os.path.join(run_dir,'training.log'))

            '''
            Generating images
            '''

            fixed_fake = netG(fixed_noise).detach().cpu()

            torch.save(
                {
                    "discriminator": netD.state_dict(),
                    "generator": netG.state_dict(),
                    "optimizer_d": optimizerD.state_dict(),
                    "optimizer_g": optimizerG.state_dict(),
                },
                os.path.join(run_dir,'torch_weights'),
            )

            gan_img, gan_label = self.generate(run_dir, 10)
            image_gen_callback.generate_image(epoch, gan_img, gan_label, run_dir)

        pass
