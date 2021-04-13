import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
torch.manual_seed(0)
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import skimage
import cv2
from skimage import io, color
from Generator import Generator
from Discriminator import Discriminator
from utils import ColorDataset

adv_criterion = nn.BCEWithLogitsLoss() 
recon_criterion = nn.L1Loss() 
lambda_recon = 200

n_epochs = 100
input_dim = 1
output_dim = 2
real_dim = 3
display_step = 200
batch_size = 4
lr = 0.0002
target_shape = 256
device = 'cuda'
#########################Generator Loss ########################################
def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):

    fake = gen(condition)
    fake_disc = disc(fake, condition)
    gen_adv_loss = adv_criterion(fake_disc, torch.ones_like(fake_disc))
    gen_rec_loss = recon_criterion(real, fake)
    gen_loss = gen_adv_loss + lambda_recon * gen_rec_loss

    return gen_loss

############################# Train Function ############################################
def train(save_model=False):


    '''
    function taken and modified from the course: Apply Generative Adverserial Networks by DeepLearning.ai on coursera

    '''
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for ab, gray in tqdm(dataloader):

            #real = nn.functional.interpolate(real, size=target_shape)
            cur_batch_size = len(gray)
            condition = gray.to(device)
            real = ab.to(device)

            ### Update discriminator ###
            disc_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake = gen(condition)
            disc_fake_hat = disc(fake.detach(), condition) # Detach generator
            disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
            disc_real_hat = disc(real, condition)
            disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True) # Update gradients
            disc_opt.step() # Update optimizer

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ### Visualization code ###
            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                else:
                    print("Pretrained initial state")
                
                f, axrr = plt.subplots(2, batch_size)
                for i in range(2):
                    for j in range(batch_size):
                        x = torch.cat((gray[j],fake.detach().cpu()[j]),0).numpy().transpose((1,2,0))
                        x[:,:,0:1] = x[:, :, 0:1] * 100
                        x[:, :, 1:3] = x[:, :, 1:3] * 255 - 128

                        if i == 0:
                            axrr[i,j].imshow(gray.squeeze().numpy()[j], cmap = "gray")
                        else:
                            axrr[i,j].imshow(color.lab2rgb(x.astype(np.float64)))
                plt.show()


                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if save_model:
                    torch.save({'gen': gen.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc': disc.state_dict(),
                        'disc_opt': disc_opt.state_dict()
                    }, f"pix2pix_{cur_step}.pth")
            cur_step += 1


########################### Weights ##############################

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

############ Defining Generator and discriminator ############################
gen = Generator(input_dim, real_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(input_dim + output_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

######################## Dataset Preperation ###################################
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(0.2),
    transforms.RandomVerticalFlip(0.2),
    #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    #transforms.ToTensor(),
])

dataset = ColorDataset("./images/tiny-imagenet-200/test/images", transform=transform)


############################ Train #############################
train()

