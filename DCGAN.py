import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn, utils
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import time

# Some great ideas taken from here : https://github.com/soumith/ganhacks


#####################
# Settings
num_train_images = 30000
batch_size = 64
n_inputs = 32 # Size of one-dimension of the image, the image shape should be n_inputsxn_inputs

z_size = 100

n_epochs = 300
dropout_probability = 0.5

# Two Timescale Update Rule (TTUR) : https://medium.com/beyondminds/advances-in-generative-adversarial-networks-7bad57028032
# "Typically, a slower update rule is used for the generator and a faster update rule is used for the discriminator"
learning_rate_discriminator = 0.0008
learning_rate_generator = 0.0002

beta_l = 0.3
beta_h = 0.999

data_dir = 'processed-celeba-small'
#####################


#####################
# Dataset
transform = transforms.Compose([transforms.CenterCrop(150), transforms.Resize(n_inputs), transforms.ToTensor()])

dataset = datasets.ImageFolder(data_dir, transform)
dataloader = utils.data.DataLoader(dataset, batch_size=batch_size, sampler=utils.data.SubsetRandomSampler([*range(num_train_images)]), num_workers=0)


#####################
# Functions

def init_weights_normal(m, mean=0.0, std=0.02):
    classname = m.__class__.__name__
    if 'conv' in classname.lower() or 'linear' in classname.lower():
        nn.init.normal_(m.weight.data, mean=mean, std=std)   

def calculate_loss(d_out, device='cpu', real=True, smoothing=False):
    # With Label smoothing : Salimans et. al. 2016
    # With Least Squares Loss : Mao et. al. 2016
    criterion = nn.MSELoss()
    
    if real:
        if smoothing : loss = criterion(d_out.squeeze(), (torch.rand(d_out.size(0))/2+0.7).to(device))
        else : loss = criterion(d_out.squeeze(), torch.ones(d_out.size(0)).to(device))
    else:
        if smoothing : loss = criterion(d_out.squeeze(), (torch.rand(d_out.size(0))*0.3).to(device))
        else : loss = criterion(d_out.squeeze(), torch.zeros(d_out.size(0)).to(device))    
        
    return loss


#####################
# Verify Dataset
dataiter = iter(dataloader)
images, labels = dataiter.next()

images = torchvision.utils.make_grid(images).numpy()
plt.imshow(np.transpose(images, (1, 2, 0)))


#####################
# Define Model
class Discriminator(nn.Module):

    def __init__(self, input_size, output_size, dropout_probability=0.5):
        super(Discriminator, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability
        
        # conv_output_size = (input_size - kernel_size + 2Padding)/stride + 1
        
        # 3xIxI
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        # 128x(I-2)x(I-2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=0, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(128)
        # 256x(I/2-2)x(I/2-2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(256)
        # 512x(I/2-4)x(I/2-4)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=0, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(512)
        # 1024x(I/4-3)x(I/4-3)
        
        self.ll2 = nn.Linear(512*(self.input_size//4-3)**2, output_size)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(self.dropout_probability)
        
    def forward(self, x):

        x = self.lrelu(self.conv1(x))
        x = self.dropout(x)
        x = self.lrelu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.lrelu(self.conv3(x))
        x = self.batch_norm3(x)
        x = self.dropout(x)
        x = self.lrelu(self.conv4(x))
        x = self.batch_norm4(x)
        x = self.dropout(x)
        x = F.sigmoid(self.ll2(x.view(-1,512*(self.input_size//4-3)**2)))
    
        return x
    
    
class Generator(nn.Module):

    def __init__(self, input_size, output_size, dropout_probability=0.5):
        super(Generator, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability

        # convt_output_size = strides * (input_size-1) + kernel_size - 2*padding
        
        self.ll1 = nn.Linear(input_size, 1024*(self.output_size//4-3)**2)
        # 1024x(O/4-3)x(O/4-3)
        self.convt1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(512)
        # 512x(O/2-4)x(O/2-4)
        self.convt2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=0, bias=False) 
        self.batch_norm2 = nn.BatchNorm2d(256) 
        # 256x(O/2-2)x(O/2-2)
        self.convt3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=0, bias=False) 
        self.batch_norm3 = nn.BatchNorm2d(128)
        # 128x(O-2)x(O-2)
        self.convt4 = nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=0, bias=False) 
        # 3xOxO
         
        self.tanh = nn.Tanh()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(self.dropout_probability)
        
    def forward(self, x):
        x = self.lrelu(self.ll1(x))
        x = self.dropout(x)
        x = self.lrelu(self.batch_norm1(self.convt1(x.view(-1,1024,self.output_size//4-3,self.output_size//4-3))))
        x = self.dropout(x)
        x = self.lrelu(self.batch_norm2(self.convt2(x)))
        x = self.dropout(x)
        x = self.lrelu(self.batch_norm3(self.convt3(x)))
        x = self.dropout(x)
        x = self.tanh(self.convt4(x))

        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
discriminator = Discriminator(input_size=n_inputs, output_size=1, dropout_probability=0.5).to(device)
generator = Generator(input_size=z_size, output_size=n_inputs, dropout_probability=0.5).to(device)

generator.apply(init_weights_normal)
discriminator.apply(init_weights_normal)

##################################
# Print to verify both models
print(discriminator)
print()
print(generator)
##################################


fixed_z = torch.randn(batch_size, z_size).to(device) # Better to sample from a gaussian distribution

# Use SGD for discriminator and ADAM for generator, See Radford et. al. 2015
d_optimizer = optim.SGD(discriminator.parameters(), lr=learning_rate_discriminator)
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate_generator, betas=(beta_l, beta_h))

start_time = time.time()
num_batches_per_epoch = num_train_images/batch_size
total_num_batches = n_epochs*num_batches_per_epoch

samples = []
losses = []
for epoch in range(1, n_epochs+1):
    train_running_loss = 0.0
    train_acc = 0.0
    discriminator.train()
    generator.train()
    
    
    
    for i, data in enumerate(dataloader,1):
        
        real_images, _ = data
        real_images = real_images.to(device)
        real_images = real_images*2-1  # Rescale input images from [0,1] to [-1, 1]

        z = torch.randn(batch_size, z_size).to(device) # Better to sample from a gaussian distribution

        # Generate fake images
        fake_images = generator(z)
        
        #################################
        # Train Discriminator
        
        d_optimizer.zero_grad()
        
        # Discriminator with real images
        d_real = discriminator(real_images)
        r_loss = calculate_loss(d_real, device=device, real=True, smoothing=True)

        # Discriminator with fake images
        d_fake = discriminator(fake_images)
        f_loss = calculate_loss(d_fake, device=device, real=False, smoothing=True)
        
        d_loss = r_loss + f_loss

        # Optimize Discriminator
        d_loss.backward()
        d_optimizer.step()
        
        #################################
        # Train Generator  
        
        g_optimizer.zero_grad()
        fake_images = generator(z)
        d_fake = discriminator(fake_images)
        g_loss = calculate_loss(d_fake, device=device, real=True, smoothing=True)

        # Optimize Generator
        g_loss.backward()
        g_optimizer.step()       
        
        # Print losses
        if i % (num_batches_per_epoch//5) == 0:
            completion = (((epoch-1)*num_batches_per_epoch) + i)/total_num_batches
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time * (1/completion - 1)
            
            em, es = divmod(elapsed_time, 60)
            eh, em = divmod(em, 60)
            
            rm, rs = divmod(remaining_time, 60)
            rh, rm = divmod(rm, 60)
            
            print('Epoch {:2d}/{} | Batch_id {:4d}/{:.0f} | d_loss: {:.4f} | g_loss: {:.4f} | Elapsed time : {:.0f}h {:02.0f}min {:02.0f}sec | Remaining time : {:.0f}h {:02.0f}min {:02.0f}sec'.format(epoch, n_epochs, i, num_batches_per_epoch, d_loss.item(), g_loss.item(), eh, em, es, rh, rm, rs))

    # Generate and save samples of fake images
    losses.append((d_loss.item(), g_loss.item()))

    generator.eval() 
    fake_images = generator(fixed_z)
    samples.append(fake_images.detach()[0:20])


#######################
# Print training losses
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("GAN Training Losses")
plt.legend()

############################################
# Print generated fake images through epochs
samples_arr = np.stack(samples, axis=0) # shape = [Epoch, Batch Size, Color Channel, Width, Height]

for epoch in range(1, samples_arr.shape[0], 10):
    fig, axes = plt.subplots(figsize=(15,2), nrows=1, ncols=7, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples_arr[epoch, 0:7, :, :, :]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(np.transpose(np.clip(((img+1)/2), a_min=0.,a_max=1.), (1, 2, 0)))
    fig.suptitle("Epoch : {}".format(epoch+1))
    

imgs = generator(torch.randn(batch_size, z_size).to(device))
img = imgs.detach()[1]
plt.imshow(np.transpose(np.clip(((img+1)/2), a_min=0.,a_max=1.), (1, 2, 0)))