import torch
from torch import nn 
import torch.nn.functional as F
import pytorch_lightning as pl

# import pyplot for visualization
import matplotlib.pyplot as plt 

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.4)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # flatten the tensor for FC layers
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)
    

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7*7*64)  # [n, 256, 7, 7]
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2) # [n, 64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2) # [n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  # [n, 1, 28, 28]
    

    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)  #256
        
        # Upsample (transposed conv) 16x16 (64 feature maps)
        x = self.ct1(x)
        x = F.relu(x)
        
        # Upsample to 34x34 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)
        
        # Convolution to 28x28 (1 feature map)
        return self.conv(x)
    
class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.0002) -> None:
        super().__init__()
        # save hyper-parameters for later use
        self.save_hyperparameters()
        
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()   
        
        self.validation_z = torch.randn(6, self.hparams.latent_dim) 
        
    def forward(self, z):
        return self.generator(z)    
        
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        # this function will be taken care of by pl, also other functions, see the document
        real_imgs, _ = batch 
        
        # noise 
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(real_imgs)
        
        # train generator: max log(D(G(z))) 
        if optimizer_idx == 0:
            fake_imgs = self(z)
            y_hat = self.discriminator(fake_imgs)
            y = torch.ones(real_imgs.size(0), 1)
            y = y.type_as(real_imgs)
            
            g_loss = self.adversarial_loss(y_hat, y)
            
            log_dict = {"g_loss": g_loss}
            # log the results to progress bar and tensorboard
            return {"loss":g_loss, "progress_bar": log_dict, "log":log_dict}
        #  max log(D(x)) + log(1-D(G(z)))
        if optimizer_idx == 1:
            # how well can it label real and fake
            # real
            y_hat_real = self.discriminator(real_imgs)
            
            y_real = torch.ones(real_imgs.size(0), 1)
            y_real = y_real.type_as(real_imgs)
            
            real_loss = self.adversarial_loss(y_hat_real, y_real)
            
            # fake
            y_hat_fake = self.discriminator(self(z).detach())
            
            y_fake = torch.zeros(real_imgs.size(0), 1)
            y_fake = y_fake.type_as(real_imgs)
            
            fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
            
            d_loss = (real_loss + fake_loss) / 2
            
            log_dict = {"d_loss": d_loss}
            # log the results to progress bar and tensorboard
            return {"loss":d_loss, "progress_bar": log_dict, "log":log_dict}
            
            
            
                
    def configure_optimizers(self):
        lr = self.hparams.lr
        # optimizer for generator and discri.
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        # return: first one is for optimizer, second one is for scheduler
        return [opt_g, opt_d], []
    
    def plot_imgs(self, save=True):
        z = self.validation_z.type_as(self.generator.lin1.weight)
        sample_imgs = self(z).cpu()
        
        print("epoch", self.current_epoch)
        fig = plt.figure()
        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(sample_imgs.detach()[i, 0, :, :], cmap = "gray_r", interpolation="none")
            plt.title("Generated Data")
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
        if save:
            plt.savefig("./savedImgs/gan_epoch_{}".format(self.current_epoch))
        else:
            plt.show()
        
    def on_train_epoch_end(self) -> None:
        self.plot_imgs()
        
    
    
    
    
