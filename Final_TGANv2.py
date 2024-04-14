import torch
import torch.nn as nn
import numpy as np
from moviepy.editor import ImageSequenceClip
from IPython.display import Image
import os
import time 
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.autograd import profiler
import torch.nn.functional as F
import random
import torch.nn.init as init
import math

# Seed value
seed_value = 1701

# 1. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 2. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 3. Set `pytorch` pseudo-random generator at a fixed value
torch.manual_seed(seed_value)

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def adjust_learning_rate(optimizer, iteration, total_iterations, initial_lr):
    """Sets the learning rate to linearly decay to 0"""
    lr = initial_lr * (1 - iteration / (100000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_weights(m):
    # Check if the module has a 'custom_initialized' attribute
    if hasattr(m, 'custom_initialized') and m.custom_initialized:
        # Skip this module because it has custom initialization
        return

    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def genSamples(g, n=8, device=device):
    '''
    Generates an n by n grid of videos, given a generator g
    '''
    try:
        print("Generating samples...")
        with torch.no_grad():
            print("Calling the generator...")
            s = g(torch.rand((n**2, 256), device=device)*2-1)
            print("Generator call completed.")

            s = s.cpu().detach().numpy()
            print("Shape of s is ", s.shape)

        out = np.zeros((3, 16, 256*n, 256*n))

        for j in range(n):
            for k in range(n):
                out[:, :, 256*j:256*(j+1), 256*k:256*(k+1)] = s[j*n+k, :, :, :, :]
            
        out = out.transpose(1, 2, 3, 0)  # Transposing to correct dimensions for GIF
        out = (out + 1) / 2 * 255
        out = out.astype(int)

        print("Shape of out is ", out.shape)

        clip = ImageSequenceClip(list(out), fps=8)
        clip.write_gif('OptDis_ChainerINIT_URAND_200epoch_chainerbatch_40k_16frame.gif', fps=8)
        print("GIF successfully created and saved.")

    except Exception as e:
        print("An error occurred in genSamples function:")
        print(str(e))

class VideoDataset(Dataset):
    def __init__(self, directory, fraction=1.0, sub_sample_rate=2):
        self.directory = directory
        self.sub_sample_rate = sub_sample_rate
        all_files = [os.path.join(self.directory, file) for file in os.listdir(self.directory)]

        valid_files = []
        for file in all_files:
            try:
                if torch.load(file).shape == (16, 3, 256, 256):
                    valid_files.append(file)
            except Exception as e:
                print(f"Error loading file {file}: {e}")

        # Randomly select a fraction of the valid files
        selected_file_count = int(len(valid_files) * fraction)
        self.files = random.sample(valid_files, selected_file_count)

    def sub_sample(self, x):
        stride = self.sub_sample_rate
        offset = np.random.randint(stride)
        return x[:, offset::stride, :, :]
    
    def pooling(self, x, ksize):
        if ksize == 1:
            return x
        C, T, H, W = x.shape
        Hd = H // ksize
        Wd = W // ksize
        return x.view(C, T, Hd, ksize, Wd, ksize).mean(dim=(3, 5))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_tensor = torch.load(self.files[idx])
        video_tensor = video_tensor.transpose(0, 1)  # Transpose to (C, T, H, W)
        x1 = video_tensor
        x2 = self.sub_sample(x1)
        x3 = self.sub_sample(x2)
        x4 = self.sub_sample(x3)
        x1 = self.pooling(x1, 8)
        x2 = self.pooling(x2, 4)
        x3 = self.pooling(x3, 2)
        return (x1, x2, x3, x4)
    
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(GenBlock, self).__init__()

        self.custom_initialized = True

        #Residual Path
        self.residual_path = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=2e-5),
            nn.ReLU(),
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, eps=2e-5),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        # Initialize residual convolutions with GlorotUniform(sqrt(2))
        for m in self.residual_path.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('relu'))
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        #Skip Connection
        self.skip_connection = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) #Why kernel_size=1? #This MAY BE KEY?
        )

        # Initialize skip connection convolution with GlorotUniform
        for m in self.skip_connection.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)  # Default gain=1 for linear activation in skip connection
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.residual_path(x) + self.skip_connection(x)
    
class ConvLSTM2DCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTM2DCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.xi_conv = nn.Conv2d(self.input_channels, 4 * self.hidden_channels, self.kernel_size, stride=1, padding = self.padding)
        self.hi_conv = nn.Conv2d(self.hidden_channels, 4 * self.hidden_channels, self.kernel_size, stride=1, bias=False, padding = self.padding)

        # Apply Kaiming Normal initialization
        nn.init.kaiming_normal_(self.xi_conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.hi_conv.weight, mode='fan_in', nonlinearity='relu')
        self.custom_initialized = True


    def forward(self, input, hidden_state):
        hidden, cell = hidden_state
        xi_conv = self.xi_conv(input)
        hi_conv = self.hi_conv(hidden)

        xi_i, xi_f, xi_o, xi_g =  torch.split(xi_conv, self.hidden_channels, dim=1)
        hi_i, hi_f, hi_o, hi_g = torch.split(hi_conv, self.hidden_channels, dim=1)
        
        i = torch.sigmoid(xi_i + hi_i) #ci
        f = torch.sigmoid(xi_f + hi_f) #cf
        o = torch.sigmoid(xi_o + hi_o) #co
        g = torch.tanh(xi_g + hi_g) 
        cell = f * cell + i * g #cc
        hidden = o * torch.tanh(cell)
        return hidden, cell

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.xi_conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.xi_conv.weight.device))
    
class Generator(nn.Module):
    def __init__(self, noise_dim, feature_map_channels, video_height, video_width, n_frames, stride=1, max_offset=0):
        super().__init__()
        self.noise_dim = noise_dim
        self.feature_map_channels = feature_map_channels
        self.video_height = video_height
        self.video_width = video_width
        self.n_frames = n_frames
        self.stride = stride
        self.max_offset = max_offset



        #Relu Layer
        self.relu = nn.ReLU()

        #Tanh Layer
        self.tanh = nn.Tanh()

        #FC Layer
        fc_output_size = feature_map_channels * (video_height//64) * (video_width//64)  
        self.fc = nn.Linear(noise_dim, fc_output_size)
        # Apply Kaiming Normal initialization to FC layer
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc.bias, 0)

        #ConvLSTM
        self.convLSTM = ConvLSTM2DCell(feature_map_channels, feature_map_channels, 3)

        #Level 1
        self.block1 = GenBlock(feature_map_channels, feature_map_channels//2)
        self.block2 = GenBlock(feature_map_channels//2, feature_map_channels//4)
        self.block3 = GenBlock(feature_map_channels//4, feature_map_channels//8)
        self.b3 = nn.BatchNorm2d(feature_map_channels//8, eps=2e-5)
        self.c3 = nn.Conv2d(feature_map_channels//8, 3, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.c3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.c3.bias, 0)

        #Level 2
        self.block4 = GenBlock(feature_map_channels//8, feature_map_channels//16)
        self.b4 = nn.BatchNorm2d(feature_map_channels//16, eps=2e-5)
        self.c4 = nn.Conv2d(feature_map_channels//16, 3, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.c4.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.c4.bias, 0)

        #Level 3
        self.block5 = GenBlock(feature_map_channels//16, feature_map_channels//32)
        self.b5 = nn.BatchNorm2d(feature_map_channels//32, eps=2e-5)
        self.c5 = nn.Conv2d(feature_map_channels//32, 3, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.c5.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.c5.bias, 0)

        #Level 4
        self.block6 = GenBlock(feature_map_channels//32, feature_map_channels//32)
        self.b6 = nn.BatchNorm2d(feature_map_channels//32, eps=2e-5)
        self.c6 = nn.Conv2d(feature_map_channels//32, 3, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.c6.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.c6.bias, 0)

    #Creating initial val for clstm
    def make_in_val(self, t, z):
        N = z.size(0) #HOW DOES THIS WORK?  #N is the batch size
        if t == 0:
            #print("t is 0")
            return self.fc(z).view(N , self.feature_map_channels, self.video_height//64, self.video_width//64)
        else:
            #print("t is not 0")
            return torch.zeros(N, self.feature_map_channels, self.video_height//64, self.video_width//64, device=device)
        
    def separate(self, x, N):
        NT, C, H, W = x.shape
        T = NT // N
        x = x.view(N, T, C, H, W)
        return x

    def join(self, x):
        N, T, C, H, W = x.shape
        return x.reshape(N*T, C, H, W), N
            
    def sub_sample(self, x, frame=2):
        return x[:, np.random.randint(frame)::frame, :, :, :]
        
    def render(self, link_bn, link_conv, x, N):
        NT, C, H, W =  x.shape
        T = NT // N
        x = self.relu(link_bn(x))
        x = link_conv(x)
        H, W = x.shape[2:]
        x = self.tanh(x).view(N, T, 3, H, W)
        x =  x.permute(0, 2, 1, 3, 4)
        return x
        
    def forward(self, z, n_frames=None):
        #Stride and offset
        n_frames = n_frames if n_frames is not None else self.n_frames  
        stride = self.stride if self.training else 1
        offset = np.random.randint(self.max_offset + 1) if self.training else 0

        #Initialize hidden and cell state for ConvLSTM
        #print("Initializing hidden and cell state for ConvLSTM")
        hidden_cell_state = self.convLSTM.init_hidden(z.size(0), (self.video_height//64, self.video_width//64))
        #print("Hidden and cell state initialized")

        #Generate Frames with ConvLSTM
        outputs = []
        total_frames = offset + (n_frames-1)*stride + 1
        for t in range(total_frames):
            x = self.make_in_val(t, z)
            hidden_cell_state = self.convLSTM(x, hidden_cell_state)
            x = hidden_cell_state[0]
            td = t - offset
            if td % stride == 0 and td >= 0:
                outputs.append(x)
        #print("Frames generated with ConvLSTM")

        #Run frames through the network
        N, C, H, W = outputs[0].shape
        x = torch.stack(outputs, dim=1)
        x = x.view(N*n_frames, C, H, W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        #During training we sub-sample and render the frames in 4 steps, returning 4 rendered videos of varying frame length and resolution
        if self.training:
            x1 = self.render(self.b3, self.c3, x, N)
            h, N = self.join(self.sub_sample(self.separate(x, N)))
            h = self.block4(h)

            x2 = self.render(self.b4, self.c4, h, N)
            h, N = self.join(self.sub_sample(self.separate(h, N)))
            h = self.block5(h)

            x3 = self.render(self.b5, self.c5, h, N)
            h, N = self.join(self.sub_sample(self.separate(h, N)))
            h = self.block6(h)

            x4 = self.render(self.b6, self.c6, h, N)
            return (x1, x2, x3, x4)
        
        #During inference we do not sub-sample, instead we pass frames through all blocks and only render at the end, returning a single video
        else:
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)

            return self.render(self.b6, self.c6, x, N)

class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DisBlock, self).__init__()

        #Define the layers
        self.relu = nn.ReLU()
        self.c1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.c_sc = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.custom_initialized = True

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize c1 and c2 with GlorotUniform math.sqrt(2)
        nn.init.xavier_uniform_(self.c1.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.c2.weight, gain=math.sqrt(2))

        # Initialize c_sc with standard GlorotUniform
        nn.init.xavier_uniform_(self.c_sc.weight)

        if self.c1.bias is not None:
            nn.init.constant_(self.c1.bias, 0)
        if self.c2.bias is not None:
            nn.init.constant_(self.c2.bias, 0)
        if self.c_sc.bias is not None:
            nn.init.constant_(self.c_sc.bias, 0)
            

    #Down Sample
    def down_sample(self, x):
        # Determine if padding is needed based on each dimension's size
        pad = [(0 if k % 2 == 0 else 1) for k in x.shape[2:]]  # For depth, height, width
    
        # Reverse the order for F.pad (D, H, W) -> (W, H, D)
        pad = pad[::-1]
    
        # Convert to the format expected by F.pad: (padLeft, padRight, padTop, padBottom, padFront, padBack)
        pad_tuple = (pad[0], pad[0], pad[1], pad[1], pad[2], pad[2])
    
        # Apply padding manually
        x_padded = F.pad(x, pad_tuple, mode='constant', value=0)
    
        # Perform average pooling without additional padding
        ksize = [(2 if k > 1 else 1) for k in x.shape[2:]]
        return F.avg_pool3d(x_padded, kernel_size=ksize, stride=ksize, padding=0)

    
    #Residual Path
    def residual_path(self, x):   
        x = self.relu(x)
        x = self.c1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.down_sample(x)
        return x

        
    #Skip Connection
    def skip_connection(self, x):
        return self.down_sample(self.c_sc(x))

        
    def forward(self, x):
        return self.residual_path(x) + self.skip_connection(x)
        
class OptimizedDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OptimizedDisBlock, self).__init__()

        #Define the layers
        self.relu = nn.ReLU()
        self.c1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.c_sc = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize c1 and c2 with GlorotUniform math.sqrt(2)
        nn.init.xavier_uniform_(self.c1.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.c2.weight, gain=math.sqrt(2))

        # Initialize c_sc with standard GlorotUniform
        nn.init.xavier_uniform_(self.c_sc.weight)

        if self.c1.bias is not None:
            nn.init.constant_(self.c1.bias, 0)
        if self.c2.bias is not None:
            nn.init.constant_(self.c2.bias, 0)
        if self.c_sc.bias is not None:
            nn.init.constant_(self.c_sc.bias, 0)

    #Residual Path
    def residual_path(self, x):
        x = self.c1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = F.avg_pool3d(x, kernel_size=(1, 2, 2))
        return x
        
    #Skip Connection
    def skip_connection(self, x):
        x = F.avg_pool3d(x, kernel_size=(1, 2, 2))
        return self.c_sc(x)
        
    def forward(self, x):
        return self.residual_path(x) + self.skip_connection(x)
        
class ResNet3D(nn.Module): #Expects shape of (N, C, T, H, W)
    def __init__(self, in_channels, mid_ch=64):
        super(ResNet3D, self).__init__()

        #Define the layers
        self.block1 = OptimizedDisBlock(in_channels, mid_ch)
        self.block2 = DisBlock(mid_ch, mid_ch*2)
        self.block3 = DisBlock(mid_ch*2, mid_ch*4)
        self.block4 = DisBlock(mid_ch*4, mid_ch*8)
        self.block5 = DisBlock(mid_ch*8, mid_ch*16)
        self.fc = nn.Linear(mid_ch*16, 1)
        self.relu = nn.ReLU()

        init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            init.constant_(self.fc.bias, 0)

    def extract_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.relu(x)
        x = x.sum(dim = [2,3,4])
        return x
    
    def forward(self, x):
        x = self.extract_features(x)
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, n_levels=4):
        super(Discriminator, self).__init__()
    
        self.levels = []
        for i in range(n_levels):
            level = ResNet3D(in_channels)
            setattr(self, f'level_{i}', level)
            self.levels.append(level)

    def forward(self, xs):
        assert(len(xs) == len(self.levels))
        outputs = [level(x) for level, x in zip(self.levels, xs)]
        outputs = torch.stack(outputs, dim=0) #MAKE SURE THIS IS STACKED RIGHT
        return outputs


def custom_collate_fn(batch):
    # Batch together each level of the tuples
    batched_levels = list(zip(*batch))
    # Stack tensors in each level
    batched_levels = [torch.stack(level) for level in batched_levels]
    return batched_levels


#Create a training function
def train(train_values, gen_values, disc_values, gen_lr=0.0001, dis_lr=0.0001, device=device, save_interval=50):
    iteration = 0

    #Extracting values from gen_values and insantiate the generator
    z_dim, feature_map_channels, video_height, video_width, n_frames = gen_values
    gen = Generator(z_dim, feature_map_channels, video_height, video_width, n_frames).to(device)

    directory, n_epochs, batch_size, n_critic, dataset_train_fraction, betas, epsilon, lambda_gp = train_values

    #Extracting values from train_values and instantiate the discriminator
    in_channels = disc_values

    #Instantiate the discriminator
    disc = Discriminator(in_channels).to(device)


    #Instantiate the optimizers
    gen_opt = torch.optim.Adam(gen.parameters(), lr=gen_lr, betas=betas, eps=epsilon)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=dis_lr, betas=betas, eps=epsilon)

    #Create dataloader
    video_dataset = VideoDataset(directory, dataset_train_fraction)
    dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    #Create dataset preprocessor

    #Calculate total number of iterations
    num_examples = len(video_dataset)
    n_iterations = n_epochs * (num_examples // batch_size)


    print("Starting Training...")
    for epoch in range(n_epochs):
        start_time = time.time()
        total_gen_loss, total_disc_loss, total_loss_real = 0, 0, 0
        num_batches = 0

        for batch in dataloader:
            #Adjust learning rate
            adjust_learning_rate(gen_opt, iteration, n_iterations, gen_lr)
            adjust_learning_rate(disc_opt, iteration, n_iterations, dis_lr)

            iteration += 1
            x_real = tuple(level.to(device) for level in batch)
            num_batches += 1
            current_batch_size = batch[0].size(0)

            #Generating reals and fakes
            # Generating reals
            for x in x_real:
                x.requires_grad_(True)
            y_real = disc(x_real)
            
            # Generating fakes for generator update
            #noise = torch.randn(current_batch_size, z_dim, device=device)
            noise = torch.FloatTensor(current_batch_size, z_dim).uniform_(-1, 1).to(device)
            x_fake_gen_update = gen(noise, n_frames=n_frames)  # Keep this version for generator update
            y_fake_gen_update = disc(x_fake_gen_update)  # This is used for calculating gen_loss
            
            # Calculate generator loss and update
            gen_opt.zero_grad()
            gen_loss = F.softplus(-y_fake_gen_update).mean()  # Example generator loss
            gen_loss.backward()
            gen_opt.step()
            total_gen_loss += gen_loss.item()
            
            # Generating fakes for discriminator update (use already generated x_fake_gen_update but detached)
            x_fake_disc_update = tuple(tensor.detach() for tensor in x_fake_gen_update)  # Detach this version for discriminator update
            y_fake_disc_update = disc(x_fake_disc_update)  # This is used for calculating disc_loss
            
            # Calculate discriminator losses
            loss_real = F.softplus(-y_real).mean()
            loss_fake = F.softplus(y_fake_disc_update).mean()
            disc_loss = loss_real + loss_fake

            # Backpropagate discriminator loss
            #disc_opt.zero_grad()
            #disc_loss.backward(retain_graph=True)  # Retain graph for gradient penalty

            total_loss_real += loss_real.item()
            
            # If applying gradient penalty or additional terms, ensure they are correctly calculated here
            #Gettings the gradients with respect to real inputs
            xd, yd = x_real, y_real
            xd = list(xd) if isinstance(xd, (list, tuple)) else [xd]
            #for x in xd:
                #x = x.requires_grad_(True)
            
            dydxs = []
            loss_gp = 0
            for i in range(len(yd)):
                dydx = torch.autograd.grad(outputs=yd[i], inputs=xd[i], grad_outputs=torch.ones_like(yd[i]), create_graph=True, retain_graph=True, only_inputs=True)[0]
                dydxs.append(dydx)
                loss = torch.sum(dydx.square())
                loss_gp += loss
            loss_gp = lambda_gp * loss_gp
            
            
            # Update discriminator
            disc_opt.zero_grad()
            disc_loss.backward(retain_graph=True)  # Retain graph for gradient penalty
            loss_gp.backward()
            disc_opt.step()
            disc_opt.zero_grad()

            total_disc_loss += disc_loss.item()

        avg_grads = dydx.mean()
        avg_gen_loss = total_gen_loss / num_batches
        avg_disc_loss = total_disc_loss / num_batches
        avg_loss_real = total_loss_real / num_batches

        # After each epoch, check if we need to save the model
        if (epoch + 1) % save_interval == 0 or epoch + 1 == n_epochs:
            save_path = f"generator_epoch_{epoch+1}.pth"
            #torch.save(gen.state_dict(), save_path)
            #print(f"Saved generator to {save_path}")
        
        print(f"Epoch {epoch+1}/{n_epochs} complete. Average Losses: gen={avg_gen_loss}, disc={avg_disc_loss}, real={avg_loss_real}, loss_gp = {loss_gp}, grad_avg = {avg_grads}")
        print(f"Time elapsed: {(time.time() - start_time)} seconds")

    return gen


#Go to project directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

#Set generator values
z_dim = 256
feature_map_channels = 1024
video_height = 256
video_width = 256
n_frames = 16
gen_values = (z_dim, feature_map_channels, video_height, video_width, n_frames)

#Set discriminator values
in_channels = 3
disc_values = in_channels

#Set training values
directory = '16_256p Frame Tensors'
n_epochs = 200
batch_size = 16
n_critic = 1
dataset_train_fraction = 0.4
betas = (0, 0.9)
epsilon = 0.00000008
lambda_gp = 0.5
gen_lr = 0.0001
dis_lr = 0.0001
train_values = (directory, n_epochs, batch_size, n_critic, dataset_train_fraction, betas, epsilon, lambda_gp)

#Train the generator
gen = train(train_values, gen_values, disc_values, gen_lr, dis_lr, device)

#Generates Samples
gen.eval()
genSamples(gen, n=4, device=device)