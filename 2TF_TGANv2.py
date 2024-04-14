import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
import tensorflow.keras.layers as L #type: ignore
import tensorflow.keras.initializers as initializers #type: ignore
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.initializers import VarianceScaling #type: ignore
from tensorflow.keras.initializers import GlorotUniform #type: ignore
import math
import numpy as np
from moviepy.editor import ImageSequenceClip
import random
import time
import sys

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(f"Tensorflow version: {tf.__version__}")
tf.keras.backend.set_floatx('float32')



def genSamples(g, n=8):
    """
    Generates an n by n grid of videos, given a generator g (TensorFlow model)
    """
    try:
        print("Generating samples...")
        # TensorFlow models operate in eager execution by default, no need for torch.no_grad equivalent
        print("Calling the generator...")
        # Generating random input for the generator
        z = tf.random.uniform((n**2, 256), minval=-1, maxval=1)
        s = g(z, training=False)
        print("Generator call completed.")

        # Converting the TensorFlow tensor to a numpy array
        s = s.numpy()
        #Shape
        print("Shape of s is ", s.shape)

        out = np.zeros((16, 256*n, 256*n, 3))

        for j in range(n):
            for k in range(n):
                out[:, 256*j:256*(j+1), 256*k:256*(k+1), :] = s[j*n+k, :, :, :, :] #WRONG SHAPE for s probably


        #out = out.transpose(3, 1, 2, 0)  # Transposing to correct dimensions for GIF
        out = (out + 1) / 2 * 255
        out = out.astype(np.uint8)

        print("Shape of out is ", out.shape)

        clip = ImageSequenceClip(list(out), fps=8)
        clip.write_gif('generated_video.gif', fps=8)
        print("GIF successfully created and saved.")

    except Exception as e:
        print("An error occurred in genSamples function:")
        print(str(e))

def genSample(gen): #Returns tuple of tensoors (single generator output)
    z = tf.random.uniform((1, 256), minval=-1, maxval=1)
    x = gen(z)
    return x

class VideoDataset():
    def __init__(self, directory, fraction=0.2, sub_sample_rate=2):
        self.directory = directory
        self.fraction = fraction
        self.sub_sample_rate = sub_sample_rate
        all_files = [os.path.join(self.directory, file) for file in os.listdir(self.directory)]

        valid_files = []
        for file in all_files:
            try:
                # Read the serialized tensor from file
                serialized_tensor = tf.io.read_file(file)
                # Deserialize the tensor
                tensor = tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)  # Adjust dtype if necessary
                # Validate the shape of the tensor
                if tensor.shape == (16, 256, 256, 3):
                    valid_files.append(file)
            except Exception as e:
                print(f"Error loading file {file}: {e}")

        # Randomly select a fraction of the valid files
        selected_file_count = int(len(valid_files) * fraction)
        self.files = random.sample(valid_files, selected_file_count)

    def sub_sample(self, x):
        stride = self.sub_sample_rate
        offset = np.random.randint(stride)
        return x[offset::stride, :, :, :]
    
    def pooling(self, x, ksize):
        if ksize == 1:
            return x
        T, H, W, C = x.shape
        Hd = H // ksize
        Wd = W // ksize
        # Reshape the tensor to merge the spatial dimensions into the pooling blocks
        x_reshaped = tf.reshape(x, (T, Hd, ksize, Wd, ksize, C))
        # Take the mean across the dimensions 3 and 5, which are the spatial dimensions within each block
        pooled_x = tf.reduce_mean(x_reshaped, axis=[2, 4])
        return pooled_x
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        serialized_tensor = tf.io.read_file(self.files[idx])
        video_tensor = tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)
        x1 = video_tensor
        x2 = self.sub_sample(x1)
        x3 = self.sub_sample(x2)
        x4 = self.sub_sample(x3)
        x1 = self.pooling(x1, 8)
        x2 = self.pooling(x2, 4)
        x3 = self.pooling(x3, 2)
        return (x1, x2, x3, x4)
    
    def __iter__(self):
        #Make the dataset iterable, allowing it to be used directly with tf.data.Dataset.from_generator.
        for idx in range(len(self)):
            yield self[idx]

class GenBlock(Model):
    def __init__(self, out_channels, scale_factor=2):
        super(GenBlock, self).__init__()
        self.scale_factor = scale_factor
        self.out_channels = out_channels
        initializer1 = VarianceScaling(scale=2., mode='fan_in', distribution='uniform')
        initializer2 = GlorotUniform()

        self.residual_path = tf.keras.Sequential([
            L.BatchNormalization(epsilon=2e-5),
            L.ReLU(),
            L.UpSampling2D(size=(scale_factor, scale_factor), interpolation='nearest'),
            L.Conv2D(out_channels, kernel_size=3, strides=1, padding='same',  kernel_initializer=initializer1),
            L.BatchNormalization(epsilon=2e-5),
            L.ReLU(),
            L.Conv2D(out_channels, kernel_size=3, strides=1, padding='same',  kernel_initializer=initializer1),
        ])

        self.shortcut = tf.keras.Sequential([
            L.UpSampling2D(size=(scale_factor, scale_factor), interpolation='nearest'),
            L.Conv2D(out_channels, kernel_size=1, strides=1, padding='valid',  kernel_initializer=initializer2),
        ])

    def call(self, x, training=None):
        x_res = self.residual_path(x, training=training)
        x_shortcut = self.shortcut(x, training=training)
        x = x_res + x_shortcut
        return x

class ConvLSTM2DCell(Model):
    def __init__(self, hidden_channels, kernel_size):
        super(ConvLSTM2DCell, self).__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.xi_conv = L.Conv2D(4 * hidden_channels, kernel_size=kernel_size, strides = 1, padding='same', kernel_initializer='he_normal')
        self.hi_conv = L.Conv2D(4 * hidden_channels, kernel_size=kernel_size, strides = 1, padding='same', kernel_initializer='he_normal')

    def call(self, inputs, states):
        hidden, cell = states
        xi_conv = self.xi_conv(inputs)
        hi_conv = self.hi_conv(hidden)

        xi, xf, xo, xg = tf.split(xi_conv, 4, axis=-1)
        hi, hf, ho, hg = tf.split(hi_conv, 4, axis=-1)

        i = tf.sigmoid(xi + hi) #ci
        f = tf.sigmoid(xf + hf) #cf
        o = tf.sigmoid(xo + ho) #co
        g = tf.tanh(xg + hg) 

        cell = f * cell + i * g #cc
        hidden = o * tf.tanh(cell) #ch
        return hidden, cell
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (tf.zeros((batch_size, height, width, self.hidden_channels)), tf.zeros((batch_size, height, width, self.hidden_channels)))
        
class Generator(Model):
    def __init__(self, noise_dim, feature_map_channels, feature_map_size, video_height, video_width, n_frames, stride=1, max_offset=0, training=True):
        super().__init__()
        self.noise_dim = noise_dim
        self.feature_map_channels = feature_map_channels
        self.feature_map_size = feature_map_size
        self.video_height = video_height
        self.video_width = video_width
        self.n_frames = n_frames
        self.stride = stride
        self.max_offset = max_offset
        self.training = training

        #ReLU layer
        self.relu = L.ReLU()

        #Tanh Layer
        self.tanh = L.Activation('tanh')

        #FC Layer
        fc_output_size = feature_map_size * feature_map_size * feature_map_channels
        self.fc = L.Dense(fc_output_size, kernel_initializer='he_normal')

        #ConvLSTM2DCell
        self.conv_lstm = ConvLSTM2DCell(hidden_channels=feature_map_channels, kernel_size=3)

        #Level 1
        self.block1 = GenBlock(feature_map_channels//2, scale_factor=2)
        self.block2 = GenBlock(feature_map_channels//4, scale_factor=2)
        self.block3 = GenBlock(feature_map_channels//8, scale_factor=2)
        self.b3 = L.BatchNormalization(epsilon=2e-5)
        self.c3 = L.Conv2D(3, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')

        #Level 2
        self.block4 = GenBlock(feature_map_channels//16, scale_factor=2)
        self.b4 = L.BatchNormalization(epsilon=2e-5)
        self.c4 = L.Conv2D(3, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')

        #Level 3
        self.block5 = GenBlock(feature_map_channels//32, scale_factor=2)
        self.b5 = L.BatchNormalization(epsilon=2e-5)
        self.c5 = L.Conv2D(3, kernel_size=3, strides=1, padding='same',  kernel_initializer='he_normal')

        #Level 4
        self.block6 = GenBlock(feature_map_channels//32, scale_factor=2)
        self.b6 = L.BatchNormalization(epsilon=2e-5)
        self.c6 = L.Conv2D(3, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')


    #Creating initial value for CLSTM
    def make_in_val(self, t, z):
        N = z.shape[0]
        if t == 0:
            z =  self.fc(z)
            return tf.reshape(z, (N, self.feature_map_size, self.feature_map_size, self.feature_map_channels))
        else:
            return tf.zeros((N, self.feature_map_size, self.feature_map_size, self.feature_map_channels))
        
    def separate(self, x, N):
        NT, H, W, C = x.shape
        T = NT // N
        x = tf.reshape(x, (N, T, H, W, C))
        return x
    
    def join(self, x):
        N, T, H, W, C = x.shape
        x = tf.reshape(x, (N * T, H, W, C))
        return x, N
    
    def sub_sample(self, x, frame=2):
        return x[:, np.random.randint(frame)::frame, :, :, :]
    
    def render(self, link_bn, link_conv, x, N):
        NT, H, W, C = x.shape
        T = NT // N
        x = self.relu(link_bn(x))
        x = link_conv(x)
        H, W = x.shape[1:3]
        x = tf.reshape(self.tanh(x), (N, T, H, W, 3))
        return x

    def call(self, z):
        #Stride and offset

        n_frames = self.n_frames if self.n_frames is not None else self.n_frames  
        stride = self.stride if self.training else 1
        offset = np.random.randint(self.max_offset + 1) if self.training else 0


        #Initialize hidden and cell states for ConvLSTM
        hidden_cell_state = self.conv_lstm.init_hidden(z.shape[0], (self.feature_map_size, self.feature_map_size))

        #Generate frames with cocnvLSTM
        outputs = []
        total_frames = offset +(n_frames - 1) * stride + 1
        for t in range(total_frames):
            x = self.make_in_val(t, z)
            hidden_cell_state = self.conv_lstm(x, hidden_cell_state)
            x = hidden_cell_state[0]
            td = t - offset
            if td % stride == 0 and td >= 0:
                outputs.append(x)

        #Run frames through the network
        N, H, W, C = outputs[0].shape
        x = tf.concat(outputs, 0) #Is 0 correct?
        #x = tf.reshape(x, (N * n_frames, H, W, C))
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
            return x1, x2, x3, x4
        
        #During inference we do not sub-sample, instead we pass frames through all blocks and only render at the end, returning a single video
        else:
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.render(self.b6, self.c6, x, N)
            print(f"Generator out shape is: {x.shape}")
            return x

class DisBlock(Model):
    def __init__(self, in_channels, out_channels):
        super(DisBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        initializer = VarianceScaling(scale=2., mode='fan_in', distribution='uniform')

        #Define the layers
        self.relu = L.ReLU()
        self.c1 = L.Conv3D(in_channels, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)
        self.c2 = L.Conv3D(out_channels, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)
        self.c_sc = L.Conv3D(out_channels, kernel_size=1, strides=1, padding='valid', kernel_initializer='glorot_uniform')

    def down_sample(self, x):
        # Determine if padding is needed based on each dimension's size
        pad = [(0 if k % 2 == 0 else 1) for k in x.shape[2:5]]  # For depth, height, width
    
        # Reverse the order for F.pad (T, H, W) -> (W, H, T)
        #pad = pad[::-1]
    
        # Convert to the format expected by F.pad: (padLeft, padRight, padTop, padBottom, padFront, padBack)
        pad_tuple = (pad[0], pad[1], pad[2])
    
        # Apply padding manually
        x_padded = L.ZeroPadding3D(padding=pad_tuple)(x)
    
        # Perform average pooling without additional padding
        ksize = [(2 if k > 1 else 1) for k in x.shape[2:]]
        return L.AveragePooling3D(pool_size=ksize, strides=ksize)(x_padded) 

    #Residual Path
    def residual_path(self, x):
        x = self.relu(x)
        x = self.c1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.down_sample(x)
        return x

    #Skip Connection
    def shortcut(self, x):
        return self.down_sample(self.c_sc(x))

    def call(self, x):
        return self.residual_path(x) + self.shortcut(x)   
    
class OptimizedDisBlock(Model):
    def __init__(self, in_channels, out_channels):
        super(OptimizedDisBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        initializer = VarianceScaling(scale=2., mode='fan_in', distribution='uniform')

        #Define the layers
        self.relu = L.ReLU()
        self.c1 = L.Conv3D(out_channels, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)
        self.c2 = L.Conv3D(out_channels, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)
        self.c_sc = L.Conv3D(out_channels, kernel_size=1, strides=1, padding='valid', kernel_initializer='glorot_uniform')
        self.avgpool3d = L.AveragePooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))

        #Residual Path
    def residual_path(self, x):
        x = self.c1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.avgpool3d(x)
        return x
    
    #Skip Connection
    def shortcut(self, x):
        x = self.avgpool3d(x)
        return self.c_sc(x)
    
    def call(self, x):
        return self.residual_path(x) + self.shortcut(x)
    
class ResNet3D(Model):
    def __init__(self, in_channels, mid_ch=64):
        super(ResNet3D, self).__init__()
        self.in_channels = in_channels
        self.mid_ch = mid_ch

        #Define the layers
        self.block1 = OptimizedDisBlock(in_channels, mid_ch)
        self.block2 = DisBlock(mid_ch, mid_ch*2)
        self.block3 = DisBlock(mid_ch*2, mid_ch*4)
        self.block4 = DisBlock(mid_ch*4, mid_ch*8)
        self.block5 = DisBlock(mid_ch*8, mid_ch*16)
        self.fc = L.Dense(1, kernel_initializer='glorot_uniform')
        self.relu = L.ReLU()

    def extract_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.relu(x)
        #Sum along dimension 2,3,4
        x = tf.reduce_sum(x, axis=(2, 3, 4)) #Correct?
        return x
    
    def call(self, x):
        x = self.extract_features(x)
        x = self.fc(x)
        return x

class Discriminator(Model):
    def __init__(self, in_channels, n_levels=4):
        super(Discriminator, self).__init__()

        self.levels = []
        for i in range(n_levels):
            level = ResNet3D(in_channels)
            self.levels.append(level)
        
    def call(self, xs):
        assert(len(xs) == len(self.levels))
        outputs = [level(x) for level, x in zip(self.levels, xs)]
        #Stack ouputs
        return tf.stack(outputs, axis=0) #Correct?

def compute_gradient_penalty(inputs, disc_outputs, lam):
    gradient_penalty_sum = 0.0  # Initialize the sum of gradient penalties
    
    for input_tensor in inputs:
        # Calculate gradients of discriminator outputs with respect to each input tensor
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            gradients = tape.gradient(disc_outputs, input_tensor)

            # Compute the L2 norm of the gradients for this input tensor
            gradients_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))

            # Calculate the gradient penalty for this input tensor and add to the sum
            gradient_penalty_sum += tf.reduce_mean(tf.square(gradients_l2_norm - 1.0))
    
    # Aggregate the penalties across all input tensors and scale by lambda
    total_gradient_penalty = lam * gradient_penalty_sum
    return total_gradient_penalty

def check_and_report(tensor, name):
    if tf.reduce_any(tf.math.is_nan(tensor)) or tf.reduce_any(tf.math.is_inf(tensor)):
        print(f"NaN or Inf detected in {name}")
    else:
        print(f"No issues in {name}")

def train(train_values, gen_values, disc_values, gen_lr=0.0001, disc_lr=0.0001, save_interval=50):

    iteration = 0

    #Extract values from gen_values and instantiate the generator
    z_dim, feature_map_channels, feature_map_size, video_height, video_width, n_frames = gen_values
    gen = Generator(z_dim, feature_map_channels, feature_map_size, video_height, video_width, n_frames)

    directory, n_epochs, batch_size, n_critic, dataset_train_faction, betas, epsilon, lambda_gp = train_values

    #Extract values from disc_values and instantiate the discriminator
    in_channels = disc_values

    disc = Discriminator(in_channels)

    #Instantiate optimizers
    gen_opt = tf.keras.optimizers.Adam(learning_rate=gen_lr, beta_1=betas[0], beta_2=betas[1], epsilon=epsilon)
    disc_opt = tf.keras.optimizers.Adam(learning_rate=disc_lr, beta_1=betas[0], beta_2=betas[1], epsilon=epsilon)

    #Create dataloader
    dataset = VideoDataset(directory)
    dataloader = tf.data.Dataset.from_generator(
        lambda: iter(dataset),  # Corrected to use iter() to clearly return an iterator from the dataset
        output_signature=(
            tf.TensorSpec(shape=(16, 32, 32, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(8, 64, 64, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(4, 128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(2, 256, 256, 3), dtype=tf.float32)
        )
    ).batch(batch_size)

    print("Starting Training...")
    for epoch in range(n_epochs):
        start_time = time.time()
        total_gen_loss, total_disc_loss, total_loss_real = 0, 0, 0
        num_batches = 0

        for batch in dataloader:
            #Adjust learning rate


            iteration += 1
            x_real = batch
            num_batches += 1
            current_batch_size = x_real[0].shape[0]

            noise = tf.random.uniform((batch_size, z_dim), -1.0, 1.0)
            # Discriminator loss
            print("Generated noise")
            with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape() as disc_tape:
                x_fake = gen(noise, training=True)
                print("Generated x_fake")
                
                y_fake = disc(x_fake, training=True)  # Compute once for both G and D updates
                y_real = disc(x_real, training=True)
                print("Generated y_fake and y_real")
                
                disc_loss = disc_loss = tf.reduce_mean(tf.math.softplus(y_fake)) + tf.reduce_mean(tf.math.softplus(-y_real))
                gen_loss = tf.reduce_mean(tf.math.softplus(-y_fake))

                # Use the checking function instead of tf.debugging.check_numerics
                check_and_report(y_fake, "y_fake")
                check_and_report(y_real, "y_real")
                check_and_report(disc_loss, "disc_loss")
                check_and_report(gen_loss, "gen_loss")

                total_disc_loss += disc_loss

                loss_real = tf.reduce_mean(tf.math.softplus(-y_real))
                total_loss_real += loss_real


            #Apply discriminator loss gradients
            print("Calculating disc gradients")
            disc_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
            print("Applying disc gradients")
            try:
                disc_opt.apply_gradients(zip(disc_gradients, disc.trainable_variables))
            except Exception as e:
                print(f"Error applying gradients: {e}")
            print("Applied disc grad")
            

                

            total_gen_loss += gen_loss

            # Apply generator loss gradients
            print("Calculating gen gradients")
            gen_gradients = gen_tape.gradient(gen_loss, gen.trainable_variables)
            gen_opt.apply_gradients(zip(gen_gradients, gen.trainable_variables))


            xd, yd = x_real, y_real
            xd = list(xd) if isinstance(xd, (tuple, list)) else [xd]

            dydxs = []
            loss_gp = 0

            #Compute and apply gradient penalty
            with tf.GradientTape() as tape:
                tape.watch(xd)
                loss_gp = compute_gradient_penalty(xd, yd, lambda_gp)

            #Apply gradient penalty
            print("Calculating loss gp gradients")
            gradients = tape.gradient(loss_gp, disc.trainable_variables)
            disc_opt.apply_gradients(zip(gradients, disc.trainable_variables))
            
        avg_gen_loss = total_gen_loss / num_batches
        avg_disc_loss = total_disc_loss / num_batches
        avg_loss_real = total_loss_real / num_batches

        # After each epoch, check if we need to save the model
        if (epoch + 1) % save_interval == 0 or epoch + 1 == n_epochs:
            save_path = f"generator_epoch_{epoch+1}.pth"
            #torch.save(gen.state_dict(), save_path)
            #print(f"Saved generator to {save_path}")
        
        print(f"Epoch {epoch+1}/{n_epochs} complete. Average Losses: gen={avg_gen_loss}, disc={avg_disc_loss}, real={avg_loss_real}, loss_gp={loss_gp}")
        print(f"Time elapsed: {(time.time() - start_time)} seconds")

        return gen
    
#Go to project directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

#Set generator values
feature_map_channels = 1024
feature_map_size = 4
video_size = 256
n_frames = 16
z_dim = 256
gen_values = (z_dim, feature_map_channels, feature_map_size, video_size, video_size, n_frames)

#Set discriminator values
in_channels = 3
disc_values = in_channels

#Set training values
directory = '16 Frame 256p TF Tensors'
n_epochs = 2
batch_size = 16
n_critic = 1
dataset_train_faction = 0.2
betas = (0.0, 0.9)
epsilon = 1e-8
lambda_gp = 0.5
gen_lr = 0.0001
disc_lr = 0.0001
save_interval = 50
train_values = (directory, n_epochs, batch_size, n_critic, dataset_train_faction, betas, epsilon, lambda_gp)

#Train the model
gen = train(train_values, gen_values, disc_values, gen_lr, disc_lr, save_interval)

#Generate samples
gen.training = False
genSamples(gen, n=3)