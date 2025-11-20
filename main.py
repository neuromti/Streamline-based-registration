import os
import nibabel as nib
import torch    
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time

TEMPLETE_SUBJECT = '959574'
#TRACTS_10_DIR = './HCP10_Tracts'
MRI_10_DIR = './HCP10_MRI'
#OUTPUT_DIR = './output/mrregister'
LAMBDA_REG = 0.01  # Weight for the smoothness regularization loss
BATCHSIZE = 1

class MRIDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
        """
        import pandas as pd
        self.data_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        struct_mri_path = self.data_frame.iloc[idx]['struct_mri_path']
        diff_mri_path = self.data_frame.iloc[idx]['diff_mri_path']
        fod_path = self.data_frame.iloc[idx]['fod_path']
        tck_path = self.data_frame.iloc[idx]['tck_path']

        struct_mri = nib.load(struct_mri_path).get_fdata()
        diff_mri = nib.load(diff_mri_path).get_fdata()
        fod = nib.load(fod_path).get_fdata()
        
        sample = {
            'struct_mri': torch.from_numpy(struct_mri).float(),
            'diff_mri': torch.from_numpy(diff_mri).float(),
            'fod': torch.from_numpy(fod).float(),
            'tck_path': tck_path
        }

        return sample
    
    
class ConvBlock3D(nn.Module):
    """Standard 3D Convolutional Block: Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=True):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not batch_norm)
        self.bn = nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Unet(nn.Module):
    """
    3D U-Net for Volumetric Registration (Predicts a 3D Deformation Field).
    Input: Concatenated Template (Fixed) and Subject (Moving) MRI (2 channels).
    Output: Deformation Field (3 channels: displacement in Z, Y, X).
    """
    def __init__(self, in_channels=2, out_channels=3, initial_filters=16, levels=5):
        super(Unet, self).__init__()
        
        self.levels = levels
        
        # Dynamic channel list for consistent architecture scaling: 16, 32, 64, 128, 256
        f = initial_filters
        self.channels = [f * (2**i) for i in range(levels)] 

        # --- ENCODER PATH (Downsampling) ---
        self.encoder = nn.ModuleList()
        # Initial block (input 2 channels -> initial_filters)
        self.encoder.append(nn.Sequential(
            ConvBlock3D(in_channels, self.channels[0]),
            ConvBlock3D(self.channels[0], self.channels[0]) # Start with two blocks
        ))
        
        # Downsampling blocks
        for i in range(levels - 1):
            block = nn.Sequential(
                # Convolution with stride=2 acts as downsampling
                ConvBlock3D(self.channels[i], self.channels[i+1], stride=2, padding=1),
                ConvBlock3D(self.channels[i+1], self.channels[i+1])
            )
            self.encoder.append(block)

        # --- BOTTLENECK ---
        self.bottleneck = nn.Sequential(
            ConvBlock3D(self.channels[-1], self.channels[-1]),
            ConvBlock3D(self.channels[-1], self.channels[-1])
        )

        # --- DECODER PATH (Upsampling) ---
        self.decoder = nn.ModuleList()
        
        for i in reversed(range(levels - 1)):
            # Transposed Convolution for Upsampling
            upconv = nn.ConvTranspose3d(
                self.channels[i+1], 
                self.channels[i],   
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )
            
            # Conv block after concatenation (channels are doubled due to skip)
            convs = nn.Sequential(
                ConvBlock3D(self.channels[i] * 2, self.channels[i]), 
                ConvBlock3D(self.channels[i], self.channels[i])
            )
            
            self.decoder.append(nn.ModuleList([upconv, convs]))

        # --- FINAL OUTPUT LAYER ---
        # Maps feature maps to the 3-channel deformation field
        self.output_conv = nn.Conv3d(self.channels[0], out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x is the concatenated input (B, 2, D, H, W)
        
        skip_connections = []
        
        # Encoder
        for i, block in enumerate(self.encoder):
            x = block(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = skip_connections.pop()
        x = self.bottleneck(x)
        
        # Decoder (Use skip connections in reverse order)
        skip_connections = skip_connections[::-1] 
        
        for i, (upconv, convs) in enumerate(self.decoder):
            # Upsample
            x = upconv(x)
            
            # Retrieve skip connection
            skip = skip_connections[i]
            
            if x.shape[2:] != skip.shape[2:]:
                D_skip, H_skip, W_skip = skip.shape[2:]
                # Crop (D, H, W) dimensions
                x = x[:, :, :D_skip, :H_skip, :W_skip] 
            
            print(x.shape, skip.shape)
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Apply convolutions
            x = convs(x)

        # Output: Deformation Field (B, 3, D, H, W)
        flow = self.output_conv(x)
        return flow

class SpatialTransformer(nn.Module):
    """Applies the predicted deformation field (flow) to warp the moving image."""
    
    def __init__(self, size):
        super(SpatialTransformer, self).__init__()
        
        # Create a fixed reference grid
        vectors = [ torch.arange(0, s) for s in size ]
        grid = torch.meshgrid(vectors)
        grid = torch.stack(grid) # Shape: [3, D, H, W]
        grid = torch.unsqueeze(grid, 0).float()  # Add batch dimension
        self.register_buffer('grid', grid)  # Register as buffer so it's moved to GPU with the model
        
    def forward(self, moving_image, flow):
        # Add the predicted flow (dV) to the reference grid (I) to obtain the sample grid (I + dV)
        new_locs = self.grid.clone() + flow
        
        # Normalize grid values to [-1, 1] for grid_sample
        size = new_locs.shape[2:]  # D, H, W
        
        for i in range(len(size)):
            d = size[i]
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (d - 1) - 0.5)
            
        # Permute to shape [B, D, H, W, 3] for grid_sample
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        
        # Sample the moving image at new locations
        warped_image = nn.functional.grid_sample(
            moving_image, 
            new_locs,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        return warped_image
    
# Registration loss (Similarity + Smoothness)

class NCCLoss(nn.Module):
    """Normalized Cross-Correlation Loss for image similarity."""
    
    def __init__(self, win=9, eps=1e-5):
        super(NCCLoss, self).__init__()
        self.eps = eps
        self.win = (win, win, win)
        self.w = nn.Parameter(torch.ones(1, 1, *self.win), requires_grad=False)
        self.padding = win // 2
        
    def forward(self, I, J):
        # I: Fixed image, J: Warped moving image
        
        # Local means
        mu_I = nn.functional.conv3d(I, self.w, padding=self.padding) 
        mu_J = nn.functional.conv3d(J, self.w, padding=self.padding)
        
        # Local cross-correlation and variances terms
        mu_I_J = mu_I * mu_J
        I_J = nn.functional.conv3d(I * J, self.w, padding=self.padding)
        
        I_sq = nn.functional.conv3d(I.pow(2), self.w, padding=self.padding)
        J_sq = nn.functional.conv3d(J.pow(2), self.w, padding=self.padding)
        mu_I_sq = mu_I.pow(2)
        mu_J_sq = mu_J.pow(2)
        
        # Calculate local covariance (numerator) and local std dev (denominator)
        cross = I_J - mu_I_J
        I_var = I_sq - mu_I_sq
        J_var = J_sq - mu_J_sq
        
        # Stabilize variance and calculate NCC
        ncc = cross / (torch.sqrt(I_var * J_var) + self.eps)
        
        # Loss is the negative mean NCC
        return -torch.mean(ncc)  
        
class BendingEnergyLoss(nn.Module):
    """Bending Energy Loss for smoothness of deformation field."""
    
    def __init__(self):
        super(BendingEnergyLoss, self).__init__()
        
    def forward(self, flow):
       # L2 Gradient Loss (Approximation of Diffusion/Smoothness Regularization)
        
        # Gradients in the 3 spatial directions (Z, Y, X)
        grad_z = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        grad_y = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        grad_x = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
        
        # L2 norm of the gradients
        smoothness_loss = torch.mean(grad_z**2) + torch.mean(grad_y**2) + torch.mean(grad_x**2)
        return smoothness_loss
    
class RegistrationLoss(nn.Module):
    """Combined Loss: Similarity (NCC) + Regularization (Smoothness)."""
    def __init__(self, lambda_reg=1.0):
        super(RegistrationLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.similarity_loss = NCCLoss()
        self.regularization_loss = BendingEnergyLoss()

    def forward(self, fixed_image, warped_image, flow):
        L_sim = self.similarity_loss(fixed_image, warped_image)
        L_reg = self.regularization_loss(flow)
        
        # Total Loss
        total_loss = L_sim + self.lambda_reg * L_reg
        
        return total_loss, L_sim, L_reg
    
def train_registration_model(model, stn, criterion, optimizer, train_loader, fixed_image_tensor, epochs=1):
    """Runs the training loop for the MRI registration model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model.to(device)
    stn.to(device)
    criterion.to(device)
    
    # Move fixed image to device and prepare for batching (unsqueeze batch dim, then repeat)
    fixed_image_tensor = fixed_image_tensor.unsqueeze(0).unsqueeze(0).to(device) #(1, 1, D, H, W)
    
    model.train()
    
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        
        for i, sample in enumerate(train_loader):
            # Load the moving image (subject MRI)
            # Unsqueeze channel dimension (1)
            moving_image = sample['struct_mri'].unsqueeze(1).to(device)  #(B, 1, D, H, W)
            
            # Repeat the template image across the batch dimension
            fixed_image_batch = fixed_image_tensor.repeat(moving_image.size(0), 1, 1, 1, 1)  #(B, 1, D, H, W)
            
            # Concatenated Fixed (Template) and Moving (Subject) inputs
            model_input = torch.cat((fixed_image_batch, moving_image), dim=1)  #(B, 2, D, H, W)
            
            optimizer.zero_grad()
            
            # Predict deformation field
            flow = model(model_input) 
            
            # Warp the moving image
            warped_moving_image = stn(moving_image, flow)
            
            # Calculate loss
            total_loss, L_sim, L_reg = criterion(fixed_image_batch, warped_moving_image, flow)
            
            # Backprogation
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Total loss: {total_loss.item():.4f}, L_sim: {L_sim.item():.4f}, L_reg: {L_reg.item():.4f}")
            
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1} finished. Avg Loss: {running_loss / len(train_loader):.4f}. Time: {epoch_time:.2f}s\n")
             
if __name__ == "__main__":
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU (will be very slow for 3D volumes).")
    
    # Load template structural mri 
    template_struct_mri_path = os.path.join(MRI_10_DIR, 'data', f"{TEMPLETE_SUBJECT}_StructuralRecommended", f"{TEMPLETE_SUBJECT}", 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz')
    try:
        template_struct_mri = nib.load(template_struct_mri_path).get_fdata()
        template_struct_mri_tensor = torch.from_numpy(template_struct_mri).float()
    except FileNotFoundError:
        print(f"Error: Template structural MRI file not found at {template_struct_mri_path}")
        exit(1)
        
    print(f"Template_struct_mri_tensor.shape: {template_struct_mri_tensor.shape}")
    
    # Data loading
    dataset = MRIDataset(csv_file='data_manifest.csv')
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Error: Dataset is empty. Please check the data_manifest.csv file.")
        exit(1)
        
    # Inspect a sample
    sample = dataset[0]
    print(f"Structural MRI shape: {sample['struct_mri'].shape}")
    print(f"Diffusion MRI shape: {sample['diff_mri'].shape}")
    print(f"FOD shape: {sample['fod'].shape}")
    print(f"TCK path: {sample['tck_path']}")

    # Splite data into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")   

    # Create data loaders
    BATCHSIZE = 1
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False) 
    
    model = Unet(in_channels=2, out_channels=3, initial_filters=16, levels=5)
    stn = SpatialTransformer(size=template_struct_mri.shape)
    criterion = RegistrationLoss(lambda_reg=LAMBDA_REG)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    train_registration_model(
            model=model,
            stn=stn,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            fixed_image_tensor=template_struct_mri_tensor,
            epochs=5 
        )