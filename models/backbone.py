import torch
import torch.nn as nn
import os


class LSTMModel(nn.Module):
    def __init__(self, input_shape=(1024, 2), classes=24, weights=None):
        super(LSTMModel, self).__init__()

        # Check if weights file exists
        if weights is not None and not os.path.exists(weights):
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization), '
                             'or the path to the weights file to be loaded.')

        # Define LSTM units
        self.lstm1 = nn.LSTM(input_size=input_shape[1], hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

        # Define Dense layer for classification
        self.fc = nn.Linear(128, classes)

        # Load weights if provided
        if weights is not None:
            self.load_state_dict(torch.load(weights))

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)

        # Second LSTM layer
        x, _ = self.lstm2(x)

        # Take the last output for classification
        x = x[:, -1, :]  # Only the last time step

        # Pass the output through a fully connected layer
        x = self.fc(x)
        x = torch.softmax(x, dim=1)

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Reshape layer is implicit in PyTorch input

        # First Convolution Block
        self.conv1 = nn.Conv2d(2, 64, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(64)

        # Second Convolution Block
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(64)

        # Third Convolution Block
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(64)

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(64)

        # Fifth Convolution Block
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.bn5 = nn.BatchNorm2d(64)

        # Sixth Convolution Block
        self.conv6 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.bn6 = nn.BatchNorm2d(64)

        # Pooling layers
        self.maxpool = nn.MaxPool2d((1, 2))
        self.avgpool = nn.AvgPool2d((1, 16))

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.fc1 = nn.Linear(64, 128)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 24)

    def forward(self, x):
        # x needs to be reshaped before passed through the network
        x = x.view(-1, 2, 1024, 1).permute(0, 1, 3, 2)  # Reshape to match the input dimensions

        # First Conv Block
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))

        # Subsequent Conv Blocks and Pooling
        x = self.maxpool(F.relu(self.bn2(self.conv2(x))))
        x = self.maxpool(F.relu(self.bn3(self.conv3(x))))
        x = self.maxpool(F.relu(self.bn4(self.conv4(x))))
        x = self.maxpool(F.relu(self.bn5(self.conv5(x))))
        x = self.maxpool(F.relu(self.bn6(self.conv6(x))))

        # Average Pooling
        x = self.avgpool(x)

        # Flatten
        x = self.flatten(x)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x


# Model instance
model = ConvNet()

import torch
import torch.nn as nn
import torch.nn.functional as F


class PETCGDNN(nn.Module):
    def __init__(self, input_shape=(1024, 2), classes=24, weights=None):
        super(PETCGDNN, self).__init__()

        # Define network layers
        self.conv1_1 = nn.Conv2d(2, 75, kernel_size=(8, 2), padding=0)  # valid padding in PyTorch is default
        self.conv1_2 = nn.Conv2d(75, 25, kernel_size=(5, 1), padding=0)

        # GRU layer
        self.gru = nn.GRU(input_size=25, hidden_size=128, batch_first=True)

        # Output layer
        self.fc = nn.Linear(128, classes)

        # Load weights if provided
        if weights is not None:
            self.load_state_dict(torch.load(weights))

    def forward(self, x):
        # Spatial feature extraction
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))

        # Reshaping for GRU layer
        # After convolutions, assume x has shape (batch_size, 25, new_height, new_width)
        # We need to flatten the spatial dimensions into the sequence dimension for the GRU
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, height * width, channels)  # reshape to (batch_size, sequence, features)

        # Temporal feature extraction
        x, _ = self.gru(x)

        # Take only the last output of GRU
        x = x[:, -1, :]

        # Classification layer
        x = self.fc(x)
        x = F.softmax(x, dim=1)

        return x


# Model instance
model = PETCGDNN()


import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
import math


class MAC_backbone(nn.Module):
    """
    Multi-representation domain Attentive Contrastive learning backbone
    Based on the paper: "Multi-representation domain attentive contrastive learning 
    based unsupervised automatic modulation recognition"
    """
    def __init__(self, feat_dim_or_args=128, num_classes=11):
        super(MAC_backbone, self).__init__()
        
        # Handle different initialization patterns from training scripts
        if hasattr(feat_dim_or_args, 'feat_dim'):
            # Case: MAC_backbone(args, num_classes) for RML2018
            self.feat_dim = feat_dim_or_args.feat_dim
            self.num_classes = num_classes
        else:
            # Case: MAC_backbone(feat_dim, num_classes) or MAC_backbone(feat_dim, feat_dim)
            self.feat_dim = feat_dim_or_args
            self.num_classes = num_classes
        
        # Shared CNN encoder for all domains
        self.shared_encoder = self._build_shared_encoder()
        
        # Projection heads for different domains
        self.projector_sd = self._build_projector()  # Source Domain (I-Q)
        self.projector_td1 = self._build_projector()  # Target Domain 1 (AN)
        self.projector_td2 = self._build_projector()  # Target Domain 2 (AF) 
        self.projector_td3 = self._build_projector()  # Target Domain 3 (WT)
        self.projector_td4 = self._build_projector()  # Target Domain 4 (FFT)
        
        # Data augmentation projector for intra-domain learning
        self.projector_aug = self._build_projector()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_shared_encoder(self):
        """Build the shared CNN encoder"""
        return nn.Sequential(
            # First Conv Block
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Second Conv Block  
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Third Conv Block
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Fourth Conv Block
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            
            # Fully Connected Layer
            nn.Linear(512, self.feat_dim),
            nn.ReLU(inplace=True)
        )
    
    def _build_projector(self):
        """Build projection head for contrastive learning"""
        return nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim)
        )
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def transform_to_amplitude_phase(self, x):
        """Transform I-Q to amplitude-phase representation (AN domain)"""
        i_data = x[:, 0, :]  # I component
        q_data = x[:, 1, :]  # Q component
        
        # Calculate amplitude and phase
        amplitude = torch.sqrt(i_data**2 + q_data**2)
        phase = torch.atan2(q_data, i_data)
        
        # Stack amplitude and phase
        return torch.stack([amplitude, phase], dim=1)
    
    def transform_to_instantaneous_frequency(self, x):
        """Transform I-Q to instantaneous frequency representation (AF domain)"""
        i_data = x[:, 0, :]  # I component
        q_data = x[:, 1, :]  # Q component
        
        # Calculate instantaneous phase
        phase = torch.atan2(q_data, i_data)
        
        # Calculate instantaneous frequency (derivative of phase)
        # Use finite difference approximation
        freq = torch.zeros_like(phase)
        freq[:, 1:] = phase[:, 1:] - phase[:, :-1]
        freq[:, 0] = freq[:, 1]  # Handle boundary
        
        # Unwrap phase jumps
        freq = torch.where(freq > math.pi, freq - 2*math.pi, freq)
        freq = torch.where(freq < -math.pi, freq + 2*math.pi, freq)
        
        # Stack original I and frequency
        return torch.stack([i_data, freq], dim=1)
    
    def transform_to_wavelet(self, x):
        """Transform I-Q to wavelet representation (WT domain)"""
        batch_size, channels, length = x.shape
        
        # Apply continuous wavelet transform
        wavelet_real = torch.zeros_like(x)
        wavelet_imag = torch.zeros_like(x)
        
        for b in range(batch_size):
            for c in range(channels):
                signal = x[b, c, :].cpu().numpy()
                
                # Apply CWT using Morlet wavelet
                scales = np.arange(1, 32)
                coefficients, _ = pywt.cwt(signal, scales, 'cmor')
                
                # Take real and imaginary parts of the first scale
                if len(coefficients) > 0:
                    wavelet_real[b, c, :] = torch.from_numpy(coefficients[0].real).float()
                    if c == 0:  # For second channel, use imaginary part
                        wavelet_imag[b, c, :] = torch.from_numpy(coefficients[0].imag).float()
                    else:
                        wavelet_imag[b, c, :] = torch.from_numpy(coefficients[0].real).float()
        
        return torch.stack([wavelet_real[:, 0, :], wavelet_real[:, 1, :]], dim=1).to(x.device)
    
    def transform_to_fft(self, x):
        """Transform I-Q to FFT representation (FFT domain)"""
        # Apply FFT to I-Q data
        complex_signal = torch.complex(x[:, 0, :], x[:, 1, :])
        fft_result = torch.fft.fft(complex_signal, dim=-1)
        
        # Extract magnitude and phase
        magnitude = torch.abs(fft_result)
        phase = torch.angle(fft_result)
        
        return torch.stack([magnitude, phase], dim=1)
    
    def apply_data_augmentation(self, x):
        """Apply data augmentation for intra-domain contrastive learning"""
        # Random selection of augmentation type
        aug_type = torch.randint(1, 5, (1,)).item()
        
        if aug_type == 1:
            # Gaussian noise
            noise = torch.normal(0, 0.05, x.shape).to(x.device)
            return x + noise
        elif aug_type == 2:
            # Phase rotation
            angle = torch.rand(1).item() * 2 * math.pi
            i_data = x[:, 0, :]
            q_data = x[:, 1, :]
            i_rot = math.cos(angle) * i_data - math.sin(angle) * q_data
            q_rot = math.sin(angle) * i_data + math.cos(angle) * q_data
            return torch.stack([i_rot, q_rot], dim=1)
        elif aug_type == 3:
            # Horizontal flip
            return torch.stack([-x[:, 0, :], x[:, 1, :]], dim=1)
        else:
            # Vertical flip  
            return torch.stack([x[:, 0, :], -x[:, 1, :]], dim=1)
    
    def forward(self, x, mod_l='AN', view_chose='ALL', mode='pretrain'):
        """
        Forward pass of MAC backbone
        
        Args:
            x: Input I-Q signals [batch_size, 2, signal_length]
            mod_l: Target domain type ('AN', 'AF', 'WT', 'FFT')
            view_chose: View mode ('ALL' for multi-domain, 'DB' for dual-domain)
            mode: Training mode ('pretrain', 'Finetuning', 'linerProbing')
        
        Returns:
            Features for contrastive learning based on mode and view_chose
        """
        
        # Source Domain (I-Q) processing
        feat_sd = self.shared_encoder(x)
        feat_l = self.projector_sd(feat_sd)  # Source domain projection
        
        if view_chose == 'DB':
            # Dual-domain mode: only one target domain
            if mod_l == 'AN':
                x_td = self.transform_to_amplitude_phase(x)
            elif mod_l == 'AF':
                x_td = self.transform_to_instantaneous_frequency(x)
            elif mod_l == 'WT':
                x_td = self.transform_to_wavelet(x)
            elif mod_l == 'FFT':
                x_td = self.transform_to_fft(x)
            else:
                x_td = self.transform_to_amplitude_phase(x)  # Default to AN
            
            feat_td = self.shared_encoder(x_td)
            feat_ab = self.projector_td1(feat_td)
            
            # Intra-domain augmentation
            x_aug = self.apply_data_augmentation(x)
            feat_aug = self.shared_encoder(x_aug)
            feat_SD = self.projector_aug(feat_aug)
            
            return feat_l, feat_ab, feat_SD
            
        elif view_chose == 'ALL':
            # Multi-domain mode: all target domains
            # Transform to all target domains
            x_an = self.transform_to_amplitude_phase(x)
            x_af = self.transform_to_instantaneous_frequency(x) 
            x_wt = self.transform_to_wavelet(x)
            x_fft = self.transform_to_fft(x)
            
            # Extract features from each domain
            feat_an = self.shared_encoder(x_an)
            feat_af = self.shared_encoder(x_af)
            feat_wt = self.shared_encoder(x_wt)
            feat_fft = self.shared_encoder(x_fft)
            
            # Apply projections
            feat_TD = self.projector_td1(feat_an)    # AN domain
            feat_TD1 = self.projector_td2(feat_af)   # AF domain  
            feat_TD2 = self.projector_td3(feat_wt)   # WT domain
            feat_TD3 = self.projector_td4(feat_fft)  # FFT domain
            
            # Intra-domain augmentation
            x_aug = self.apply_data_augmentation(x)
            feat_aug = self.shared_encoder(x_aug)
            feat_SD1 = self.projector_aug(feat_aug)
            
            return feat_l, feat_TD, feat_TD1, feat_TD2, feat_TD3, feat_SD1
        
        else:
            raise ValueError(f"Unknown view_chose: {view_chose}")
