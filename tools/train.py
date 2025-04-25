import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from dataset.loader import load_custom_image_folder
from model.vae import VAE
import os
import argparse

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = './checkpoints'

# --- Hyperparameters ---
num_epochs = 100
batch_size = 4
lr = 6e-4
beta = 0.00025
accum_steps = 2
effective_batch_size = batch_size * accum_steps

# --- Data ---
train_loader = load_custom_image_folder(
    source_dir='./custom',
    train_dir='./custom/train',
    test_dir='./custom/test',
    batch_size=batch_size,
    img_size=64
)

# --- Model, Optimizer, Loss ---
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# --- Training Loop ---
def train():
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Effective batch size: {effective_batch_size}")
    
    train_losses = []
    optimizer.zero_grad() # Initialize gradient

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)

            # --- Forward pass ---
            reconstructed, encoded = model(images)

            # Extract mean and log_variance from encoder
            mean, log_variance = torch.chunk(encoded, 2, dim=1)

            # --- Compute Loss ---
            # Reconstruction Loss 
            recon_loss = nn.MSELoss()(reconstructed, images) 
            
            # KL Divergence Loss
            # Sum over latent dimensions (C, H, W), then average over batch dimension (B)
            kl_div = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp()) 
            loss = recon_loss + beta * kl_div

            # Normalize loss for gradient accumulation
            loss = loss / accum_steps

            # --- Backward pass ---
            loss.backward()

            # --- Optimizer Step (with accumulation) --- 
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # --- Logging ---
            total_train_loss += loss.item() 
            total_recon_loss += recon_loss.item() 
            total_kl_loss += kl_div.item() 

            if (i + 1) % (100 // accum_steps) == 0: # Log every ~100 effective steps
                 print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                       f'Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Div: {kl_div.item():.4f}')

            with torch.no_grad():
                # Take the first image from the batch
                sample_image = images[0].unsqueeze(0)
                sample_reconstructed = model(sample_image)[0]
                sample_image = (sample_image * 0.5) + 0.5
                sample_reconstructed = (sample_reconstructed * 0.5) + 0.5
                torchvision.utils.save_image(sample_reconstructed, "sample_reconstructed.png")
    
        # --- Epoch End ---
        train_losses.append(total_train_loss / len(train_loader))

        print(f'--- Epoch {epoch+1} Finished ---')


        # --- Save Model Checkpoint ---
        if epoch % 10 == 0 or epoch == num_epochs - 1: # Save every 10 epochs and the last one
            checkpoint_path = os.path.join(checkpoint_dir, f"vae_model_epoch_{epoch+1:03d}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("--- Training finished ---") 

if __name__ == "__main__":
    train()