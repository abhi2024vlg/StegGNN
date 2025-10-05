import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.dataset import create_dataloader
from models.steganography_model import stegGNN
from utils.training import adjust_learning_rate, forward_pass, save_checkpoint, load_checkpoint

def train(model, dataloader, config):
    """Main training function"""
    criterion = nn.MSELoss().to(config.device)
    params = list(model.parameters())
    optimizer = optim.Adam(
        params, 
        lr=config.initial_lr, 
        betas=config.optimizer_betas, 
        eps=1e-6, 
        weight_decay=config.weight_decay
    )
    
    model = model.to(config.device).train()

    # Load checkpoint if exists
    checkpoint_path = ""
    best_model_path = ""
    
    epoch_start = 0
    best_loss = float('inf')
    
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        epoch_start, _ = load_checkpoint(model, optimizer, checkpoint_path)
        
    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path)
        best_loss = best_checkpoint["loss"]
        print(f"Previous best loss: {best_loss:.4f}")

    for epoch in range(epoch_start, config.num_epochs):
        adjust_learning_rate(optimizer, epoch, config)
        
        if epoch == 0:
            print(f"Training started ({config.num_epochs} epochs)")
            
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        running_loss = 0.0
        hiding_loss = 0.0
        revealing_loss = 0.0

        for batch_idx, (image, _) in enumerate(progress_bar):
            secret_img = image[image.shape[0]//2:].to(config.device)
            cover_img = image[:image.shape[0]//2].to(config.device)

            optimizer.zero_grad()
            
            loss, hiding_error, revealing_error = forward_pass(
                secret_img, cover_img, model, criterion, config
            )

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            hiding_loss += hiding_error.item()
            revealing_loss += revealing_error.item()
            
            progress_bar.set_postfix({'loss': running_loss / (batch_idx + 1)})
            
        # Calculate average losses
        avg_loss = running_loss / len(dataloader)
        avg_hiding_loss = hiding_loss / len(dataloader)
        avg_revealing_loss = revealing_loss / len(dataloader)
        
        print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {avg_loss:.4f}, '
              f'Hiding_loss: {avg_hiding_loss:.4f}, Revealing_loss: {avg_revealing_loss:.4f}')
        
        # Save checkpoints
        should_save = (
            (epoch == config.num_epochs - 1) or
            ((epoch + 1) % 100 == 0)
        )
        
        if should_save:
            if epoch == config.num_epochs - 1:
                checkpoint_name = 'checkpoints/final_checkpoint.pt'
            else:
                checkpoint_name = f'checkpoints/checkpoint_epoch_{epoch+1}.pt'
            
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_name)
            
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, best_loss, 'checkpoints/best_model.pt')
            print(f"New best model saved with loss: {best_loss:.4f}")

if __name__ == "__main__":
    config = Config()
    
    # Create model
    model = stegGNN(config.img_size, config.in_channels, config.embedding_dim)
    
    # Create dataloader
    train_loader = create_dataloader(config)
    
    # Start training
    train(model, train_loader, config)
