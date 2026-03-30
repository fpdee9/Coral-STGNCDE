import torch
import torchcde
import numpy as np
import time
import os
import copy
from coral_model import CoralSTGNCDE

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "data/processed/"
MODEL_SAVE_PATH = "coral_model_best.pth"

HIDDEN_DIM = 24      
EPOCHS = 500         
LR = 0.005           
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 106     
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_tensor(tensor):
    # Safeguard 1: Catch stray NaNs
    tensor = torch.nan_to_num(tensor, nan=0.0) 
    
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)
    
    # Safeguard 2: Prevent division by tiny numbers
    std[std < 1e-4] = 1.0  
    
    normed = (tensor - mean) / std
    return torch.nan_to_num(normed, nan=0.0)

def main():
    print(f"--- STARTING TRAINING ON {DEVICE} ---")
    
    if not os.path.exists(f"{DATA_DIR}X.pt"):
        print(f"Error: {DATA_DIR}X.pt not found.")
        return

    X_raw = torch.load(f"{DATA_DIR}X.pt").float()
    y = torch.load(f"{DATA_DIR}y.pt").float()
    mask = torch.load(f"{DATA_DIR}mask.pt").float()
    adj = torch.load(f"{DATA_DIR}adjacency_matrix.pt").float()
    
    # mathematically guarantee the graph cannot explode from overlapping nodes
    row_sums = adj.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0 
    adj = adj / row_sums
    
    num_sites, num_times, num_features = X_raw.shape
    print(f"   > Sites: {num_sites}, Time Steps: {num_times}")

    # Normalize
    X_normalized = normalize_tensor(X_raw)
    
    X_time_first = X_normalized.permute(1, 0, 2) 
    X_flat = X_time_first.reshape(num_times, -1).to(DEVICE)
    
    y = y.permute(1, 0, 2).to(DEVICE)
    mask = mask.permute(1, 0, 2).to(DEVICE)
    adj = adj.to(DEVICE)
    
    print("Interpolating Data...")
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_flat)
    
    model = CoralSTGNCDE(
        num_sites=num_sites,
        input_features=num_features,
        hidden_dim=HIDDEN_DIM,
        output_features=1,
        adj_matrix=adj
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # If the Test Loss doesn't improve for 10 checks (50 epochs), cut the LR in half.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_test_rmse = float('inf')
    best_epoch = 0
    
    print("\n--- BEGINNING TRAINING ---")
    start_time = time.time()
    SPLIT_IDX = int(num_times * 0.8)
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        pred = model(train_coeffs)
        
        train_pred = pred[:SPLIT_IDX]
        train_y    = y[:SPLIT_IDX]
        train_mask = mask[:SPLIT_IDX]
        loss = ((train_pred - train_y) ** 2 * train_mask).sum() / (train_mask.sum() + 1e-6)
        
        loss.backward()
        
        # INCREASED GRADIENT CLIPPING: Helps prevent zig-zags from exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
        
        optimizer.step()
        
        # Validation / Saving
        if (epoch + 1) % 5 == 0: 
            model.eval()
            with torch.no_grad():
                test_pred = pred[SPLIT_IDX:]
                test_y    = y[SPLIT_IDX:]
                test_mask = mask[SPLIT_IDX:]
                
                test_mse = ((test_pred - test_y) ** 2 * test_mask).sum() / (test_mask.sum() + 1e-6)
                test_rmse = torch.sqrt(test_mse)
                train_rmse = torch.sqrt(loss)
                
                print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train: {train_rmse:.4f} | Test: {test_rmse:.4f}", end="")
                
                # Step the scheduler based on Test RMSE
                scheduler.step(test_rmse)
                
                # SAVE IF BEST
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print(f"  <-- SAVED (New Best)")
                else:
                    print("")

    print(f"\n--- TRAINING COMPLETE ---")
    print(f"Best Test RMSE: {best_test_rmse:.4f} at Epoch {best_epoch}")
    print(f"Best Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()