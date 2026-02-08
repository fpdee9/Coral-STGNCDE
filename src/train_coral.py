import torch
import torchcde
import time
from coral_model import CoralSTGNCDE

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "data/processed/"
HIDDEN_DIM = 8   # Small dimension to prevent overfitting on sparse data
EPOCHS = 300
LR = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- STARTING TRAINING ON {DEVICE} ---")
    
    # 1. Load Data
    print("Loading Tensors...")
    X = torch.load(f"{DATA_DIR}X.pt").float()   # (Sites, Time, 3)
    y = torch.load(f"{DATA_DIR}y.pt").float()   # (Sites, Time, 1)
    mask = torch.load(f"{DATA_DIR}mask.pt").float()
    adj = torch.load(f"{DATA_DIR}adjacency_matrix.pt").float()
    
    # 2. Reshape for Model (Time, Sites, Feats)
    # Treat the entire 106-site system as ONE sample with complex structure
    # X: (106, 14610, 3) -> (14610, 106, 3)
    X = X.permute(1, 0, 2).to(DEVICE) # (Time, Sites, Feats)
    y = y.permute(1, 0, 2).to(DEVICE)
    mask = mask.permute(1, 0, 2).to(DEVICE)
    adj = adj.to(DEVICE)
    
    # 3. Create Continuous Path (Splines)
    # This converts discrete daily data into a continuous mathematical function
    print("Interpolating Continuous Path (Cubic Hermite Spline)...")
    # We flatten sites into channels for the spline construction
    # Input to spline: (Batch, Time, Channels) -> (1, 14610, 106*3)
    T, S, F = X.shape
    X_flat = X.reshape(T, S * F)
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_flat)
    print("   > Path constructed.")

    # 4. Initialize Model
    model = CoralSTGNCDE(
        num_sites=S,
        input_features=F, # 3 (SST, DHW, CO2)
        hidden_dim=HIDDEN_DIM,
        output_features=1,
        adj_matrix=adj
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # SPLIT (Approx 80/20 split by time)
    # 14610 days total. 
    # Train: 0 - 11688 (1985-2016)
    # Test: 11688 - End (2017-2024)
    SPLIT_IDX = 11688 
    
    print("\n--- BEGINNING EPOCHS ---")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Forward Pass
        # Returns: (Time, Sites, 1)
        pred = model(coeffs)
        
        # --- CALC TRAINING LOSS ---
        # Slice: [:SPLIT_IDX]
        train_pred = pred[:SPLIT_IDX, :, :]
        train_y    = y[:SPLIT_IDX, :, :]
        train_mask = mask[:SPLIT_IDX, :, :]
        
        # MSE Loss
        loss = (train_pred - train_y) ** 2
        # Apply Mask (Only count days with data)
        loss = (loss * train_mask).sum() / (train_mask.sum() + 1e-6)
        
        loss.backward()
        optimizer.step()
        
        # --- REPORTING ---
        if epoch % 10 == 0:
            with torch.no_grad():
                # Test Loss
                test_pred = pred[SPLIT_IDX:, :, :]
                test_y    = y[SPLIT_IDX:, :, :]
                test_mask = mask[SPLIT_IDX:, :, :]
                
                test_loss = (test_pred - test_y) ** 2
                test_loss = (test_loss * test_mask).sum() / (test_mask.sum() + 1e-6)
                
                # RMSE (Root Mean Squared Error) - more interpretable
                train_rmse = torch.sqrt(loss)
                test_rmse = torch.sqrt(test_loss)
                
                print(f"Epoch {epoch:03d} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

    duration = time.time() - start_time
    print(f"\nTraining Finished in {duration/60:.1f} minutes.")
    
    # Save
    torch.save(model.state_dict(), "coral_model_v1.pth")
    print("Model saved to coral_model_v1.pth")

if __name__ == "__main__":
    main()