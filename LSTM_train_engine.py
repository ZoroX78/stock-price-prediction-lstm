import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_data_loader import MultiStockDataset, DataLoader

# =========================================================
# 1. THE DEEP LEARNING MODEL DEFINITION
# =========================================================
class StockLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_prob=0.2):
        super(StockLSTMClassifier, self).__init__()
        
        # The Core LSTM Layer
        # batch_first=True tells PyTorch our data is structured as (Batch, Sequence, Features)
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # Dropout layer to mitigate overfitting
        self.dropout = nn.Dropout(dropout_prob)
        
        # Fully connected layer that converts the hidden states into a single prediction output
        self.fc = nn.Linear(hidden_dim, 1)
        
        # Sigmoid squashes the output value cleanly between 0.0 and 1.0
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Pass inputs through the LSTM layers
        # lstm_out shape: (Batch_Size, Sequence_Length, Hidden_Dim)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Extract ONLY the last time step output (Day 60) to form the prediction
        last_time_step = lstm_out[:, -1, :]
        
        # Pass through the linear layer and apply sigmoid activation
        out = self.fc(self.dropout(last_time_step))
        return self.sigmoid(out).squeeze(-1)

# =========================================================
# 2. THE TRAINING ENGINE
# =========================================================
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    # Binary Cross Entropy Loss is the standard for 0 or 1 classification jobs
    criterion = nn.BCELoss()
    
    # AdamW is a highly reliable optimizer with weight decay to prevent extreme parameters
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🖥️ Training pipeline executing on target hardware device: {device}\n")
    
    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Reset gradients from previous batch steps
            optimizer.zero_grad()
            
            # Forward Pass: Predict probabilities
            predictions = model(batch_x)
            
            # Compute Loss
            loss = criterion(predictions, batch_y)
            
            # Backward Pass: Calculate gradients
            loss.backward()
            
            # Update Model Weights
            optimizer.step()
            
            running_train_loss += loss.item() * batch_x.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        # --- VALIDATION PHASE (Evaluation mode) ---
        model.eval()
        running_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad(): # Disable gradient updates to save RAM/Compute
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                running_val_loss += loss.item() * batch_x.size(0)
                
                # Turn probability threshold into discrete metrics: 
                # If prediction >= 0.5, predict Up (1), else Down (0)
                binary_preds = (predictions >= 0.5).float()
                correct_predictions += (binary_preds == batch_y).sum().item()
                total_samples += batch_y.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = (correct_predictions / total_samples) * 100
        
        print(f"Epoch [{epoch}/{epochs}] ➡️ Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        
    # Save the finalized model parameters to disk
    torch.save(model.state_dict(), "lstm_stock_model.pt")
    print("\n💾 Model parameters successfully compiled and stored to 'lstm_stock_model.pt'")

if __name__ == "__main__":
    # Hyperparameters
    PROCESSED_DATA_DIR = "data/processed"
    SEQUENCE_LENGTH = 60
    BATCH_SIZE = 64
    INPUT_DIM = 14     # Our 14 unique technical indicator input features
    HIDDEN_DIM = 64    # Number of internal features extracted by LSTM nodes
    NUM_LAYERS = 2     # Stacking two LSTMs on top of each other
    
    # Load Data Loaders
    print("📊 Preparing data structures...")
    train_dataset = MultiStockDataset(PROCESSED_DATA_DIR, sequence_length=SEQUENCE_LENGTH, split_type='train')
    val_dataset   = MultiStockDataset(PROCESSED_DATA_DIR, sequence_length=SEQUENCE_LENGTH, split_type='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model Instance
    model = StockLSTMClassifier(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    
    # Execute Training Loop
    train_model(model, train_loader, val_loader, epochs=50, lr=0.001)