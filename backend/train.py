import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from preprocess import prepare_datasets

from model import TransformerForecaster

from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MODEL_PATH
)

# ==========================================
# DEVICE
# ==========================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", DEVICE)


# ==========================================
# DATASET CLASS
# ==========================================

class TimeSeriesDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


# ==========================================
# CREATE DATALOADERS
# ==========================================

def create_dataloaders():

    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        scaler,
        df
    ) = prepare_datasets()

    train_dataset = (
        TimeSeriesDataset(
            X_train,
            y_train
        )
    )

    valid_dataset = (
        TimeSeriesDataset(
            X_valid,
            y_valid
        )
    )

    test_dataset = (
        TimeSeriesDataset(
            X_test,
            y_test
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
        train_loader,
        valid_loader,
        test_loader
    )


# ==========================================
# TRAIN FUNCTION
# ==========================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(DEVICE)

        y_batch = y_batch.to(DEVICE)

        # ==================================
        # ZERO GRADIENTS
        # ==================================

        optimizer.zero_grad()

        # ==================================
        # FORWARD PASS
        # ==================================

        predictions = model(X_batch)

        # ==================================
        # LOSS
        # ==================================

        loss = criterion(
            predictions,
            y_batch
        )

        # ==================================
        # BACKPROPAGATION
        # ==================================

        loss.backward()

        # ==================================
        # GRADIENT CLIPPING
        # ==================================

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # ==================================
        # OPTIMIZER STEP
        # ==================================

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    return avg_loss


# ==========================================
# VALIDATION FUNCTION
# ==========================================

def validate_one_epoch(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0

    all_predictions = []

    all_targets = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)

            y_batch = y_batch.to(DEVICE)

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            total_loss += loss.item()

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                y_batch.cpu().numpy()
            )

    avg_loss = total_loss / len(loader)

    return (
        avg_loss,
        np.array(all_predictions),
        np.array(all_targets)
    )


# ==========================================
# METRICS
# ==========================================

def calculate_metrics(
    predictions,
    targets
):

    mae = mean_absolute_error(
        targets,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            targets,
            predictions
        )
    )

    mape = mean_absolute_percentage_error(
        targets,
        predictions
    )

    # ======================================
    # DIRECTIONAL ACCURACY
    # ======================================

    direction_pred = np.sign(
        predictions
    )

    direction_true = np.sign(
        targets
    )

    directional_accuracy = np.mean(
        direction_pred == direction_true
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Directional_Accuracy":
            directional_accuracy
    }


# ==========================================
# TRAINING LOOP
# ==========================================

def train_model():

    # ======================================
    # DATALOADERS
    # ======================================

    (
        train_loader,
        valid_loader,
        test_loader
    ) = create_dataloaders()

    # ======================================
    # MODEL
    # ======================================

    model = (
        TransformerForecaster()
        .to(DEVICE)
    )

    # ======================================
    # LOSS FUNCTION
    # ======================================

    criterion = nn.HuberLoss()

    # ======================================
    # OPTIMIZER
    # ======================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ======================================
    # LR SCHEDULER
    # ======================================

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )
    )

    # ======================================
    # EARLY STOPPING
    # ======================================

    best_valid_loss = float("inf")

    patience_counter = 0

    EARLY_STOPPING_PATIENCE = 10

    # ======================================
    # HISTORY
    # ======================================

    train_losses = []

    valid_losses = []

    # ======================================
    # EPOCH LOOP
    # ======================================

    for epoch in range(EPOCHS):

        # ==================================
        # TRAIN
        # ==================================

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        # ==================================
        # VALIDATE
        # ==================================

        valid_loss, predictions, targets = (
            validate_one_epoch(
                model,
                valid_loader,
                criterion
            )
        )

        # ==================================
        # SCHEDULER STEP
        # ==================================

        scheduler.step(valid_loss)

        # ==================================
        # SAVE HISTORY
        # ==================================

        train_losses.append(train_loss)

        valid_losses.append(valid_loss)

        # ==================================
        # PRINT PROGRESS
        # ==================================

        print(
            f"Epoch {epoch+1}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.6f}"
        )

        print(
            f"Valid Loss: {valid_loss:.6f}"
        )

        # ==================================
        # EARLY STOPPING
        # ==================================

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            patience_counter = 0

            # ==============================
            # SAVE MODEL
            # ==============================

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "Best model saved."
            )

        else:

            patience_counter += 1

        # ==================================
        # STOP TRAINING
        # ==================================

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "Early stopping triggered."
            )

            break

        print("-" * 50)

    # ======================================
    # LOAD BEST MODEL
    # ======================================

    model.load_state_dict(
        torch.load(MODEL_PATH)
    )

    # ======================================
    # TEST EVALUATION
    # ======================================

    test_loss, test_predictions, test_targets = (
        validate_one_epoch(
            model,
            test_loader,
            criterion
        )
    )

    # ======================================
    # METRICS
    # ======================================

    metrics = calculate_metrics(
        test_predictions,
        test_targets
    )

    print("\n=================================")
    print("FINAL TEST METRICS")
    print("=================================")

    for key, value in metrics.items():

        print(
            f"{key}: {value:.6f}"
        )

    # ======================================
    # PLOT LOSSES
    # ======================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        valid_losses,
        label="Validation Loss"
    )

    plt.title(
        "Training vs Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.show()

    # ======================================
    # PLOT PREDICTIONS
    # ======================================

    plt.figure(figsize=(12, 6))

    plt.plot(
        test_targets[:200],
        label="Actual Returns"
    )

    plt.plot(
        test_predictions[:200],
        label="Predicted Returns"
    )

    plt.title(
        "Actual vs Predicted Returns"
    )

    plt.xlabel("Time")

    plt.ylabel("Return")

    plt.legend()

    plt.grid(True)

    plt.show()

    return model


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("\nTRAINING STARTED\n")

    trained_model = train_model()

    print("\nTRAINING COMPLETED")