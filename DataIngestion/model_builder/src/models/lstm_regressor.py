import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim.lr_scheduler as lr_scheduler

# --- PyTorch Lightning System Definition ---
class LSTMRegressor(pl.LightningModule):
    """PyTorch Lightning system for training the LSTM surrogate model."""
    def __init__(self, backbone: nn.Module, learning_rate: float = 1e-3, **kwargs):
        """Initializes the LightningModule.

        Args:
            backbone (nn.Module): The core neural network (e.g., LSTM followed by Linear layer).
            learning_rate (float): The learning rate for the optimizer.
        """
        super().__init__()
        # Use save_hyperparameters, but ignore the backbone to keep checkpoints lighter
        # The backbone architecture itself is implicitly saved via the state_dict.
        # We only need to save hyperparameters needed for instantiation (like learning_rate).
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.criterion = nn.MSELoss() # Mean Squared Error for regression

    def forward(self, x):
        """Forward pass delegates to the backbone network."""
        return self.backbone(x)

    def _common_step(self, batch, batch_idx, stage):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Calculate metrics
        rmse = torch.sqrt(loss) # RMSE is sqrt(MSE)
        mae = nn.functional.l1_loss(y_hat, y) # MAE

        # Log metrics
        # Use sync_dist=True if running multi-GPU/multi-node training
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{stage}_rmse', rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{stage}_mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    # If you add a test loop, define test_step here
    def test_step(self, batch, batch_idx):
        # Log metrics with 'test_' prefix
        return self._common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Add ReduceLROnPlateau scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',      # Reduce LR when monitored metric stops decreasing
            factor=0.1,      # Reduce LR by factor of 10
            patience=3,      # Wait 3 epochs with no improvement before reducing
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", # Metric to monitor for scheduler patience
                "interval": "epoch",   # Check after each epoch
                "frequency": 1         # Check every 1 epoch
            },
        } 