from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb


def test_logging(logger: tb.SummaryWriter):
    """
    Logs the dummy loss and accuracy for a simplified training and validation loop.

    For training, logs the training loss every iteration and the average accuracy every epoch.
    For validation, logs the average accuracy every epoch.

    Args:
        logger (tb.SummaryWriter): TensorBoard summary writer to log the metrics.
    """
    global_step = 0
    for epoch in range(10):
        # Store metrics to calculate averages
        train_acc_list = []
        val_acc_list = []

        # Training loop
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9 ** (epoch + iteration / 20.0)
            dummy_train_accuracy = epoch / 10.0 + torch.randn(1).item()
            train_acc_list.append(dummy_train_accuracy)

            # Log training loss
            logger.add_scalar('train_loss', dummy_train_loss, global_step)

            global_step += 1

        # Log average training accuracy for the epoch
        avg_train_accuracy = sum(train_acc_list) / len(train_acc_list)
        logger.add_scalar('train_accuracy', avg_train_accuracy, global_step)

        # Validation loop
        torch.manual_seed(epoch)
        for _ in range(10):
            dummy_validation_accuracy = epoch / 10.0 + torch.randn(1).item()
            val_acc_list.append(dummy_validation_accuracy)

        # Log average validation accuracy for the epoch
        avg_val_accuracy = sum(val_acc_list) / len(val_acc_list)
        logger.add_scalar('val_accuracy', avg_val_accuracy, global_step)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    args = parser.parse_args()

    log_dir = Path(args.exp_dir) / f"logger_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    test_logging(logger)
