"""Logging utilities for training and TensorBoard."""

import logging
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary


def save_checkpoint(model, optimizer, scheduler, epoch, save_path, best=False, hparams=None):
    """Save model state."""
    os.makedirs(save_path, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_name": type(optimizer).__name__,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "initial_hparams": hparams,
    }
    torch.save(
        checkpoint, os.path.join(save_path, "best_model.pth" if best else f"checkpoint_{epoch}.pth")
    )


def setup_logging(args, model, result_dir):
    """Creates folder, setup logger and tensorboard writer.

    Args:
        args (Namespace):
        model (torch module): _description_

    Returns:
        dict: returns tensorboard writer and file handler
    """
    if not os.path.exists(result_dir + model.name):
        os.makedirs(result_dir + model.name)

    listdir = [
        d
        for d in os.listdir(result_dir + model.name)
        if os.path.isdir(result_dir + model.name + "/" + d)
    ]
    run_name = "/run" + str(len(listdir))
    args.log_path = result_dir + model.name + run_name + "/"

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    logfilename = args.log_path + "train.log"
    handler = logging.FileHandler(logfilename)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)-15s %(levelname)-8s %(name)s:%(lineno)d \n%(message)s\n"
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # Log hyperparameters
    config = vars(args)
    _lst = [f"{key}:\t{val}\n" for key, val in config.items()]
    logger.info("".join(_lst))

    logger.info(summary(model, verbose=0, col_width=16))

    # Tensorboard
    writer = SummaryWriter(log_dir=args.log_path + "/tb")

    return {"tb": writer, "file": logger}


def log_results(logger, loss, epoch):
    """Save loss at every checkpoint.

    Args:
        logger (dict): tensorboard writer and file handler
        loss (dict): training loss and validation loss for a epoch
        epoch (int):
    """
    exc = (
        f"Epoch: {epoch} \t"
        f"Training Loss: {loss['train']:.6f} \t"
        f"Validation Loss: {loss['validation']:.6f}"
    )
    logger["file"].info(exc)
    logger["tb"].add_scalars("Loss", loss, epoch)
