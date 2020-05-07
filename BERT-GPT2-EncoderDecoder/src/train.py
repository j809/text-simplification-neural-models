import abc
import os
import sys
import tqdm
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Any
from pathlib import Path
from src.utils import BatchResult, EpochResult, FitResult

return_acc = False


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.
    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, scheduler, device='cpu'):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.batch_idx = 0
        self.epoch = 0
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None, writer=None, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        #  train_loss, train_acc, test_loss, test_acc = [], [], [], []
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        #  best_acc = None
        epochs_without_improvement = 0

        while actual_num_epochs < num_epochs:
            epoch = actual_num_epochs
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch + 1}/{num_epochs} ---', verbose)

            train_result = self.train_epoch(dl_train, **kw)
            train_loss.append(torch.tensor(train_result.losses).mean().item())

            if writer is not None:
                writer.add_scalar('Loss/train', train_loss[-1], epoch)

            test_result = self.test_epoch(dl_test, **kw)
            test_loss.append(torch.tensor(test_result.losses).mean().item())

            if writer is not None:
                writer.add_scalar('Loss/test', test_loss[-1], epoch)

            actual_num_epochs += 1

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs,
                         train_loss, test_loss)

    def test(self, dl_test: DataLoader, checkpoints=None, post_epoch_fn=None, writer=None, **kw) -> FitResult:

        test_loss = []

        verbose = False  # pass this to train/test_epoch.

        test_result = self.test_epoch(dl_test, **kw)
        test_loss.append(torch.tensor(test_result.losses).mean().item())
            
        if writer is not None:
            writer.add_scalar('Loss/test', test_loss[-1], 0)

        if post_epoch_fn:
            post_epoch_fn(0, EpochResult([0], 0), test_result, verbose)

        return FitResult(0,
                         0, test_loss)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train()  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, mode='train', **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.eval()  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, mode='test', **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None, mode='train') -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        global return_acc
        losses = []
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                if batch_idx == num_batches - 1:
                    return_acc = True

                batch_res = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)

            avg_loss = sum(losses) / num_batches
            num = num_samples if mode == 'test' else num_samples / num_batches

            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f})')
        return_acc = False

        return EpochResult(losses=losses)


class BERT_GPT2_Trainer(Trainer):
    def __init__(self, model, tokenizer, loss_fn, optimizer, scheduler, max_len=128, device=None):
        super().__init__(model, loss_fn, optimizer, scheduler, device)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def train_epoch(self, dl_train: DataLoader, **kw):
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        x, encoder_attention_mask, y, decoder_attention_mask = batch
        x = x.to(self.device)
        y = y.to(self.device)
        encoder_attention_mask = encoder_attention_mask.to(self.device)
        decoder_attention_mask = decoder_attention_mask.to(self.device)

        model_kwargs = {
            "encoder_attention_mask": encoder_attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_lm_labels": y
        }

        self.optimizer.zero_grad()

        outputs = self.model(x, y, **model_kwargs)

        loss = outputs[0]
        loss = loss.clone().mean()
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        if return_acc:
            self.model.eval()
            acc = self.test_batch(batch)
            self.model.train()

        return BatchResult(loss.item())

    def test_batch(self, batch) -> BatchResult:
        x, encoder_attention_mask, y, decoder_attention_mask = batch
        y = y.to(self.device)
        encoder_attention_mask = encoder_attention_mask.to(self.device)
        decoder_attention_mask = decoder_attention_mask.to(self.device)

        total_loss = torch.zeros(1, dtype=float)

        batch_size = x.size(0)

        best_res = [torch.tensor(1e9) for _ in range(batch_size)]

        inp_x = []
        inp_y = []
        inp_e_a_m = []
        inp_d_a_m = []

        curr_x = x.clone().to(self.device)
        inp_x.append(curr_x)
        inp_y.append(y.clone())
        inp_e_a_m.append(encoder_attention_mask.clone())
        inp_d_a_m.append(decoder_attention_mask.clone())

        inp_x = torch.cat(inp_x)
        inp_y = torch.cat(inp_y)
        inp_e_a_m = torch.cat(inp_e_a_m)
        inp_d_a_m = torch.cat(inp_d_a_m)

        model_kwargs = {
            "encoder_attention_mask": inp_e_a_m,
            "decoder_attention_mask": inp_d_a_m,
            "decoder_lm_labels": inp_y
        }

        with torch.no_grad():
            outputs = self.model(inp_x, inp_y, **model_kwargs)
            loss = outputs[0]

        loss = loss.view(inp_x.size(0), -1)
        loss = loss.mean(dim=1)

        for j in range(batch_size):
            curr_loss = loss[j]
            if loss[j] < best_res[j]:
                best_res[j] = curr_loss

        total_loss = torch.sum(torch.tensor([l for l in best_res]))

        return BatchResult(total_loss.item())