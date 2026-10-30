import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse

from .dataset import DittoDataset
from torch.utils import data
from transformers import AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tensorboardX import SummaryWriter

lm_mp = {
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased',
    'bert': 'bert-base-uncased'
}


class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta',
                 alpha_aug=0.8, pad_token_id=0):
        super().__init__()

        model_name = lm_mp.get(lm, lm)
        self.bert = AutoModel.from_pretrained(model_name)

        self.device = device
        self.alpha_aug = alpha_aug
        self.pad_token_id = pad_token_id

        hidden_size = self.bert.config.hidden_size
        self.fc = nn.Linear(hidden_size, 2)

    def _encode(self, x):
        attention_mask = (x != self.pad_token_id).long()

        outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask
        )

        return outputs.last_hidden_state[:, 0, :]

    def forward(self, x1, x2=None, embed=False):
        x1 = x1.to(self.device)

        if x2 is not None:
            x2 = x2.to(self.device)

            enc = self._encode(torch.cat((x1, x2)))
            batch_size = len(x1)

            enc1 = enc[:batch_size]
            enc2 = enc[batch_size:]

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self._encode(x1)

        if embed:
            return enc

        return self.fc(enc)


def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0 # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th


def train_step(train_iter, model, optimizer, scheduler, hp, scaler=None):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    use_fp16 = hp.fp16
    use_bf16 = hp.bf16

    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16 if use_fp16 else torch.bfloat16,
            enabled=use_fp16 or use_bf16
        ):
            if len(batch) == 2:
                x, y = batch
                prediction = model(x)
            else:
                x1, x2, y = batch
                prediction = model(x1, x2)

            loss = criterion(prediction, y.to(model.device))

        if use_fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        if i % 10 == 0:
            print(f"step: {i}, loss: {loss.item()}")

        del loss


def train(trainset, validset, testset, run_tag, hp):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device, lm=hp.lm, alpha_aug=hp.alpha_aug, pad_token_id=trainset.tokenizer.pad_token_id).to(device)
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=hp.fp16
    )
    
    num_steps = len(train_iter) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)

    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(1, hp.n_epochs+1):
        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp, scaler)

        # eval
        model.eval()
        dev_f1, th = evaluate(model, valid_iter)
        test_f1 = evaluate(model, test_iter, threshold=th)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # logging
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, epoch)

    writer.close()
    return best_test_f1, best_dev_f1
