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
    """Late-fusion entity matching with shared or heterogeneous encoders."""

    def __init__(self, device='cuda', lm='roberta',
             alpha_aug=0.8, pad_token_id=0,
             lm_left=None, lm_right=None,
             pad_token_id_right=None, projection_dim=256,
             zero_shot=False, bert_pooling=False):
        super().__init__()

        lm_left = lm_left or lm
        lm_right = lm_right or lm

        self.device = device
        self.alpha_aug = alpha_aug

        self.heterogeneous = (lm_left != lm_right)
        self.zero_shot = zero_shot
        self.bert_pooling = bert_pooling

        # Left encoder
        self.bert = AutoModel.from_pretrained(lm_mp.get(lm_left, lm_left))
        self.pad_token_id = pad_token_id

        if self.heterogeneous:
            # Independent right encoder
            self.bert_right = AutoModel.from_pretrained(
                lm_mp.get(lm_right, lm_right)
            )
            self.pad_token_id_right = pad_token_id_right

            if self.zero_shot or projection_dim == 0:
                # No learned projection heads
                if self.bert.config.hidden_size != self.bert_right.config.hidden_size:
                    raise ValueError(
                        "Without projections, left/right embedding dimensions must match."
                    )

                self.proj_left = nn.Identity()
                self.proj_right = nn.Identity()
            else:
                # Learn mappings into a common embedding space
                self.proj_left = nn.Linear(
                    self.bert.config.hidden_size, projection_dim
                )
                self.proj_right = nn.Linear(
                    self.bert_right.config.hidden_size, projection_dim
                )
        else:
            # Preserve original homogeneous architecture
            self.bert_right = self.bert
            self.pad_token_id_right = self.pad_token_id
            self.proj_left = nn.Identity()
            self.proj_right = nn.Identity()

    """
    def _encode(self, x, encoder, pad_token_id):
        attention_mask = (x != pad_token_id).long()

        outputs = encoder(
            input_ids=x,
            attention_mask=attention_mask
        )

        return outputs.last_hidden_state[:, 0, :]
    """

    def _encode(self, x, encoder, pad_token_id):
        attention_mask = (x != pad_token_id).long()

        outputs = encoder(
            input_ids=x,
            attention_mask=attention_mask
        )

        if self.bert_pooling:
            # mean pooling over non-padding tokens
            mask = attention_mask.unsqueeze(-1)
            embeddings = outputs.last_hidden_state

            return (
                (embeddings * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1)
            )
            
        return outputs.last_hidden_state[:, 0, :]


    def forward(self, x_left, x_right, embed=False):
        x_left = x_left.to(self.device)
        x_right = x_right.to(self.device)

        # Independently encode left and right entities
        z_left = self._encode(
            x_left, self.bert, self.pad_token_id
        )
        z_right = self._encode(
            x_right, self.bert_right, self.pad_token_id_right
        )

        # Project into shared space (identity for homogeneous case)
        z_left = self.proj_left(z_left.float())
        z_right = self.proj_right(z_right.float())

        if embed:
            return z_left, z_right

        return F.cosine_similarity(z_left, z_right, dim=1)


def evaluate(model, iterator, threshold=None):
    all_scores = []
    all_y = []

    model.eval()
    with torch.no_grad():
        for x_left, x_right, y in iterator:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(model.device == "cuda")
            ):
                scores = model(x_left, x_right)

            all_scores.extend(scores.float().cpu().tolist())
            all_y.extend(y.cpu().tolist())

    all_scores = np.asarray(all_scores)
    all_y = np.asarray(all_y)

    if threshold is None:
        pos_scores = all_scores[all_y == 1]
        neg_scores = all_scores[all_y == 0]

        print(
            f"Positive similarity: {pos_scores.mean():.4f}, "
            f"Negative similarity: {neg_scores.mean():.4f}, "
            f"Separation: {pos_scores.mean() - neg_scores.mean():.4f}"
        )

    if threshold is not None:
        pred = (all_scores >= threshold).astype(int)
        return metrics.f1_score(all_y, pred, zero_division=0)

    # Find best threshold on validation data
    thresholds = np.unique(all_scores)
    best_f1 = -1.0
    best_th = 0.0

    for th in thresholds:
        pred = (all_scores >= th).astype(int)
        f1 = metrics.f1_score(all_y, pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)

    return best_f1, best_th


def contrastive_loss(similarity, labels, margin=0.3):
    """Supervised pairwise contrastive loss using cosine similarity."""
    labels = labels.float()
    similarity = similarity.float()

    positive_loss = labels * (1.0 - similarity).pow(2)
    negative_loss = (1.0 - labels) * F.relu(similarity - margin).pow(2)

    return (positive_loss + negative_loss).mean()


def supervised_infonce_loss(z_left, z_right, labels, temperature=0.07):
    """
    InfoNCE using labeled positive pairs and explicitly labeled
    negative pairs as competing candidates.

    Does not assume unlabeled cross-pairs are negatives.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    labels = labels.bool()

    # Normalize embeddings for cosine similarity
    z_left = F.normalize(z_left.float(), dim=1)
    z_right = F.normalize(z_right.float(), dim=1)

    similarity = (z_left * z_right).sum(dim=1) / temperature

    positives = similarity[labels]
    negatives = similarity[~labels]

    # InfoNCE requires positive and negative examples
    if positives.numel() == 0 or negatives.numel() == 0:
        return 0.0 * similarity.sum()

    # Each positive competes against the explicitly labeled negatives
    logits = torch.cat([
        positives.unsqueeze(1),
        negatives.unsqueeze(0).expand(positives.size(0), -1)
    ], dim=1)

    targets = torch.zeros(positives.size(0), dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, targets)


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
    # criterion = nn.BCEWithLogitsLoss()
    use_fp16 = hp.fp16
    use_bf16 = hp.bf16

    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16 if use_fp16 else torch.bfloat16,
            enabled=use_fp16 or use_bf16
        ):
            x_left, x_right, y = batch
            if hp.loss == "contrastive":
                # contrastive loss: encourage model to seperate positives and negatives in embedding space
                similarity = model(x_left, x_right)
                loss = contrastive_loss(similarity, y.to(model.device), margin=hp.contrastive_margin)
            elif hp.loss == "infonce":
                # infoNCE contrastive learning
                z_left, z_right = model(x_left, x_right, embed=True)
                loss = supervised_infonce_loss(z_left, z_right, y.to(model.device), temperature=hp.temperature)

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
                                 batch_size=hp.batch_size*8, # reduce more if memory is an issue
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*8, # reduce more if memory is an issue
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device, lm=hp.lm, lm_left=hp.lm_left, lm_right=hp.lm_right, alpha_aug=hp.alpha_aug,
                       pad_token_id=trainset.tokenizer_left.pad_token_id, pad_token_id_right=trainset.tokenizer_right.pad_token_id,
                       projection_dim=hp.projection_dim, zero_shot=hp.zero_shot, bert_pooling=hp.bert_pooling).to(device)

    # zero shot: extract embeddings, late fusion similarity, and evaluate w/o training
    if hp.zero_shot:
        model.eval()
        print(f"ZERO-SHOT: {hp.lm_left} / {hp.lm_right}, threshold={hp.zero_shot_threshold}")
        val_f1 = evaluate(model, valid_iter, threshold=hp.zero_shot_threshold)
        test_f1 = evaluate(model, test_iter, threshold=hp.zero_shot_threshold)
        print(f"Zero-shot validation F1: {val_f1:.4f}")
        print(f"Zero-shot test F1: {test_f1:.4f}")
        return test_f1, val_f1

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
                directory = os.path.join(hp.logdir, run_tag)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(directory, 'model.pt')
                ckpt = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'threshold': float(th),
                    'lm': hp.lm,
                    'lm_left': hp.lm_left,
                    'lm_right': hp.lm_right,
                    'projection_dim': hp.projection_dim,
                    'loss': hp.loss,
                    'contrastive_margin': hp.contrastive_margin,
                    'temperature': hp.temperature
                }
                torch.save(ckpt, ckpt_path)

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # logging
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, epoch)

    writer.close()
    return best_test_f1, best_dev_f1
