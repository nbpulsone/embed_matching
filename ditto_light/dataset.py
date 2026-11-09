import torch

from torch.utils import data
from transformers import AutoTokenizer

from .augment import Augmenter

# map lm name to huggingface's pre-trained model names
lm_mp = {
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased',
    'bert': 'bert-base-uncased'
}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


class DittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self, path, max_len=256, size=None, lm='roberta', da=None, lm_left=None, lm_right=None):
        lm_left = lm_left or lm
        lm_right = lm_right or lm

        self.tokenizer_left = get_tokenizer(lm_left)

        if lm_left == lm_right:
            self.tokenizer_right = self.tokenizer_left
        else:
            self.tokenizer_right = get_tokenizer(lm_right)

        #self.tokenizer = get_tokenizer(lm)
        self.tokenizer = self.tokenizer_left
        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size

        if isinstance(path, list):
            lines = path
        else:
            lines = open(path)

        for line in lines:
            s1, s2, label = line.strip().split('\t')
            self.pairs.append((s1, s2))
            self.labels.append(int(label))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]
        self.da = da
        if da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]

        # encode left and right entities independently with their respective encoders
        x_left = self.tokenizer_left.encode(left, max_length=self.max_len, truncation=True)
        x_right = self.tokenizer_right.encode(right, max_length=self.max_len, truncation=True)

        return x_left, x_right, self.labels[idx]


    def pad(self, batch):
        x_left, x_right, y = zip(*batch)

        pad_left = self.tokenizer_left.pad_token_id
        pad_right = self.tokenizer_right.pad_token_id

        # Pad each encoder's inputs independently
        maxlen_left = max(len(x) for x in x_left)
        maxlen_right = max(len(x) for x in x_right)

        x_left = [
            x + [pad_left] * (maxlen_left - len(x))
            for x in x_left
        ]

        x_right = [
            x + [pad_right] * (maxlen_right - len(x))
            for x in x_right
        ]

        return (
            torch.LongTensor(x_left),
            torch.LongTensor(x_right),
            torch.LongTensor(y)
        )
