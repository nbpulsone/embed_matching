import os
import argparse
import json
import sys
import torch
import numpy as np
import random

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.ditto import train

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/Beer")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--contrastive_margin", type=float, default=0.3)
    parser.add_argument("--loss", choices=["contrastive", "infonce"], default="contrastive")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lm_left", type=str, default=None)
    parser.add_argument("--lm_right", type=str, default=None)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--zero_shot_threshold", type=float, default=0.5)
    parser.add_argument("--bert_pooling", dest="bert_pooling", action="store_true")

    hp = parser.parse_args()

    # determine whether heterogenous encoders are used for left and right entities
    hp.lm_left = hp.lm_left or hp.lm
    hp.lm_right = hp.lm_right or hp.lm
    hp.heterogeneous = hp.lm_left != hp.lm_right

    if hp.da is not None:
        raise ValueError(
            "MixDA is not yet supported for late-fusion training. "
            "Use --da None (default)."
        )

    if hp.fp16 and hp.bf16:
        raise ValueError("Use either --fp16 or --bf16, not both.")

    if (hp.fp16 or hp.bf16) and not torch.cuda.is_available():
        raise ValueError("Mixed precision requires a CUDA GPU.")

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # only a single task for baseline
    task = hp.task

    # create the tag of the run
    run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, hp.lm, hp.da,
            hp.dk, hp.summarize, str(hp.size), hp.run_id)
    run_tag = run_tag.replace('/', '_')
    run_tag += (
        f"_left={hp.lm_left.replace('/', '_')}"
        f"_right={hp.lm_right.replace('/', '_')}"
        f"_loss={hp.loss}"
        f"_proj={hp.projection_dim}"
    )
    if hp.zero_shot:
        run_tag += "_zeroshot"

    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']

    # summarize the sequences up to the max sequence length
    if hp.summarize:
        summarizer = Summarizer(config, lm=hp.lm)
        trainset = summarizer.transform_file(trainset, max_len=hp.max_len)
        validset = summarizer.transform_file(validset, max_len=hp.max_len)
        testset = summarizer.transform_file(testset, max_len=hp.max_len)

    if hp.dk is not None:
        if hp.dk == 'product':
            injector = ProductDKInjector(config, hp.dk)
        else:
            injector = GeneralDKInjector(config, hp.dk)

        trainset = injector.transform_file(trainset)
        validset = injector.transform_file(validset)
        testset = injector.transform_file(testset)

    # load train/dev/test sets
    train_dataset = DittoDataset(trainset,
                                 lm_left=hp.lm_left,
                                 lm_right=hp.lm_right,
                                 lm=hp.lm,
                                 max_len=hp.max_len,
                                 size=hp.size,
                                 da=hp.da)
    valid_dataset = DittoDataset(validset, lm_left=hp.lm_left, lm_right=hp.lm_right, lm=hp.lm, max_len=hp.max_len)
    test_dataset = DittoDataset(testset, lm_left=hp.lm_left, lm_right=hp.lm_right, lm=hp.lm, max_len=hp.max_len)

    # train and evaluate the model
    best_test_f1, best_val_f1 = train(train_dataset, valid_dataset, test_dataset, run_tag, hp)

    print(
    f"~~~~ FINAL F1 RESULTS "
    f"(loss={hp.loss}, "
    f"parameter={hp.temperature if hp.loss == 'infonce' else hp.contrastive_margin}) ~~~~")
    print(f"val f1 = {best_val_f1}")
    print(f"test f1 = {best_test_f1}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
