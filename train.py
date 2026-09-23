#!/usr/bin/env python3
"""Train and evaluate the paper's RQR-KGC model."""

import argparse, json, logging, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from curriculum import curriculum_progress, select_curriculum_negatives
from data import BidirectionalOneShotIterator, TestDataset, TrainDataset
from model import RQRMKGC


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=Path, required=True)
    p.add_argument("--save_path", type=Path, required=True)
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_valid", action="store_true")
    p.add_argument("--do_test", action="store_true")
    p.add_argument("--hidden_dim", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=10.0)
    p.add_argument("--negative_sample_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--test_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--residual_lr_scale", type=float, default=0.5)
    p.add_argument("--scale_lr_scale", type=float, default=4.0)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--decay_steps", nargs="+", type=int, default=(20000, 50000, 80000))
    p.add_argument("--lr_decay_factor", type=float, default=10.0)
    p.add_argument("--regularization", type=float, default=0.4)
    p.add_argument("--residual_regularization", type=float, default=1e-4)
    p.add_argument("--adversarial_temperature", type=float, default=0.5)
    p.add_argument("--rank_loss_weight", type=float, default=0.005)
    p.add_argument("--rank_loss_margin", type=float, default=0.5)
    p.add_argument("--rank_loss_topk", type=int, default=10)
    p.add_argument("--rank_loss_cutoffs", nargs="+", type=int, default=(1, 3, 10))
    p.add_argument("--curriculum_hard_negative_min_ratio", type=float, default=0.05)
    p.add_argument("--curriculum_hard_negative_max_ratio", type=float, default=0.05)
    p.add_argument("--curriculum_hard_negative_start_step", type=int, default=4000)
    p.add_argument("--curriculum_hard_negative_ramp_steps", type=int, default=8000)
    p.add_argument("--residual_bound", type=float, default=0.1)
    p.add_argument("--endpoint_bound", type=float, default=0.5)
    p.add_argument("--valid_steps", type=int, default=2000)
    p.add_argument("--log_steps", type=int, default=100)
    p.add_argument("--cpu_num", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260806)
    return p.parse_args()


def read_dict(path):
    with path.open() as handle:
        return {value: int(index) for index, value in
                (line.rstrip("\n").split("\t") for line in handle)}


def read_triples(path, entities, relations):
    triples = []
    with path.open() as handle:
        for line in handle:
            h, r, t = line.rstrip("\n").split("\t")
            triples.append((entities[h], relations[r], entities[t]))
    return triples


def make_optimizer(model, args):
    special = {id(model.forward_residual), id(model.reverse_residual),
               id(model.endpoint_scale_raw)}
    base = [p for p in model.parameters() if p.requires_grad and id(p) not in special]
    return torch.optim.Adam([
        {"params": base, "lr": args.learning_rate},
        {"params": [model.forward_residual, model.reverse_residual],
         "lr": args.learning_rate * args.residual_lr_scale},
        {"params": [model.endpoint_scale_raw],
         "lr": args.learning_rate * args.scale_lr_scale},
    ])


def train_step(model, optimizer, iterator, args, step, device):
    model.train(); batch = next(iterator)
    if len(batch) == 6:
        positive, uniform, hard, cap, weight, mode = batch
        progress = curriculum_progress(step,
            args.curriculum_hard_negative_start_step,
            args.curriculum_hard_negative_ramp_steps)
        negative = select_curriculum_negatives(uniform, hard, cap, progress)
    else:
        positive, negative, weight, mode = batch
    positive, negative, weight = positive.to(device), negative.to(device), weight.to(device)
    optimizer.zero_grad(set_to_none=True)
    neg = model((positive, negative), mode)
    neg_ll = (F.softmax(neg * args.adversarial_temperature, 1).detach()
              * F.logsigmoid(-neg)).sum(1)
    positive_mode = "head-single" if mode == "head-batch" else "tail-single"
    pos = model(positive, positive_mode).squeeze(1)
    denominator = weight.sum()
    loss = -0.5 * ((weight * F.logsigmoid(pos)).sum() / denominator
                   + (weight * neg_ll).sum() / denominator)
    top = neg.topk(min(args.rank_loss_topk, neg.size(1)), 1).values
    pair = F.softplus(top - pos[:, None] + args.rank_loss_margin).mean()
    cutoffs = [k for k in args.rank_loss_cutoffs if k <= neg.size(1)]
    selected = neg.topk(max(cutoffs), 1).values[:,
        torch.tensor([k - 1 for k in cutoffs], device=device)]
    cutoff = F.softplus(selected - pos[:, None] + args.rank_loss_margin).mean()
    _, translation, _ = torch.chunk(model.relation_embedding, 3, 1)
    base_reg = torch.cat((model.entity_embedding.square().sum(1),
                          translation.square().sum(1))).mean()
    residual_reg = 0.5 * (model.forward_residual.square().sum(1).mean()
                          + model.reverse_residual.square().sum(1).mean())
    total = (loss + args.rank_loss_weight * 0.5 * (pair + cutoff)
             + args.regularization * base_reg
             + args.residual_regularization * residual_reg)
    total.backward(); optimizer.step()
    return total.item()


def evaluate(model, triples, all_true, nentity, nrelation, args, device):
    model.eval(); ranks = []
    with torch.no_grad():
        for mode in ("head-batch", "tail-batch"):
            loader = DataLoader(TestDataset(triples, all_true, nentity, nrelation, mode),
                batch_size=args.test_batch_size, num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn)
            for positive, candidates, bias, _ in loader:
                positive = positive.to(device)
                scores = model((positive, candidates.to(device)), mode) + bias.to(device)
                order = scores.argsort(1, descending=True)
                target = positive[:, 0] if mode == "head-batch" else positive[:, 2]
                ranks.extend(((order == target[:, None]).nonzero()[:, 1] + 1).tolist())
    rank = torch.tensor(ranks, dtype=torch.float64)
    return {"MRR": (1 / rank).mean().item(), "MR": rank.mean().item(),
            "HITS@1": (rank <= 1).double().mean().item(),
            "HITS@3": (rank <= 3).double().mean().item(),
            "HITS@10": (rank <= 10).double().mean().item()}


def save(path, model, args, step, metrics):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model_state_dict": model.state_dict(),
                "valid": metrics}, path / "checkpoint")
    (path / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")


def main():
    args = parse_args(); args.save_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.save_path / "train.log")])
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    entities, relations = read_dict(args.data_path / "entities.dict"), read_dict(args.data_path / "relations.dict")
    train = read_triples(args.data_path / "train.txt", entities, relations)
    valid = read_triples(args.data_path / "valid.txt", entities, relations)
    test = read_triples(args.data_path / "test.txt", entities, relations)
    all_true = train + valid + test
    model = RQRMKGC(len(entities), len(relations), args.hidden_dim, args.gamma,
                    args.residual_bound, args.endpoint_bound).to(device)
    optimizer = make_optimizer(model, args); best_mrr = -1.; best = args.save_path / "best_valid"
    if args.do_train:
        loaders = [DataLoader(TrainDataset(train, len(entities), len(relations),
            args.negative_sample_size, mode,
            curriculum_hard_negative_min_ratio=args.curriculum_hard_negative_min_ratio,
            curriculum_hard_negative_max_ratio=args.curriculum_hard_negative_max_ratio),
            batch_size=args.batch_size, shuffle=True, num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn) for mode in ("head-batch", "tail-batch")]
        iterator = BidirectionalOneShotIterator(*loaders)
        for step in range(args.max_steps):
            if step in args.decay_steps:
                for group in optimizer.param_groups: group["lr"] /= args.lr_decay_factor
            loss = train_step(model, optimizer, iterator, args, step, device)
            if step % args.log_steps == 0: logging.info("step=%d loss=%.6f", step, loss)
            if args.do_valid and ((step + 1) % args.valid_steps == 0 or step + 1 == args.max_steps):
                metrics = evaluate(model, valid, all_true, len(entities), len(relations), args, device)
                logging.info("valid step=%d %s", step + 1, metrics)
                if metrics["MRR"] > best_mrr:
                    best_mrr = metrics["MRR"]; save(best, model, args, step + 1, metrics)
    if args.do_test:
        checkpoint = torch.load(best / "checkpoint", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics = evaluate(model, test, all_true, len(entities), len(relations), args, device)
        result = {"step": checkpoint["step"], **metrics}
        logging.info("test %s", result)
        (args.save_path / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__": main()
