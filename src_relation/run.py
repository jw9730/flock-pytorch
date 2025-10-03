import os
import sys
import csv
import time
import math
import pprint
import copy
from itertools import islice
import tqdm

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from flock import tasks, util
from flock.models import Flock

separator = ">" * 30
line = "-" * 30


def train_and_validate(
    cfg,
    model,
    train_data,
    valid_data,
    device,
    logger,
    filtered_data=None,
    batch_per_epoch=None,
):
    if cfg.train.num_epoch == 0:
        return -1

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_triplets = torch.cat(
        [train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]
    ).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(
        train_triplets, cfg.train.batch_size, sampler=sampler
    )

    batch_per_epoch = batch_per_epoch or len(train_loader)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = test(
        cfg,
        model,
        valid_data,
        filtered_data=filtered_data,
        device=device,
        logger=logger,
    )
    best_epoch = -1

    train_graph = train_data

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                batch = tasks.negative_sampling_relation(
                    train_graph,
                    batch,
                    cfg.task.num_negative,
                    strict=cfg.task.strict_negative,
                )
                pred = parallel_model(train_data, batch)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(
                    pred, target, reduction="none"
                )
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(
                            pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1
                        )
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning(
                        "batch id: "
                        + str(batch_id)
                        + " binary cross entropy: %g" % loss
                    )
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(
            cfg,
            model,
            valid_data,
            filtered_data=filtered_data,
            device=device,
            logger=logger,
        )
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    if best_epoch != -1:
        state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
        model.load_state_dict(state["model"])
    else:
        state = torch.load("model_epoch_1.pth", map_location=device)
        model.load_state_dict(state["model"])
    util.synchronize()
    return best_epoch


@torch.no_grad()
def test(
    cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False
):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat(
        [test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]
    ).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)

    # use `test_batch_size` for testing if it is specified
    if "test_batch_size" in cfg.train:
        test_batch_size = cfg.train.test_batch_size
    else:
        test_batch_size = cfg.train.batch_size
    test_loader = torch_data.DataLoader(test_triplets, test_batch_size, sampler=sampler)

    model.eval()
    rankings = []
    num_negatives = []
    test_graph = test_data

    for batch in tqdm.tqdm(test_loader):
        pos_h_index, pos_t_index, pos_r_index = batch.t()

        r_batch = tasks.all_negative_relation(test_graph, batch)

        r_pred = model(test_graph, r_batch)

        r_ranking = tasks.compute_ranking_relation(r_pred, pos_r_index, None)
        rankings += [r_ranking]
        num_r_negative = test_graph.num_relations // 2 - 1
        num_negatives += [num_r_negative.expand_as(r_ranking)]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)

    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)

    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank] : cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank] : cum_size[rank]] = num_negative

    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

    metrics = {}
    if rank == 0:
        for metric in cfg.task.metric:
            _ranking = all_ranking
            _num_neg = all_num_negative
            _metric_name = metric

            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        num_comb = (
                            math.factorial(num_sample - 1)
                            / math.factorial(i)
                            / math.factorial(num_sample - i - 1)
                        )
                        score += (
                            num_comb
                            * (fp_rate**i)
                            * ((1 - fp_rate) ** (num_sample - i - 1))
                        )
                    score = score.mean()
                else:
                    score = (_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
            metrics[metric] = score
    mrr = (1 / all_ranking.float()).mean()

    return mrr if not return_metrics else metrics


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)

    path = os.path.dirname(os.path.expanduser(__file__))
    path = os.path.join(path, "results")
    if args.subdir is not None:
        path = os.path.join(path, args.subdir)
    os.makedirs(path, exist_ok=True)

    results_file = os.path.join(
        path, f"results_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv"
    )

    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    if "fast_test" in cfg.train:
        num_val_edges = cfg.train.fast_test
        if util.get_rank() == 0:
            logger.warning(f"Fast evaluation on {num_val_edges} samples in validation")
        short_valid = copy.deepcopy(valid_data)

        mask = torch.randperm(short_valid.target_edge_index.shape[1])[:num_val_edges]
        short_valid.target_edge_index = short_valid.target_edge_index[:, mask]
        short_valid.target_edge_type = short_valid.target_edge_type[mask]

    if cfg.model["class"] == "Flock":
        model = Flock(model_cfg=cfg.model)
    else:
        raise NotImplementedError(
            f"Model class {cfg.model['class']} is not implemented"
        )

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    model = model.to(device)

    if task_name == "InductiveInference":
        # filtering for inductive datasets
        # Grail, MTDEA, HM datasets have validation sets based off the training graph
        # ILPC, Ingram have validation sets from the inference graph
        # filtering dataset should contain all true edges (base graph + (valid) + test)
        if "ILPC" in cfg.dataset["class"] or "Ingram" in cfg.dataset["class"]:
            # add inference, valid, test as the validation and test filtering graphs
            full_inference_edges = torch.cat(
                [
                    valid_data.edge_index,
                    valid_data.target_edge_index,
                    test_data.target_edge_index,
                ],
                dim=1,
            )
            full_inference_etypes = torch.cat(
                [
                    valid_data.edge_type,
                    valid_data.target_edge_type,
                    test_data.target_edge_type,
                ]
            )
            test_filtered_data = Data(
                edge_index=full_inference_edges,
                edge_type=full_inference_etypes,
                num_nodes=test_data.num_nodes,
            )
            val_filtered_data = test_filtered_data
        else:
            # test filtering graph: inference edges + test edges
            full_inference_edges = torch.cat(
                [test_data.edge_index, test_data.target_edge_index], dim=1
            )
            full_inference_etypes = torch.cat(
                [test_data.edge_type, test_data.target_edge_type]
            )
            test_filtered_data = Data(
                edge_index=full_inference_edges,
                edge_type=full_inference_etypes,
                num_nodes=test_data.num_nodes,
            )

            # validation filtering graph: train edges + validation edges
            val_filtered_data = Data(
                edge_index=torch.cat(
                    [train_data.edge_index, valid_data.target_edge_index], dim=1
                ),
                edge_type=torch.cat(
                    [train_data.edge_type, valid_data.target_edge_type]
                ),
            )
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = Data(
            edge_index=dataset._data.target_edge_index,
            edge_type=dataset._data.target_edge_type,
            num_nodes=dataset[0].num_nodes,
        )
        val_filtered_data = test_filtered_data = filtered_data

    val_filtered_data = val_filtered_data.to(device)
    test_filtered_data = test_filtered_data.to(device)

    best_epoch = train_and_validate(
        cfg,
        model,
        train_data,
        valid_data if "fast_test" not in cfg.train else short_valid,
        filtered_data=val_filtered_data,
        device=device,
        batch_per_epoch=cfg.train.batch_per_epoch,
        logger=logger,
    )
    if best_epoch == -1:
        state = torch.load(cfg.checkpoint, map_location=device)
        model.load_state_dict(state["model"])
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    metrics = test(
        cfg,
        model,
        test_data,
        filtered_data=test_filtered_data,
        return_metrics=True,
        device=device,
        logger=logger,
    )

    metrics = {k: v.item() for k, v in metrics.items()}
    metrics["dataset"] = repr(dataset)
    # write to the log file
    with open(results_file, "a", newline="") as csv_file:
        fieldnames = ["dataset"] + list(metrics.keys())[:-1]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=",")
        if csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow(metrics)
