"""GRPO for a small external-expert router, with a separate demographic constraint loss.

This trains routing parameters only. Expert caches are frozen. No invented probabilities,
embeddings or patient outcomes are used. See FAIR_ROUTING_RESEARCH.md for scope/limits.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from audit_expert_routing import racial_summary, scores
from fairroute_data import (RACES, check_separation, feature_matrix, file_hash,
                            load_bundle, prepare_bundle)
from fairvision_local_vlm import TASKS, atomic_json, binary, digest, write_csv


class RoutingAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        # No dropout: behavior and update probabilities must describe the same policy.
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, num_experts))

    def forward(self, features):
        return self.network(features)


def utilities(predictions, truth, tp_bonus=.3, fn_penalty=.5, costs=None, cost_weight=0.):
    correct = (predictions == truth[:, None]).float()
    tp = (predictions == 1) & (truth[:, None] == 1)
    fn = (predictions == 0) & (truth[:, None] == 1)
    reward = correct + tp_bonus * tp.float() - fn_penalty * fn.float()
    if costs is not None:
        reward = reward - cost_weight * costs[None, :]
    return reward


def group_advantages(rewards):
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    scale = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (centered / (scale + 1e-8)).detach()


def clipped_grpo_loss(logits, actions, old_log_probs, advantages, case_weights, epsilon):
    new_log_probs = logits.log_softmax(dim=-1).gather(1, actions)
    ratios = (new_log_probs - old_log_probs).exp()
    surrogate = torch.minimum(ratios * advantages, ratios.clamp(1 - epsilon, 1 + epsilon) * advantages)
    return -(surrogate.mean(dim=1) * case_weights).mean()


def fairness_pools(data, minimum):
    pools = {}
    support = {}
    for task in TASKS:
        groups = []
        support[task] = {}
        for race in RACES[:-1]:
            indices = np.asarray([i for i, case in enumerate(data["cases"])
                                  if case["task"] == task and case["race"] == race and binary(case["y_true"]) == 1], dtype=int)
            support[task][race] = len(indices)
            if len(indices) >= minimum:
                groups.append((race, indices))
        if len(groups) >= 2:
            pools[task] = groups
    return pools, support


def fairness_loss(router, features, predictions, pools, threshold, rng, samples):
    penalties = []
    for groups in pools.values():
        rates = []
        for _, indices in groups:
            chosen = rng.choice(indices, size=min(samples, len(indices)), replace=False)
            selected = torch.as_tensor(chosen, device=features.device)
            probability = router(features[selected]).softmax(dim=-1)
            # All selected cases are positive. This is differentiable expected FNR.
            rates.append((probability * (1 - predictions[selected])).sum(dim=-1).mean())
        rates = torch.stack(rates)
        penalties.append(torch.relu(rates.max() - rates.min() - threshold).square())
    return torch.stack(penalties).mean() if penalties else features.sum() * 0.


def class_weights(data):
    keys = [(case["task"], binary(case["y_true"])) for case in data["cases"]]
    counts = {key: keys.count(key) for key in set(keys)}
    # Outside GRPO normalization: per-case scaling inside rewards would cancel.
    return np.asarray([len(keys) / (len(counts) * counts[key]) for key in keys], dtype=np.float32)


def tensors(data, settings, device, scaler=None):
    matrix = feature_matrix(data, settings["race_aware"], settings["mode"])
    if scaler is None:
        mean, std = matrix.mean(axis=0), matrix.std(axis=0)
        std[std < 1e-6] = 1.
        scaler = dict(mean=mean.tolist(), std=std.tolist())
    if len(scaler["mean"]) != matrix.shape[1]:
        raise ValueError("Router feature dimensions differ")
    matrix = (matrix - np.asarray(scaler["mean"], dtype=np.float32)) / np.asarray(scaler["std"], dtype=np.float32)
    return (torch.as_tensor(matrix, device=device),
            torch.tensor([case["expert_preds"] for case in data["cases"]], dtype=torch.float32, device=device),
            torch.tensor([binary(case["y_true"]) for case in data["cases"]], dtype=torch.float32, device=device), scaler)


def baseline_mappings(validation, minimum, shrinkage=50.):
    global_map, race_map, reliability_map = {}, {}, {}
    experts = validation["experts"]
    for task in TASKS:
        cases = [c for c in validation["cases"] if c["task"] == task]
        if not cases:
            continue
        truth = np.asarray([binary(c["y_true"]) for c in cases])
        race = np.asarray([c["race"] for c in cases])
        reports = [racial_summary(truth, np.asarray([c["expert_preds"][e] for c in cases]), race)
                   for e in range(len(experts))]
        global_map[task] = max(range(len(experts)), key=lambda e: reports[e][0]["f1_macro"])
        for group in RACES[:-1]:
            rows = [report[1].get(group) for report in reports]
            if rows[0] is None or min(rows[0]["positive"], rows[0]["negative"]) < minimum:
                race_map[f"{task}|{group}"] = global_map[task]
                reliability_map[f"{task}|{group}"] = global_map[task]
                continue
            race_map[f"{task}|{group}"] = max(range(len(experts)), key=lambda e: rows[e]["f1_macro"])
            weight = rows[0]["n"] / (rows[0]["n"] + shrinkage)
            reliability_map[f"{task}|{group}"] = max(
                range(len(experts)), key=lambda e: weight * rows[e]["balanced_accuracy"] +
                (1 - weight) * reports[e][0]["balanced_accuracy"])
    return dict(global_expert=global_map, race_expert=race_map, shrunken_reliability=reliability_map,
                minimum_class_support=minimum, shrinkage=shrinkage)


def metric_tables(data, predictions):
    overall, racial = [], []
    for task in TASKS:
        indices = np.asarray([i for i, case in enumerate(data["cases"]) if case["task"] == task], dtype=int)
        if not len(indices):
            continue
        truth = np.asarray([binary(data["cases"][i]["y_true"]) for i in indices])
        races = np.asarray([data["cases"][i]["race"] for i in indices])
        for method, values in predictions.items():
            summary, groups = racial_summary(truth, np.asarray(values)[indices], races)
            overall.append(dict(task=task, method=method, race_fnr_gap=summary["race_sensitivity_gap"], **summary))
            racial.extend(dict(task=task, method=method, race=group,
                               fnr=1 - row["sensitivity"] if row["sensitivity"] is not None else None,
                               **row) for group, row in groups.items())
    return overall, racial


def selection_score(data, predictions, minimum, threshold, weight):
    overall, groups = metric_tables(data, {"router": predictions})
    violations, supported = [], []
    for task in TASKS:
        fnrs = [row["fnr"] for row in groups if row["task"] == task and row["positive"] >= minimum]
        if len(fnrs) >= 2:
            supported.append(task)
            violations.append(max(max(fnrs) - min(fnrs) - threshold, 0.))
    macro = float(np.mean([row["f1_macro"] for row in overall]))
    penalty = float(np.mean(violations)) if violations else 0.
    return macro - weight * penalty, dict(mean_task_f1=macro, supported_fairness_tasks=supported,
                                         fnr_gap_violation=penalty if supported else None)


def identity_snapshot(data):
    return dict(experts=data["experts"], identity=data["provenance"]["identity"],
                protocol=[data["provenance"].get("prompts_sha256"), data["provenance"].get("oct_slices")],
                embedding_dim=len(data["cases"][0].get("embedding", [])),
                patients=sorted({str(c["patient_id"]) for c in data["cases"]}),
                case_keys=sorted({f"{c['task']}|{c['case_id']}" for c in data["cases"]}),
                image_hashes=sorted({c["npz_sha256"] for c in data["cases"] if c.get("npz_sha256")}))


def check_test_identity(checkpoint, data):
    current = identity_snapshot(data)
    for old in checkpoint["split_identities"]:
        for field in ("experts", "identity", "protocol", "embedding_dim"):
            if current[field] != old[field]:
                raise ValueError(f"Test protocol mismatch: {field}")
        for field in ("patients", "case_keys", "image_hashes"):
            if set(current[field]) & set(old[field]):
                raise ValueError(f"Test/train-validation leakage: {field}")


def train(args):
    output = Path(args.out_dir)
    if output.exists():
        raise ValueError("Training output exists; choose a new experiment directory")
    training, validation = load_bundle(args.train, "train"), load_bundle(args.validation, "validation")
    check_separation(training, validation)
    if {case["task"] for case in training["cases"]} != {case["task"] for case in validation["cases"]}:
        raise ValueError("Training and validation tasks differ")
    experts = training["experts"]
    costs = args.expert_costs or [0.] * len(experts)
    if len(costs) != len(experts) or any(cost < 0 for cost in costs):
        raise ValueError("Provide one nonnegative cost per expert, in bundle order")
    if args.mode == "arbitration" and args.cost_weight:
        raise ValueError("Arbitration already uses all expert outputs; selected-expert call savings cannot be rewarded")
    settings = dict(mode=args.mode, race_aware=args.race_aware)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use a GPU allocation or --device cpu")
    features, predictions, truth, scaler = tensors(training, settings, device)
    val_features, val_predictions, _, _ = tensors(validation, settings, device, scaler)
    router = RoutingAgent(features.shape[1], args.hidden_dim, len(experts)).to(device)
    reference = copy.deepcopy(router).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate)
    pools, support = fairness_pools(training, args.minimum_positive)
    if args.fairness_weight > 0 and not pools:
        raise ValueError("No disease has two supported positive racial groups; use more training data or fairness-weight 0 for an unconstrained baseline")
    rewards = utilities(predictions, truth, args.tp_bonus, args.fn_penalty,
                        torch.tensor(costs, device=device), args.cost_weight).detach()
    weights = torch.tensor(class_weights(training), device=device)
    rng = np.random.default_rng(args.seed)
    sampling = torch.Generator().manual_seed(args.seed)
    mappings = baseline_mappings(validation, args.minimum_positive)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(algorithm=args.algorithm, **settings, experts=experts, expert_costs=costs,
                  epochs=args.epochs, group_size=args.group_size, inner_updates=args.inner_updates,
                  epsilon_clip=args.epsilon_clip, reference_kl_weight=args.kl_weight,
                  entropy_weight=args.entropy_weight, fairness_weight=args.fairness_weight,
                  fnr_gap_threshold=args.fnr_gap_threshold, minimum_positive=args.minimum_positive,
                  fair_samples=args.fair_samples, tp_bonus=args.tp_bonus, fn_penalty=args.fn_penalty,
                  cost_weight=args.cost_weight, batch_size=args.batch_size, seed=args.seed,
                  learning_rate=args.learning_rate, hidden_dim=args.hidden_dim,
                  class_balance="equal task x label weight applied outside group-relative normalization",
                  fair_training_support=support, fair_training_tasks=list(pools),
                  training_bundle_sha256=file_hash(args.train), validation_bundle_sha256=file_hash(args.validation),
                  script_sha256=file_hash(__file__), data_code_sha256=file_hash(Path(__file__).with_name("fairroute_data.py")),
                  metrics_code_sha256=file_hash(Path(__file__).with_name("audit_expert_routing.py")),
                  torch=str(torch.__version__), numpy=np.__version__,
                  limitations="Small categorical-policy GRPO with a separate expected-FNR hinge loss; not GRPO fine-tuning of an expert LLM. No held-out clinical/fairness guarantee.")
    atomic_json(output / "training_config.json", config)
    history, best = [], -float("inf")
    for epoch in range(args.epochs):
        losses, policy_losses, fair_losses, kls, entropies, zeros = [], [], [], [], [], []
        order = rng.permutation(len(training["cases"]))
        router.train()
        for offset in range(0, len(order), args.batch_size):
            indices = torch.as_tensor(order[offset:offset + args.batch_size], device=device)
            batch = features[indices]
            with torch.no_grad():
                old_log = router(batch).log_softmax(dim=-1)
                ref_log = reference(batch).log_softmax(dim=-1)
                actions = torch.multinomial(old_log.exp().cpu(), args.group_size, replacement=True,
                                            generator=sampling).to(device)
                old_action_log = old_log.gather(1, actions).detach()
                advantages = group_advantages(rewards[indices].gather(1, actions))
                zeros.append(float((advantages.abs().sum(dim=1) == 0).float().mean()))
            # Match optimizer-step budgets for GRPO and the full-information baseline.
            for _ in range(args.inner_updates):
                logits = router(batch)
                log_probability = logits.log_softmax(dim=-1)
                probability = log_probability.exp()
                if args.algorithm == "grpo":
                    policy_loss = clipped_grpo_loss(logits, actions, old_action_log, advantages,
                                                   weights[indices], args.epsilon_clip)
                else:
                    # Full-information supervised policy baseline using frozen expert outcomes.
                    policy_loss = -((probability * rewards[indices]).sum(dim=1) * weights[indices]).mean()
                kl = (probability * (log_probability - ref_log)).sum(dim=1).mean()
                entropy = -(probability * log_probability).sum(dim=1).mean()
                fair = fairness_loss(router, features, predictions, pools, args.fnr_gap_threshold,
                                     rng, args.fair_samples) if args.fairness_weight else features.sum() * 0.
                loss = policy_loss + args.kl_weight * kl - args.entropy_weight * entropy + args.fairness_weight * fair
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss; stopping")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(router.parameters(), 1.)
                optimizer.step()
                for collection, value in ((losses, loss), (policy_losses, policy_loss), (fair_losses, fair),
                                          (kls, kl), (entropies, entropy)):
                    collection.append(float(value.detach()))
        router.eval()
        with torch.no_grad():
            actions = router(val_features).argmax(dim=-1)
            chosen = val_predictions.gather(1, actions[:, None]).squeeze(1).cpu().numpy()
        score, detail = selection_score(validation, chosen, args.minimum_positive,
                                        args.fnr_gap_threshold, args.fairness_weight)
        record = dict(epoch=epoch + 1, loss=float(np.mean(losses)), policy_loss=float(np.mean(policy_losses)),
                      fairness_loss=float(np.mean(fair_losses)), kl=float(np.mean(kls)),
                      entropy=float(np.mean(entropies)), zero_advantage_case_fraction=float(np.mean(zeros)),
                      validation_score=score, **detail)
        history.append(record)
        atomic_json(output / "history.json", history)
        if score > best:
            best = score
            checkpoint = dict(state_dict=router.state_dict(), reference_state_dict=reference.state_dict(),
                              settings=settings, scaler=scaler,
                              network=dict(input_dim=features.shape[1], hidden_dim=args.hidden_dim, num_experts=len(experts)),
                              config=config, best_epoch=epoch + 1, validation_score=score,
                              baseline_mappings=mappings,
                              split_identities=[identity_snapshot(training), identity_snapshot(validation)])
            temporary = output / "best_router.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best_router.pt")
        print(json.dumps(record), flush=True)
    print(f"Saved validation-selected router: {output / 'best_router.pt'}", flush=True)


def evaluate(args):
    output = Path(args.out_dir)
    if output.exists():
        raise ValueError("Evaluation output exists; choose a new directory")
    data = load_bundle(args.data, "test")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    check_test_identity(checkpoint, data)
    verification = getattr(args, "verification", "none")
    if verification == "disagreement" and checkpoint["settings"]["mode"] != "arbitration":
        raise ValueError("Disagreement verification inspects all experts and requires arbitration mode")
    router = RoutingAgent(**checkpoint["network"]).to(args.device)
    router.load_state_dict(checkpoint["state_dict"])
    router.eval()
    features, expert_predictions, _, _ = tensors(data, checkpoint["settings"], args.device, checkpoint["scaler"])
    with torch.no_grad():
        probabilities = router(features).softmax(dim=-1)
        actions = probabilities.argmax(dim=-1)
        selected = expert_predictions.gather(1, actions[:, None]).squeeze(1).cpu().numpy().astype(int)
        selected_probability = probabilities.gather(1, actions[:, None]).squeeze(1).cpu().numpy()
    action_list = actions.cpu().tolist()
    experts = data["experts"]
    method = checkpoint["config"]["algorithm"] + ("_fair" if checkpoint["config"]["fairness_weight"] else "_unconstrained")
    predictions = {method: selected}
    matrix = expert_predictions.cpu().numpy().astype(int)
    defer = (matrix != matrix[:, :1]).any(axis=1) if verification == "disagreement" else np.zeros(len(matrix), dtype=bool)
    predictions.update({name: matrix[:, e] for e, name in enumerate(experts)})
    if len(experts) % 2:
        predictions["majority_vote"] = (matrix.sum(axis=1) > len(experts) // 2).astype(int)
    mappings = checkpoint["baseline_mappings"]
    for name, mapping in (("validation_global_expert", mappings["global_expert"]),
                          ("validation_race_expert", mappings["race_expert"]),
                          ("validation_shrunken_reliability", mappings["shrunken_reliability"])):
        choices = []
        for case in data["cases"]:
            task = case["task"]
            if task not in mappings["global_expert"]:
                raise ValueError("Test task absent from validation")
            key = task if name == "validation_global_expert" else f"{task}|{case['race']}"
            choices.append(mapping.get(key, mappings["global_expert"][task]))
        predictions[name] = matrix[np.arange(len(matrix)), choices]
    overall, groups = metric_tables(data, predictions)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "metrics.csv", overall)
    write_csv(output / "race_metrics.csv", groups)
    write_csv(output / "routing_decisions.csv", [dict(task=case["task"], case_id=case["case_id"],
              patient_id=case["patient_id"], race=case["race"], y_true=case["y_true"], y_pred=int(prediction),
              selected_expert=experts[action], selected_expert_probability=float(probability))
              for case, action, prediction, probability in zip(data["cases"], action_list, selected, selected_probability)])
    if verification == "disagreement":
        selective = []
        for task in TASKS:
            population = np.asarray([c["task"] == task for c in data["cases"]])
            if not population.any():
                continue
            races = np.asarray([c["race"] for c in data["cases"]])
            for group in ["ALL", *sorted(set(races[population]) - {"unknown"})]:
                mask = population if group == "ALL" else population & (races == group)
                accepted = mask & ~defer
                labels = np.asarray([binary(c["y_true"]) for c in data["cases"]])
                selective.append(dict(task=task, race=group, total_n=int(mask.sum()),
                                      accepted_n=int(accepted.sum()), deferred_n=int((mask & defer).sum()),
                                      coverage=float(accepted.sum() / mask.sum()),
                                      **scores(labels[accepted], selected[accepted])))
        write_csv(output / "selective_metrics.csv", selective)
        write_csv(output / "verification_decisions.csv", [dict(task=c["task"], case_id=c["case_id"],
                  escalation_recommended=bool(flag), reason="expert_disagreement" if flag else "unanimous")
                  for c, flag in zip(data["cases"], defer)])
    atomic_json(output / "evaluation_config.json", dict(
        checkpoint_sha256=file_hash(args.checkpoint), test_bundle_sha256=file_hash(args.data),
        training_config=checkpoint["config"], best_epoch=checkpoint["best_epoch"], verification=verification,
        note="Expert-selection probability is not disease probability. Test labels used only for evaluation. All caches are frozen; arbitration does not reduce expert-call cost."))
    for row in overall:
        print(json.dumps(row), flush=True)
    print(f"Saved held-out evaluation: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    prepare.add_argument("--split", choices=("train", "validation", "test"), required=True)
    prepare.add_argument("--out-file", type=Path, required=True)
    prepare.add_argument("--patient-map", type=Path)
    prepare.add_argument("--confirm-one-case-per-patient", action="store_true")
    prepare.add_argument("--embeddings", type=Path, help="Optional real task/case-keyed embeddings JSON; no synthetic defaults")
    trainer = commands.add_parser("train")
    trainer.add_argument("--train", type=Path, required=True)
    trainer.add_argument("--validation", type=Path, required=True)
    trainer.add_argument("--out-dir", type=Path, required=True)
    trainer.add_argument("--algorithm", choices=("grpo", "supervised"), default="grpo")
    trainer.add_argument("--mode", choices=("arbitration", "prequery"), default="arbitration")
    trainer.add_argument("--race-aware", action="store_true")
    trainer.add_argument("--epochs", type=int, default=10)
    trainer.add_argument("--batch-size", type=int, default=64)
    trainer.add_argument("--hidden-dim", type=int, default=64)
    trainer.add_argument("--group-size", type=int, default=8)
    trainer.add_argument("--inner-updates", type=int, default=4)
    trainer.add_argument("--learning-rate", type=float, default=.001)
    trainer.add_argument("--epsilon-clip", type=float, default=.2)
    trainer.add_argument("--kl-weight", type=float, default=.02)
    trainer.add_argument("--entropy-weight", type=float, default=.01)
    trainer.add_argument("--fairness-weight", type=float, default=.5)
    trainer.add_argument("--fnr-gap-threshold", type=float, default=.15)
    trainer.add_argument("--minimum-positive", type=int, default=20)
    trainer.add_argument("--fair-samples", type=int, default=32)
    trainer.add_argument("--tp-bonus", type=float, default=.3)
    trainer.add_argument("--fn-penalty", type=float, default=.5)
    trainer.add_argument("--cost-weight", type=float, default=0.)
    trainer.add_argument("--expert-costs", type=float, nargs="+")
    trainer.add_argument("--seed", type=int, default=2026)
    trainer.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("--checkpoint", type=Path, required=True)
    evaluator.add_argument("--data", type=Path, required=True)
    evaluator.add_argument("--out-dir", type=Path, required=True)
    evaluator.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluator.add_argument("--verification", choices=("none", "disagreement"), default="none",
                           help="Optional deterministic disagreement/defer check; not a clinical or LLM verifier")
    args = parser.parse_args()
    if args.command == "prepare":
        data = prepare_bundle(args.run_dirs, args.split, args.out_file, args.patient_map,
                              args.confirm_one_case_per_patient, args.embeddings)
        print(f"Saved {args.split} bundle: {len(data['cases'])} cases, experts={data['experts']}")
    elif args.command == "train":
        if (min(args.epochs, args.batch_size, args.hidden_dim, args.inner_updates, args.minimum_positive, args.fair_samples) < 1
                or args.group_size < 2 or not 0 < args.epsilon_clip < 1 or args.learning_rate <= 0
                or not 0 <= args.fnr_gap_threshold <= 1
                or min(args.kl_weight, args.entropy_weight, args.fairness_weight, args.cost_weight, args.tp_bonus, args.fn_penalty) < 0):
            parser.error("Invalid GRPO/training settings")
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
