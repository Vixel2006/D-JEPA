"""Cross-seed aggregation and console reporting."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from pidjepa.models import ARM_CONFIGS, FEATURE_SETS, arm_family
from pidjepa.utils import format_mean_std

if TYPE_CHECKING:
    from pidjepa.data import Bundle


def paired_bootstrap(
    baseline: list[np.ndarray],
    treatment: list[np.ndarray],
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap over test examples for a difference in accuracy."""
    differences = np.stack([a.astype(float) - b.astype(float) for a, b in zip(baseline, treatment)])
    per_example = differences.mean(0)
    rng = np.random.RandomState(seed)
    bootstrap_means = np.array(
        [
            per_example[rng.randint(0, len(per_example), len(per_example))].mean()
            for _ in range(n_bootstrap)
        ]
    )
    return (
        per_example.mean() * 100,
        np.percentile(bootstrap_means, 2.5) * 100,
        np.percentile(bootstrap_means, 97.5) * 100,
    )


def mean_pid_components(pid_values: list[dict[str, float]]) -> dict[str, float]:
    """Average the available PID components across seeds."""
    keys = ["I1", "I2", "I12", "R", "U1", "U2", "S", "null_I12"]
    return {
        key: float(np.mean([pid[key] for pid in pid_values if key in pid]))
        for key in keys
        if any(key in pid for pid in pid_values)
    }


def print_pid_table(title: str, field_name: str, results: list[dict], rule: str) -> None:
    """Print a PID table with one row per arm/pair/subset and one column per atom."""
    print("\n" + rule + f"\n{title}\n" + rule)
    print(
        "  bits; mean over seeds.  "
        "Red=redundancy  Unq1=vision-unique  Unq2=text-unique  Syn=synergy"
    )
    keys = list(results[0][field_name].keys())
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for key in keys:
        groups[(key[0], key[1]) + tuple(key[3:])].append(key)

    for group, grouped_keys in groups.items():
        suffix = f" | subset={group[2]}" if len(group) > 2 else ""
        print(f"\n[{group[0]} | {group[1]}{suffix}]")
        header = f"  {'target':<10}{'Red':>7}{'Unq1':>7}{'Unq2':>7}{'Syn':>7}{'total':>8}"
        if field_name == "fpid":
            header += f"{'null':>7}"
        print(header)
        for key in grouped_keys:
            pid_values = [
                result[field_name][key] for result in results if key in result[field_name]
            ]
            mean = mean_pid_components(pid_values)
            total = mean["R"] + mean["U1"] + mean["U2"] + mean["S"]
            row = (
                f"  {key[2]:<10}{mean['R']:7.3f}{mean['U1']:7.3f}"
                f"{mean['U2']:7.3f}{mean['S']:7.3f}{total:8.3f}"
            )
            if field_name == "fpid":
                row += f"{mean.get('null_I12', float('nan')):7.3f}"
            print(row)


def aggregate_results(results: list[dict], bundle: Bundle) -> dict:
    """Aggregate per-seed outputs into the serializable ``report.json`` payload."""
    tasks = list(bundle.tasks)
    arms = ["raw"] + list(ARM_CONFIGS)
    summary: dict = {}
    rule = "=" * 112

    print("\n" + rule + "\n1. PROBE ACCURACY (%, mean ± std over seeds, test set)\n" + rule)
    summary["acc"] = {}
    for task in tasks:
        majority = format_mean_std([result["majority"][task] for result in results])
        print(f"\n[{task}] {bundle.task_descriptions.get(task, '')}   majority={majority}")
        print(f"{'arm':<16}{'featset':<9}{'acc':>16}")
        for arm in arms:
            for feature_set_name in FEATURE_SETS[arm_family(arm)]:
                values = [result["acc"][(arm, feature_set_name, task)] for result in results]
                summary["acc"][f"{arm}|{feature_set_name}|{task}"] = values
                print(f"{arm:<16}{feature_set_name:<9}{format_mean_std(values):>16}")

    print(
        "\n"
        + rule
        + "\n2. PAIRED DIFFERENCES (joint probes, pp, 95% paired-bootstrap CI over test examples)\n"
        + rule
    )
    comparisons = [
        ("djepa", "alignment"),
        ("djepa", "alignment_rec"),
        ("djepa", "djepa_noindep"),
        ("djepa", "djepa_norec"),
        ("alignment_rec", "alignment"),
        ("raw", "djepa"),
    ]
    summary["paired"] = {}
    print(f"{'comparison':<28}" + "".join(f"{task:>26}" for task in tasks))
    for baseline, treatment in comparisons:
        cells = []
        for task in tasks:
            difference, low, high = paired_bootstrap(
                [result["corr"][(baseline, "joint", task)] for result in results],
                [result["corr"][(treatment, "joint", task)] for result in results],
            )
            summary["paired"][f"{baseline}-{treatment}|{task}"] = (difference, low, high)
            marker = "*" if (low > 0 or high < 0) else " "
            cells.append(f"{difference:+6.2f} [{low:+6.2f},{high:+6.2f}]{marker}")
        print(f"{baseline + ' - ' + treatment:<28}" + "".join(f"{cell:>26}" for cell in cells))
    print("  (* = CI excludes 0)")

    print(
        "\n" + rule + "\n3. MODALITY USE: gap = joint - max(vision-only, text-only)  (pp)\n" + rule
    )
    summary["gap"] = {}
    print(f"{'arm':<16}" + "".join(f"{task:>20}" for task in tasks))
    for arm in arms:
        cells = []
        for task in tasks:
            gaps = [
                result["acc"][(arm, "joint", task)]
                - max(result["acc"][(arm, "v", task)], result["acc"][(arm, "t", task)])
                for result in results
            ]
            summary["gap"][f"{arm}|{task}"] = gaps
            cells.append(format_mean_std(gaps))
        print(f"{arm:<16}" + "".join(f"{cell:>20}" for cell in cells))

    print(
        "\n"
        + rule
        + "\n4. COLLAPSE DIAGNOSTICS (test): mean per-dim std / effective rank / dims\n"
        + rule
    )
    summary["diag"] = {}
    for arm in arms:
        cells = []
        block_names = sorted({key[1] for key in results[0]["diag"] if key[0] == arm})
        for block_name in block_names:
            std = np.mean([result["diag"][(arm, block_name)][0] for result in results])
            effective_rank = np.mean([result["diag"][(arm, block_name)][1] for result in results])
            summary["diag"][f"{arm}|{block_name}"] = (std, effective_rank)
            cells.append(f"{block_name}: std={std:.2f} rank={effective_rank:.1f}")
        print(f"  {arm:<16}" + "   ".join(cells))

    print(
        "\n" + rule + "\n5. FACTOR IDENTIFICATION (ridge R^2, held-out; mean over seeds)\n" + rule
    )
    summary["idmat"] = {}
    factor_names = list(bundle.factors)
    for arm in arms:
        row_names = sorted({key[1] for key in results[0]["idmat"] if key[0] == arm})
        print(f"\n[{arm}]  " + "".join(f"{factor:>12}" for factor in factor_names))
        for row_name in row_names:
            values = [
                np.nanmean([result["idmat"][(arm, row_name, factor)] for result in results])
                for factor in factor_names
            ]
            summary["idmat"][f"{arm}|{row_name}"] = dict(zip(factor_names, values))
            print(f"  {row_name:<10}" + "".join(f"{value:12.3f}" for value in values))

    if bundle.has_ground_truth:
        print("\n" + rule + "\n6. DECOMPOSITION SCORES (synthetic ground truth)\n" + rule)
        score_names = [
            "shared_captured",
            "private_retained",
            "c_to_shared",
            "u_to_private",
            "leakage",
            "DS",
        ]
        print(f"{'arm':<16}" + "".join(f"{name:>18}" for name in score_names))
        summary["scalars"] = {}
        for arm in arms:
            cells = []
            for name in score_names:
                values = [
                    result["scalars"][arm][name]
                    for result in results
                    if name in result["scalars"][arm]
                ]
                summary["scalars"][f"{arm}|{name}"] = values
                cells.append(format_mean_std(values, 3) if values else "-")
            print(f"{arm:<16}" + "".join(f"{cell:>18}" for cell in cells))

    print(
        "\n" + rule + "\n7. CROSS-MODAL ALIGNMENT (held-out; real vs shuffled-pair null)\n" + rule
    )
    summary["align"] = {}
    for key in results[0]["align"]:
        cells = []
        for metric in results[0]["align"][key]:
            values = [result["align"][key][metric] for result in results]
            summary["align"][f"{key[0]}|{key[1]}|{metric}"] = values
            cells.append(f"{metric}={np.mean(values):.3f}")
        print(f"  {key[0]:<16}{key[1]:<8} " + "  ".join(cells))

    print(
        "\n"
        + rule
        + "\n8. CAUSAL BLOCK SWAP (donor label != own label)  flip-rate% / donor-transfer%\n"
        + rule
    )
    summary["swap"] = {}
    for arm in ARM_CONFIGS:
        print(f"\n[{arm}]")
        for task in tasks:
            cells = []
            for block_name in FEATURE_SETS[arm_family(arm)]["joint"]:
                flip_rate = (
                    np.nanmean([result["swap"][(arm, task, block_name)][0] for result in results])
                    * 100
                )
                transfer_rate = (
                    np.nanmean([result["swap"][(arm, task, block_name)][1] for result in results])
                    * 100
                )
                summary["swap"][f"{arm}|{task}|{block_name}"] = (flip_rate, transfer_rate)
                cells.append(f"{block_name}:{flip_rate:5.1f}/{transfer_rate:5.1f}")
            print(f"  {task:<10}" + "   ".join(cells))

    print_pid_table(
        "9. PROBE-PID (discrete tasks; I = H(Y) - CE of nonlinear probe; MMI redundancy)",
        "ppid",
        results,
        rule,
    )
    if results[0]["fpid"]:
        print_pid_table(
            "10. FLOW-PID (continuous targets; Gaussian latent flows; MMI redundancy)",
            "fpid",
            results,
            rule,
        )
    summary["ppid"] = {
        "|".join(map(str, key)): mean_pid_components([result["ppid"][key] for result in results])
        for key in results[0]["ppid"]
    }
    summary["fpid"] = {
        "|".join(map(str, key)): mean_pid_components(
            [result["fpid"][key] for result in results if key in result["fpid"]]
        )
        for key in results[0]["fpid"]
    }

    if bundle.categories is not None:
        print(
            "\n"
            + rule
            + "\n12. PER-QUESTION-TYPE ACCURACY on the answer task (joint probes)\n"
            + rule
        )
        test_categories = bundle.categories[bundle.split["test"].numpy()]
        summary["per_cat_acc"] = {}
        for arm in arms:
            cells = []
            for category in sorted(set(test_categories)):
                mask = test_categories == category
                values = [
                    result["corr"][(arm, "joint", "answer")][mask].mean() * 100
                    for result in results
                ]
                summary["per_cat_acc"][f"{arm}|{category}"] = values
                cells.append(f"{category}={format_mean_std(values, 1)}")
            print(f"  {arm:<16}" + "   ".join(cells))
    return summary
