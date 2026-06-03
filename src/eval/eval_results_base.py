import argparse
import glob
import json
import os
import re

import numpy as np
from datasets import load_dataset
from math_verify import parse, verify
from tqdm import tqdm


def verify_ans(gold, answer):
    if answer is None:
        return False

    try:
        gold = parse("$" + str(gold) + "$")
        answer = parse("$" + str(answer) + "$")
    except Exception as e:
        return False

    try:
        output = verify(gold, answer)
        return output
    except BaseException as e:
        return False


def extract_boxed_inner(s):
    results = []
    i = 0
    while True:
        start = s.find(r"\boxed{", i)
        if start == -1:
            break
        # advance past the “\boxed{”
        j = start + len(r"\boxed{")
        depth = 1
        while j < len(s) and depth > 0:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            # everything from just after the first '{' up to j-1
            content = s[start + len(r"\boxed{") : j - 1]
            results.append(content)
            i = j
        else:
            # unbalanced braces: bail out
            break
    if len(results) >= 1:
        return results[-1]  # Return the last boxed content usually
    else:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated results (Pass@3, Acc@3, Tokens, Thinking Ratio)"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Directory with saved results"
    )
    args = parser.parse_args()

    results_dir = args.results_dir

    target_datasets = [
        "math",
        "amc23",
        "aime24",
        "college_math",
        "olympiadbench",
        "minerva_math",
    ]

    test_samples = {
        "aime24": 30,
        "amc23": 40,
        "college_math": 200,
        "math": 500,
        "minerva_math": 272,
        "olympiadbench": 200,
    }

    metric_results = {}

    print(f"Evaluating results in: {results_dir}")

    total_weighted_acc3 = 0
    total_weighted_pass3 = 0
    total_weighted_tokens = 0
    total_weighted_run_mean = 0
    total_weighted_run_std = 0
    total_samples = 0
    total_null = 0

    for dataset in target_datasets:
        # Specific file path based on user instruction
        fpath = os.path.join(results_dir, dataset, "result-0-None.jsonl")

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")

        print(f"Processing {dataset}...")

        data = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except:
                    pass

        if not data:
            print(f"No valid data found for {dataset}")
            continue

        assert (
            len(data) == test_samples[dataset]
        ), f"Number of samples in results file {dataset} ({len(data)}) does not match expected number ({test_samples[dataset]})"

        pass3_hits = 0
        acc3_accum = 0

        tokens_accum = 0
        total_gens = 0

        n_items = len(data)
        null_count = 0

        run_corrects = []

        is_minerva = dataset == "minerva_math"

        for item in tqdm(data, desc=f"Eval {dataset}"):
            gt = item.get("gt_answer")
            if is_minerva:
                gt = extract_boxed_inner(str(gt))

            gens = item.get("generated_solutions", [])

            correct_flags = []

            for i, sol_obj in enumerate(gens):
                pred = sol_obj.get("pred_answer")
                if pred is None:
                    solution = sol_obj.get("solution")
                    pred = extract_boxed_inner(str(solution))
                is_correct = verify_ans(gt, pred)
                correct_flags.append(is_correct)

                while len(run_corrects) <= i:
                    run_corrects.append([])
                run_corrects[i].append(is_correct)

                if pred is not None:
                    tokens_accum += sol_obj.get("completion_tokens", 0)

                    total_gens += 1
                else:
                    null_count += 1

            if not correct_flags:
                pass_val = 0
                acc_val = 0
            else:
                pass_val = 1 if any(correct_flags) else 0
                acc_val = sum(correct_flags) / len(correct_flags)

            pass3_hits += pass_val
            acc3_accum += acc_val

        pass3_score = pass3_hits / n_items
        acc3_score = acc3_accum / n_items

        run_accs = [np.mean(run) for run in run_corrects if len(run) > 0]
        run_acc_mean = float(np.mean(run_accs)) if run_accs else 0.0
        run_acc_std = float(np.std(run_accs)) if run_accs else 0.0

        avg_tokens = tokens_accum / total_gens if total_gens > 0 else 0

        metric_results[dataset] = {
            "samples": n_items,
            "pass@3": pass3_score,
            "acc@3": acc3_score,
            "run_acc_mean": run_acc_mean,
            "run_acc_std": run_acc_std,
            "avg_tokens": avg_tokens,
            "null_count": null_count,
        }

        total_weighted_acc3 += acc3_score * n_items
        total_weighted_pass3 += pass3_score * n_items
        total_weighted_run_mean += run_acc_mean * n_items
        total_weighted_run_std += run_acc_std * n_items

        total_weighted_tokens += avg_tokens * n_items

        total_samples += n_items
        total_null += null_count

        print(
            f"  > {dataset}: Pass@3={pass3_score:.2%}, Acc@3={acc3_score:.2%}, RunMean={run_acc_mean:.2%}, RunStd={run_acc_std:.2%}, Tokens={avg_tokens:.1f}"
        )

    # Compute Averages
    if total_samples > 0:
        metric_results["weighted_average"] = {
            "pass@3": total_weighted_pass3 / total_samples,
            "acc@3": total_weighted_acc3 / total_samples,
            "run_acc_mean": total_weighted_run_mean / total_samples,
            "run_acc_std": total_weighted_run_std / total_samples,
            "avg_tokens": total_weighted_tokens / total_samples,
            "null_count": total_null,
        }

    # Simple average over present datasets
    if metric_results:
        ds_keys = [k for k in metric_results.keys() if k != "weighted_average"]
        if ds_keys:
            avg_pass3 = sum(metric_results[k]["pass@3"] for k in ds_keys) / len(ds_keys)
            avg_acc3 = sum(metric_results[k]["acc@3"] for k in ds_keys) / len(ds_keys)
            avg_run_mean = sum(
                metric_results[k]["run_acc_mean"] for k in ds_keys
            ) / len(ds_keys)
            avg_run_std = sum(metric_results[k]["run_acc_std"] for k in ds_keys) / len(
                ds_keys
            )
            avg_tok = sum(metric_results[k]["avg_tokens"] for k in ds_keys) / len(
                ds_keys
            )
            avg_null = sum(metric_results[k]["null_count"] for k in ds_keys) / len(
                ds_keys
            )

            metric_results["average"] = {
                "pass@3": avg_pass3,
                "acc@3": avg_acc3,
                "run_acc_mean": avg_run_mean,
                "run_acc_std": avg_run_std,
                "avg_tokens": avg_tok,
                "null_count": avg_null,
            }

    # Save Results
    output_path = os.path.join(results_dir, "eval_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metric_results, f, ensure_ascii=False, indent=2)

    print(f"\nSummary saved to {output_path}")
    print(json.dumps(metric_results, indent=2))


if __name__ == "__main__":
    main()
