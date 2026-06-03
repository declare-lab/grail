import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Positional Analysis
# ---------------------------------------------------------------------------
@dataclass
class PositionalRecord:
    norm_position: float
    weight: float
    raw_score: float
    is_correct: bool


def extract_positional_record_single(
    tokenizer,
    tok_ids: list[int],
    weights: list[float],
    scores: list[float],
    is_correct: bool,
):
    dummy = tokenizer.encode("a", add_special_tokens=False)
    dummy_len = len(dummy)
    boxed_ids = tokenizer.encode(r"a\boxed{", add_special_tokens=False)[dummy_len:]
    K = len(boxed_ids)

    boxed_start = None
    for i in range(len(tok_ids) - K + 1):
        if tok_ids[i : i + K] == boxed_ids:
            boxed_start = i
            break

    records = []
    if boxed_start is None or boxed_start < 2:
        return records

    R = boxed_start
    for t in range(R):
        norm_pos = t / max(R - 1, 1)
        records.append(
            PositionalRecord(
                norm_position=norm_pos,
                weight=weights[t],
                raw_score=scores[t],
                is_correct=is_correct,
            )
        )
    return records


def aggregate_positional_records(
    records: list[PositionalRecord], n_bins: int = 20, split_by_outcome: bool = True
) -> dict:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def _bin_records(recs):
        bins_w = [[] for _ in range(n_bins)]
        bins_s = [[] for _ in range(n_bins)]
        for r in recs:
            idx = min(int(np.searchsorted(bin_edges[1:], r.norm_position)), n_bins - 1)
            bins_w[idx].append(r.weight)
            bins_s[idx].append(r.raw_score)

        mean_w = np.array([np.mean(b) if b else np.nan for b in bins_w])
        std_w = np.array([np.std(b) if b else np.nan for b in bins_w])
        sem_w = np.array(
            [np.std(b) / np.sqrt(len(b)) if len(b) > 1 else np.nan for b in bins_w]
        )
        mean_s = np.array([np.mean(b) if b else np.nan for b in bins_s])
        count = np.array([len(b) for b in bins_w])
        return mean_w, std_w, sem_w, mean_s, count

    result = {"bin_centres": bin_centres.tolist()}
    mw, sw, sem, ms, cnt = _bin_records(records)
    result.update(
        {
            "mean_weight": mw.tolist(),
            "std_weight": sw.tolist(),
            "sem_weight": sem.tolist(),
            "mean_raw_score": ms.tolist(),
            "count": cnt.tolist(),
        }
    )

    if split_by_outcome:
        correct_recs = [r for r in records if r.is_correct]
        wrong_recs = [r for r in records if not r.is_correct]

        mw_c, sw_c, sem_c, ms_c, cnt_c = _bin_records(correct_recs)
        mw_w, sw_w, sem_w, ms_w, cnt_w = _bin_records(wrong_recs)

        result.update(
            {
                "mean_weight_correct": mw_c.tolist(),
                "std_weight_correct": sw_c.tolist(),
                "sem_weight_correct": sem_c.tolist(),
                "mean_weight_wrong": mw_w.tolist(),
                "std_weight_wrong": sw_w.tolist(),
                "sem_weight_wrong": sem_w.tolist(),
                "count_correct": cnt_c.tolist(),
                "count_wrong": cnt_w.tolist(),
            }
        )
    return result


# ---------------------------------------------------------------------------
# GRAIL Weight Computation
# ---------------------------------------------------------------------------
def selective_log_softmax(logits, index):
    if logits.dtype in [torch.float16, torch.bfloat16]:
        logits = logits.to(torch.float32)
    selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(
        -1
    )
    logsumexp_values = torch.logsumexp(logits, dim=-1)
    per_token_logps = selected_logits - logsumexp_values
    return per_token_logps


def find_boxed_char_span(text: str) -> tuple[int, int] | None:
    MARKER = r"\boxed{"
    marker_len = len(MARKER)
    i = 0
    while True:
        start = text.find(MARKER, i)
        if start == -1:
            return None
        j = start + marker_len
        depth = 1
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            return (start, j)
        else:
            i = start + marker_len


def get_answer_mask(
    tokenizer, completion_ids: torch.Tensor, seq_len: int, device: torch.device
) -> torch.Tensor:
    answer_mask = torch.zeros(completion_ids.shape, dtype=torch.bool, device=device)
    dummy_str = "a"
    dummy_ids = tokenizer.encode(dummy_str, add_special_tokens=False)
    dummy_len = len(dummy_ids)

    def _encode_mid(text: str) -> list[int]:
        combined = tokenizer.encode(dummy_str + text, add_special_tokens=False)
        return combined[dummy_len:]

    pad_id = tokenizer.pad_token_id
    for i in range(completion_ids.shape[0]):
        ids = completion_ids[i].tolist()
        n_leading_pads = 0
        if pad_id is not None:
            for t in ids:
                if t == pad_id:
                    n_leading_pads += 1
                else:
                    break

        text = tokenizer.decode(ids, skip_special_tokens=True)
        span = find_boxed_char_span(text)
        if span is None:
            continue

        span_start, span_end = span
        prefix_tokens = _encode_mid(text[:span_start])
        span_tokens = _encode_mid(text[span_start:span_end])

        tok_start = n_leading_pads + len(prefix_tokens)
        tok_end = tok_start + len(span_tokens)

        tok_end = min(tok_end, seq_len)
        if tok_start < tok_end:
            answer_mask[i, tok_start:tok_end] = True

    return answer_mask


def get_boxed_span_masks(
    tokenizer, completion_ids: torch.Tensor, seq_len: int, device: torch.device
):
    BOXED_MARKER = r"\boxed{"
    answer_mask = torch.zeros(completion_ids.shape, dtype=torch.bool, device=device)
    boundary_mask = torch.zeros(completion_ids.shape, dtype=torch.bool, device=device)
    content_mask = torch.zeros(completion_ids.shape, dtype=torch.bool, device=device)
    post_mask = torch.zeros(completion_ids.shape, dtype=torch.bool, device=device)

    dummy_str = "a"
    dummy_len = len(tokenizer.encode(dummy_str, add_special_tokens=False))

    def _encode_mid(text: str) -> list[int]:
        return tokenizer.encode(dummy_str + text, add_special_tokens=False)[dummy_len:]

    pad_id = tokenizer.pad_token_id
    for i in range(completion_ids.shape[0]):
        ids = completion_ids[i].tolist()
        n_leading_pads = 0
        if pad_id is not None:
            for t in ids:
                if t == pad_id:
                    n_leading_pads += 1
                else:
                    break

        text = tokenizer.decode(ids, skip_special_tokens=True)
        span = find_boxed_char_span(text)
        if span is None:
            continue

        span_start, span_end = span
        prefix_toks = _encode_mid(text[:span_start])
        span_toks = _encode_mid(text[span_start:span_end])
        tok_start = n_leading_pads + len(prefix_toks)
        tok_end = min(tok_start + len(span_toks), seq_len)

        if tok_start >= tok_end:
            continue

        marker_toks = _encode_mid(BOXED_MARKER)
        marker_tok_end = min(tok_start + len(marker_toks), tok_end)

        answer_mask[i, tok_start:tok_end] = True
        boundary_mask[i, tok_start:marker_tok_end] = True
        boundary_mask[i, tok_end - 1] = True
        content_mask[i, marker_tok_end : tok_end - 1] = True
        if tok_end < seq_len:
            post_mask[i, tok_end:] = True

    return answer_mask, boundary_mask, content_mask, post_mask


def compute_grail_weights_standalone(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
    completion_ids: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor = None,
    hyperparams: dict = None,
):
    device = model.device
    seq_len = completion_ids.shape[1]

    hp = hyperparams or {}
    temperature = hp.get("temperature", 1.0)
    grail_rollout_symmetry = hp.get("grail_rollout_symmetry", "both")
    grail_w_min = hp.get("grail_w_min", 0.5)
    grail_w_max = hp.get("grail_w_max", 5.0)
    grail_mean = hp.get("grail_mean", 1.0)
    grail_std = hp.get("grail_std", 0.5)

    with torch.enable_grad():
        base_model = model
        embed_device = base_model.get_input_embeddings().weight.device
        input_embeds = base_model.get_input_embeddings()(input_ids.to(embed_device))
        input_embeds_leaf = input_embeds.detach().requires_grad_(True)

        transformer_outputs = base_model.model(
            inputs_embeds=input_embeds_leaf,
            attention_mask=attention_mask.to(embed_device),
            use_cache=False,
            output_hidden_states=True,
        )
        last_hidden = transformer_outputs.last_hidden_state
        leaf_tensor = input_embeds_leaf

        last_hidden_completion = last_hidden[:, :-1, :]
        last_hidden_completion = last_hidden_completion[:, -logits_to_keep:, :]

        lm_weight = base_model.lm_head.weight.detach()
        lm_bias = (
            base_model.lm_head.bias.detach()
            if getattr(base_model.lm_head, "bias", None) is not None
            else None
        )

        last_hidden_completion = last_hidden_completion.to(lm_weight.device)
        logits = torch.nn.functional.linear(last_hidden_completion, lm_weight, lm_bias)
        logits = logits / temperature
        logps = selective_log_softmax(logits, completion_ids.to(logits.device))

        answer_mask = get_answer_mask(tokenizer, completion_ids, seq_len, device).to(
            logits.device
        )

        if not answer_mask.any():
            ans_loss = (logps * 0.0).sum()
        else:
            ans_loss = -logps[answer_mask].sum()

        grads = torch.autograd.grad(outputs=ans_loss, inputs=leaf_tensor)[0]

    grads_completion = grads[:, -(logits_to_keep + 1) : -1, :].detach()
    embeds_completion = leaf_tensor[:, -(logits_to_keep + 1) : -1, :].detach()

    with torch.no_grad():
        scores = (grads_completion.float() * embeds_completion.float()).norm(dim=-1)
        no_answer = ~answer_mask.any(dim=-1, keepdim=True)
        fallback = no_answer

        if advantages is not None:
            advantages = advantages.to(scores.device)
            if grail_rollout_symmetry == "correct":
                symmetry_fallback = (
                    (advantages <= 0).unsqueeze(-1)
                    if advantages.dim() == 1
                    else (advantages <= 0)
                )
                fallback = fallback | symmetry_fallback
            elif grail_rollout_symmetry == "wrong":
                symmetry_fallback = (
                    (advantages > 0).unsqueeze(-1)
                    if advantages.dim() == 1
                    else (advantages > 0)
                )
                fallback = fallback | symmetry_fallback

        mask = mask.to(scores.device)
        float_mask = mask.float()

        if grail_std:
            log_scores = torch.log(scores + 1e-8)
            valid_counts = float_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

            mean_log_scores = (log_scores * float_mask).sum(
                dim=-1, keepdim=True
            ) / valid_counts
            variance = ((log_scores - mean_log_scores) ** 2) * float_mask
            std_log_scores = torch.sqrt(
                variance.sum(dim=-1, keepdim=True) / valid_counts + 1e-8
            )
            z_scores = (log_scores - mean_log_scores) / std_log_scores

            raw_weights = grail_mean + (z_scores * grail_std)
            clipped_w = raw_weights.clamp(grail_w_min, grail_w_max)
            weights = torch.where(fallback, torch.ones_like(clipped_w), clipped_w)
            weights = weights * float_mask
        else:
            raise ValueError("grail_std must be set for standalone weight computation")

        _, boundary_mask_grail, content_mask_grail, post_mask_grail = (
            get_boxed_span_masks(tokenizer, completion_ids, seq_len, device)
        )

        boundary_mask_grail = boundary_mask_grail.to(scores.device)
        content_mask_grail = content_mask_grail.to(scores.device)
        post_mask_grail = post_mask_grail.to(scores.device)

        weights[content_mask_grail] = grail_w_max
        weights[boundary_mask_grail] = grail_mean
        post_and_valid = post_mask_grail & mask.bool()
        weights[post_and_valid] = grail_mean

    return weights.detach(), scores.detach()


# ---------------------------------------------------------------------------
# 4. Main Processing
# ---------------------------------------------------------------------------
def process_checkpoint(
    step, model_path, results_file, hyperparams, output_dir, batch_size
):
    print(f"Processing step {step} from {results_file}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    with open(results_file, "r") as f:
        data = [json.loads(line) for line in f]

    from eval_results_base import extract_boxed_inner, verify_ans

    all_positional_records = []

    # Flatten rollouts
    all_rollouts = []
    for item in data:
        question = item["question"]
        gt = item.get("gt_answer", "")
        prompt_text = (
            question
            + " Please reason step by step, and put your final answer within \\boxed{}."
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

        for sol_obj in item.get("generated_solutions", []):
            solution_text = sol_obj.get("solution", "")
            if not solution_text:
                continue

            pred = sol_obj.get("pred_answer")
            if pred is None:
                pred = extract_boxed_inner(solution_text)

            is_correct = verify_ans(gt, pred)
            reward = 1.0 if is_correct else 0.0

            completion_ids = tokenizer.encode(solution_text, add_special_tokens=False)
            all_rollouts.append(
                {"prompt_ids": prompt_ids, "comp_ids": completion_ids, "reward": reward}
            )

    # Batching processing
    for i in tqdm(
        range(0, len(all_rollouts), batch_size), desc=f"Evaluating Checkpoint {step}"
    ):
        chunk = all_rollouts[i : i + batch_size]

        batch_input_ids = []
        batch_comp_ids = []
        batch_rewards = []

        for item in chunk:
            batch_input_ids.append(item["prompt_ids"] + item["comp_ids"])
            batch_comp_ids.append(item["comp_ids"])
            batch_rewards.append(item["reward"])

        pad_token_id = tokenizer.pad_token_id

        max_input_len = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = []
        attention_masks = []
        for ids in batch_input_ids:
            pad_len = max_input_len - len(ids)
            padded_input_ids.append([pad_token_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        max_comp_len = max(len(ids) for ids in batch_comp_ids)
        padded_comp_ids = []
        masks = []
        for ids in batch_comp_ids:
            pad_len = max_comp_len - len(ids)
            padded_comp_ids.append([pad_token_id] * pad_len + ids)
            masks.append([0] * pad_len + [1] * len(ids))

        input_ids_tensor = torch.tensor(padded_input_ids, device=model.device)
        attention_mask_tensor = torch.tensor(attention_masks, device=model.device)
        comp_ids_tensor = torch.tensor(padded_comp_ids, device=model.device)
        mask_tensor = torch.tensor(masks, device=model.device)
        rewards_tensor = torch.tensor(batch_rewards, device=model.device)

        logits_to_keep = max_comp_len

        try:
            weights, scores = compute_grail_weights_standalone(
                model,
                tokenizer,
                input_ids_tensor,
                attention_mask_tensor,
                logits_to_keep,
                comp_ids_tensor,
                mask_tensor,
                advantages=rewards_tensor,
                hyperparams=hyperparams,
            )
        except Exception as e:
            print(f"Skipping a batch due to error: {e}")
            continue

        weights = weights.cpu().numpy()
        scores = scores.cpu().numpy()

        for b in range(len(chunk)):
            pad_len = max_comp_len - len(batch_comp_ids[b])
            valid_weights = weights[b, pad_len:].tolist()
            valid_scores = scores[b, pad_len:].tolist()
            valid_comp_ids = batch_comp_ids[b]
            is_correct = batch_rewards[b] > 0.5

            pos_records = extract_positional_record_single(
                tokenizer, valid_comp_ids, valid_weights, valid_scores, is_correct
            )
            all_positional_records.extend(pos_records)

    pos_summary = aggregate_positional_records(all_positional_records)

    pos_out = os.path.join(output_dir, f"checkpoint_{step}_pos_summary.json")

    with open(pos_out, "w") as f:
        json.dump(pos_summary, f, indent=2)
    print(f"Saved stats for step {step} to {output_dir}")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[50, 100, 150, 200])
    parser.add_argument("--output_dir", type=str, default="token_analysis_stats")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hyperparams = {
        "use_grail": True,
        "grail_std": 0.5,
        "grail_mean": 1.0,
        "grail_w_min": 0.5,
        "grail_w_max": 5.0,
        "grail_leaf_source": "embeddings",
        "grail_rollout_symmetry": "both",
        "temperature": 0.6,
    }

    for step in args.steps:
        model_path = os.path.join(args.checkpoints_dir, f"checkpoint-{step}")

        checkpoint_results_file = os.path.join(
            args.results_dir, f"checkpoint-{step}", "math", "result-0-None.jsonl"
        )
        fallback_file = os.path.join(args.results_dir, "math", "result-0-None.jsonl")
        if not os.path.exists(checkpoint_results_file):
            checkpoint_results_file = fallback_file

        if not os.path.exists(model_path) or not os.path.exists(
            checkpoint_results_file
        ):
            print(f"Skipping step {step} (missing model or results file)")
            continue

        process_checkpoint(
            step,
            model_path,
            checkpoint_results_file,
            hyperparams,
            args.output_dir,
            args.batch_size,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
