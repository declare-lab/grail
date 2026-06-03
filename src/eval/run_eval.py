import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from openai import APIConnectionError, OpenAI, RateLimitError
from tqdm import tqdm

MAX_RETRIES = 3
write_lock = threading.Lock()


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


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


def get_model_response(
    client: OpenAI,
    model_name: str,
    messages: List[Dict],
    temperature: float,
    top_p: float,
    top_k: int,
    n: int,
) -> List[Dict[str, Any]]:
    """
    Calls the vLLM server via OpenAI client.
    Returns a list of dictionaries containing 'content' and 'token_count'.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=8192,  # 4096, #
            top_p=top_p,
            extra_body={
                "top_k": top_k,
                "skip_special_tokens": False,
                "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            n=n,
            timeout=1200,
        )

        results = []
        for choice in response.choices:
            message = choice.message
            content = message.content
            token_ids = getattr(choice, "token_ids", [])
            results.append({"content": content, "token_count": len(token_ids)})
        return results

    except Exception as e:
        print(f"Error in model response: {e}")
        raise e


def process_item(
    item: Dict[str, Any],
    args,
    client: OpenAI,
    ans_key: str,
) -> Dict[str, Any]:
    question = item["problem"]

    messages = [
        {
            "role": "user",
            "content": question
            + " Please reason step by step, and put your final answer within \boxed{}.",
        }
    ]

    solutions = []
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            solutions = get_model_response(
                client,
                args.policy_model_path,
                messages,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                n=3,
            )
            break
        except (APIConnectionError, RateLimitError):
            time.sleep(2)
            attempt += 1
        except Exception as e:
            print(f"Failed to process item {item.get('problem', '')[:20]}...: {e}")
            break

    generated_solutions = []
    for sol_data in solutions:
        content = sol_data["content"]
        if content:
            generated_solutions.append(
                {
                    "solution": content,
                    "pred_answer": extract_boxed_inner(content),
                    "completion_tokens": sol_data["token_count"],
                }
            )

    return {
        "question": question,
        "generated_solutions": generated_solutions,
        "gt_answer": item.get(ans_key, ""),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Eval Pipeline with vLLM Server"
    )
    parser.add_argument(
        "--policy_model_path",
        type=str,
        required=True,
        help="Model name (e.g. path or alias on vLLM server)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=[
            "math",
            "amc23",
            "aime25",
            "aime24",
            "college_math",
            "minerva_math",
            "olympiadbench",
            "aime",
            "gpqa",
            "bbeh",
            "amc",
            "arc_c",
        ],
        help="Dataset to Evaluate on",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the results."
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API Base URL",
    )

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top P")
    parser.add_argument("--top_k", type=int, default=20, help="Top K")

    parser.add_argument(
        "--max_workers", type=int, default=32, help="Number of parallel workers"
    )
    parser.add_argument("--data_begin", type=int, default=0, help="Starting index")
    parser.add_argument("--data_end", type=int, default=None, help="Ending index")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset Paths
    DATA_PATHS = {
        "math": "./eval_data/MATH/Math-OAI.jsonl",
        "amc23": "./eval_data/AMC23/test.jsonl",
        "aime24": "./eval_data/AIME24/test.jsonl",
        "college_math": "./eval_data/College_Math/college_math_200.jsonl",
        "minerva_math": "./eval_data/Minerva-MATH/minerva-math.jsonl",
        "olympiadbench": "./eval_data/OlympiadBench/olympiadbench_200.jsonl",
    }

    if args.data == "minerva_math":
        ans_key = "solution"
    elif args.data == "olympiadbench":
        ans_key = "final_answer"
    else:
        ans_key = "answer"

    print(f"Loading dataset: {args.data}...")
    try:
        dataset = read_jsonl(DATA_PATHS[args.data])
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATA_PATHS[args.data]}")
        return

    # Slicing
    start = args.data_begin
    end = args.data_end if args.data_end is not None else len(dataset)
    dataset = dataset[start:end]

    print(f"Number of samples to process: {len(dataset)}")

    # Initialize Client
    client = OpenAI(base_url=args.api_base, api_key="EMPTY")

    # Output File
    output_file = os.path.join(
        args.output_dir, f"result-{args.data_begin}-{args.data_end}.jsonl"
    )

    # Resume capability check
    completed_questions = set()
    if os.path.exists(output_file):
        print(f"Found existing output file {output_file}. Resuming...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    completed_questions.add(d["question"])
                except:
                    pass

    items_to_process = [
        item for item in dataset if item["problem"] not in completed_questions
    ]
    print(f"Remaining items: {len(items_to_process)}")

    # Parallel Execution
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_item, item, args, client, ans_key): item
            for item in items_to_process
        }

        with open(output_file, "a", encoding="utf-8") as f:
            for future in tqdm(
                as_completed(futures), total=len(items_to_process), desc="Evaluating"
            ):
                try:
                    result = future.result()
                    with write_lock:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                except Exception as e:
                    print(f"Worker exception: {e}")

    print(f"Done! Results saved to {output_file}")


if __name__ == "__main__":
    main()
