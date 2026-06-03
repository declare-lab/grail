# GRAIL: Gradient-Reweighted Advantages for Reinforcement Learning with Verifiable Rewards

Official codebase for the paper **"GRAIL: Gradient-Reweighted Advantages for Reinforcement Learning with Verifiable Rewards"** (June 2026), authored by *Tej Deep Pala, Vernon Toh, and Soujanya Poria* at the **DeCLaRe Lab, Nanyang Technological University**.

## 📌 Abstract

Reinforcement learning with verifiable rewards (e.g., GRPO) is a standard paradigm to enhance mathematical and logical reasoning in Large Language Models (LLMs). However, standard GRPO broadcasts a uniform sequence-level advantage scalar to all tokens in a rollout. This uniform credit assignment dilutes the gradient signal: flawed reasoning steps, intermediate derivation errors, and filler words receive the same advantage signal as pivotal, logically sound steps.

To solve this, we introduce **Gradient-Reweighted Advantage (GRAIL)**, an intrinsic, token-wise advantage reweighting method. GRAIL uses **gradient-activation saliency** to assign larger advantage signal to tokens that are locally sensitive to the final answer.

Across five model architectures (Qwen3-4B/8B, OctoThinker-3B/8B-Short, R1-Distill-Llama-8B) evaluated on six mathematical reasoning benchmarks (MATH500, AIME24, AMC23, MinervaMATH, CollegeMATH, OlympiadBench), **GRAIL consistently outperforms standard GRPO**, yielding an average absolute accuracy boost of **3.60%** and Pass@3 improvement of **3.05%** without requiring any external process-level supervision (PRMs).

## 📁 Repository Structure

```
grail/
├── requirements.txt            # System dependencies
├── README.md                   # Project documentation
├── src/
│   ├── train/
│   │   ├── grpo.py             # Main trainer launcher
│   │   ├── grpo.sh             # Training execution bash script
│   │   ├── grail_trainer.py    # Customized GrailTrainer implementing loss logic
│   │   ├── rewards.py          # Math outcome-based reward function
│   │   ├── training_configs/   # Directory for YAML configuration files
│   │   │   ├── deepspeed_zero2.yaml
│   │   │   ├── qwen3_grpo.yaml
│   │   │   ├── qwen3_grail.yaml
│   │   │   └── qwen3_oar_g.yaml
│   │   └── utils/              # Arguments, logs, and datasets utilities
│   └── eval/
│       ├── run_eval.py          # Main benchmarking execution script
│       ├── run_eval.sh          # Serving & evaluation pipeline orchestration
│       ├── start_vllm.sh        # Serving policy models via vLLM
│       ├── eval_results_base.py # Combines results across datasets
│       ├── compute_checkpoint_stats.py  # Standalone token-level saliency stats
│       ├── token_analysis.sh   # Visualizes checkpoint saliency dynamics
│       ├── aggregate_and_plot.py  # Plotting script for positional analysis
│       ├── convert_to_excel.py # Generates unified XLSX result reports
│       ├── compile_res.sh      # Aggregates evaluation results
│       └── eval_data/          # Benchmark dataset JSONL files (AIME24, MATH, etc.)
```

---

## ⚙️ Installation & Setup

Set up a Python environment (Conda recommended) and install dependencies:

```bash
conda create -n grail python=3.11 -y
conda activate grail
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## 🚀 Training Instructions

Train model configurations using DeepSpeed ZeRO-2 and the accelerate launcher.

### Running GRAIL Training:
Modify variables (e.g. `WANDB_API_KEY`, `HF_TOKEN`) inside [grpo.sh](./src/train/grpo.sh) and launch:

```bash
cd src/train
bash grpo.sh
```

By default, training leverages the config [qwen3_grail.yaml](./src/train/training_configs/qwen3_grail.yaml) which includes the following key parameters:

```yaml
# GRAIL Parameters
use_grail: true
grail_std: 0.5                  # Spread scaling factor (\sigma_w)
grail_mean: 1.0                 # Baseline neutral weight (w_mean)
grail_w_min: 0.5                # Minimum weight bounds (w_min)
grail_w_max: 5.0                # Maximum weight bounds (w_max)
grail_leaf_source: "embeddings" # Saliency leaf source
grail_rollout_symmetry: "wrong" # Apply reweighting on: "all", "correct", or "wrong"
```

## 📊 Serving & Benchmark Evaluation

Evaluations are computed over 6 reasoning suites (`aime24`, `math`, `college_math`, `minerva_math`, `olympiadbench`, `amc23`) using vLLM for high-throughput serving.

### 1. Launch the vLLM Server
Setup your model path inside [start_vllm.sh](./src/eval/start_vllm.sh) and run:
```bash
cd src/eval
bash start_vllm.sh
```

### 2. Run Benchmarks
Run evaluations by targeting the served port (configured in [run_eval.sh](./src/eval/run_eval.sh)):
```bash
cd src/eval
bash run_eval.sh
```

### 3. Generate Reports
Compile JSONL evaluation results into a single Excel sheet using [compile_res.sh](./src/eval/compile_res.sh):
```bash
bash compile_res.sh
```

## 📈 Saliency Diagnostics & Token Analysis

To evaluate how token-level saliency weights change over training checkpoints, execute the token analysis suite:

```bash
cd src/eval
bash token_analysis.sh
```

This pipeline:
1. Runs [compute_checkpoint_stats.py](./src/eval/compute_checkpoint_stats.py) to extract token gradients and weights post-hoc.
2. Runs [aggregate_and_plot.py](./src/eval/aggregate_and_plot.py) to generate plots showing the U-shape distribution of weights across normalized reasoning spans.
