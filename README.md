# USG: Personalized News Headline Generation via Multi-Agent User Simulation

Personalized news headline generation is essential for user engagement but faces challenges due to data scarcity and factual inconsistency risks. Existing methods often rely on indirect preference modeling, which limits effectiveness. To address this, we introduce **USG**, a **Multi-Agent User Simulator** framework that generates high-quality personalized headlines. It leverages three interactive agents:

- **User Agent** ($\mathcal{U}$): Evaluates user alignment using a long-term profile and short-term click history, augmented by an NRMS behavioral relevance score. Outputs a preference score and actionable critique.
- **Editor Agent** ($\mathcal{E}$): Enforces editorial standards (factual consistency, no misleading/clickbait, stylistic appropriateness, entity accuracy). Acts as a binary discriminator.
- **Writer Agent** ($\mathcal{W}$): Iteratively refines headlines via a Critique-and-Refine algorithm, integrating accumulated feedback from both agents across rounds.

The resulting high-fidelity synthetic dataset is then used to fine-tune a robust personalized headline **Generator** ($\mathcal{G}$).

---

## Dataset

Download the **PENS** dataset in the `data/` folder:

```
data/
```

---

## Requirements

- Python 3.13+
- Install dependencies:

```bash
pip install -r requirements.txt
```


The three simulator agents and the generator are served via **vLLM**. Start the servers before running any pipeline step:

| Port | Model | Role |
|------|-------|------|
| 8000 | `qwen3-8b` | User Agent + Writer Agent + profile synthesis |
| 8001 | `qwen3-8b` | Editor Agent |
| 8002 | Fine-tuned Generator | Generator inference (Stage 2) |

---

## Directory Structure

```
sub_code/
├── README.md
├── requirements.txt
├── Simulator_Construction/          # Step 1: initialize the three simulator agents
│   ├── Editor/
│   │   ├── distill_data.py          # generate contrastive distillation data via GPT-4.1
│   │   ├── train.py                 # convert distillation data → LLaMA-Factory SFT format
│   │   └── inference.py             # EditorAgent class (port 8001)
│   ├── User/
│   │   ├── inference.py             # UserAgent class (port 8000)
│   │   └── nrms_scorer.py           # NRMS behavioral relevance scorer
│   └── Writer/
│       └── inference.py             # WriterAgent class (port 8000)
├── Stage1_Personalized_Headline_Synthesis/   # Step 2: synthesize D_syn
│   ├── gen_personalized_headline.py # One-to-Many data generation (training)
│   ├── test_multiagent.py           # multi-agent inference on test set
│   └── run.sh
└── Stage2_Generator_Tuning/         # Step 3: fine-tune and evaluate Generator
    ├── gen_agent.py                 # GeneratorAgent inference on test set
    └── run.sh
```

---

## Pipeline

### Step 1: Simulator Construction

Initialize the three simulator agents (User $\mathcal{U}$, Editor $\mathcal{E}$, Writer $\mathcal{W}$).

**Directory**: `Simulator_Construction/`

#### 1a. Generate Editor distillation data

`Editor/distill_data.py` calls GPT-4.1 to synthesize contrastive training data. For each news article it produces one positive sample (original headline, decision `correct`) and three negative samples (each violating a different editorial requirement, decision `incorrect`).

```bash
python Simulator_Construction/Editor/distill_data.py
# Output: Simulator_Construction/Editor/data/headline.txt
```

#### 1b. Convert to SFT format and fine-tune

`Editor/train.py` converts `headline.txt` into LLaMA-Factory SFT format (`data/sft_edit.json`), then fine-tune the Editor model with LLaMA-Factory.

```bash
python Simulator_Construction/Editor/train.py
# Output: Simulator_Construction/Editor/data/sft_edit.json
# Then: fine-tune with LLaMA-Factory and serve at port 8001
```

The User and Writer agents use their respective `inference.py` files directly (no separate training data generation step required).

---

### Step 2: Personalized Headline Synthesis

Synthesize the personalized headline dataset $D_{syn}$ using the Critique-and-Refine multi-agent loop.

**Directory**: `Stage1_Personalized_Headline_Synthesis/`

#### 2a. Generate training data (One-to-Many)

`gen_personalized_headline.py` iterates over `TrainSamples.pkl`. For each news article, up to **3 users** (One-to-Many, $M=3$) are processed. The Critique-and-Refine loop runs up to `MAX_ITERATIONS=3` rounds per (article, user) pair. Records passing rejection sampling are saved with a synthesized rationale.

Key parameters (editable at top of file):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USERS_PER_NEWS` | 3 | users per article (One-to-Many $M$) |
| `TARGET_TOTAL` | 30000 | total (article, user) pairs to save |
| `TAU_HIGH` | 0.85 | preference score threshold for early exit |
| `TAU_LOW` | 0.3 | rejection sampling lower bound |
| `MAX_ITERATIONS` | 3 | max Critique-and-Refine rounds |

```bash
cd Stage1_Personalized_Headline_Synthesis
sh run.sh
# Equivalent to: python gen_personalized_headline.py 3
# Output: data/personalized_headline_train_m3.jsonl
#         data/personalized_headline_train_m3_detail.jsonl
```

Supports **resume**: re-running skips already-saved (article, user) pairs.

#### 2b. Evaluate on test set (multi-agent inference)

`test_multiagent.py` runs the full Editor → User → Writer pipeline on `TestSamples.pkl`.

```bash
python Stage1_Personalized_Headline_Synthesis/test_multiagent.py
# Output: Stage1_Personalized_Headline_Synthesis/data/pred_multiagent_editor_orig.txt
#         Stage1_Personalized_Headline_Synthesis/data/pred_multiagent_editor_orig_trajectory.jsonl
```

---

### Step 3: Generator Tuning

Fine-tune the generator $\mathcal{G}_\theta$ on $D_{syn}$ from Step 2, then run inference on the test set.

**Directory**: `Stage2_Generator_Tuning/`

#### 3a. Fine-tune

Fine-tune a base LLM (e.g., Qwen3-8B) on `data/personalized_headline_train_m3.jsonl` using LLaMA-Factory or equivalent SFT framework. Serve the fine-tuned model at port 8002.

#### 3b. Run inference

`gen_agent.py` loads the fine-tuned generator (port 8002) and runs it on the full test set. The prompt format is identical to the training data (Generator Prompt Template).

```bash
cd Stage2_Generator_Tuning
sh run.sh
# Equivalent to: python gen_agent.py
# Output: data/pred_generator.txt
```

Supports **resume**: re-running skips already-predicted samples.

---

## Agent Prompt Overview

| Agent | System message | Key output fields |
|-------|---------------|-------------------|
| **User** $\mathcal{U}$ | `"You are a news website user evaluating whether a headline matches your preferences."` | `analysis`, `critique`, `preference_score` (0–1) |
| **Editor** $\mathcal{E}$ | `"You are a Senior News Editor responsible for auditing headline quality."` | `analysis`, `critique`, `decision` (`[correct]`/`[incorrect]`) |
| **Writer** $\mathcal{W}$ | `"You are a Personalized Headline Writer balancing editorial standards and user preferences."` | `reasoning`, `headline` |
| **Generator** $\mathcal{G}$ | `"You are a Personalized Headline Writer balancing editorial standards and user preferences."` | `reasoning`, `personalized_headline` |


