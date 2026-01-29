# USG: Personalized News Headline Generation via Multi-Agent User Simulation

Personalized news headline generation is essential for user engagement but faces challenges due to data scarcity and factual inconsistency risks. Existing methods often rely on indirect preference modeling, which limits effectiveness. To address this, we introduce **USG**, a **Multi-Agent User Simulator** framework that generates high-quality personalized headlines. It leverages three interactive agents:
- **User Agent**: Provides subjective feedback to guide personalization.
- **Editor Agent**: Ensures factual correctness and adherence to editorial standards.
- **Writer Agent**: Iteratively refines headlines using a Critique-and-Refine algorithm.

The resulting high-fidelity dataset is then used to train a robust generator model.

## Dataset
Please download the **PENS** dataset and place it in the `data` folder.

## Requirements
- Python 3.12.7
- Install dependencies:
```bash
pip install -r requirements.txt
```

## Pipeline

### Step 1: Simulator Construction
Initialize and train the three simulators (User, Editor, Writer) to prepare for the synthesis stage.

**Directory**: `Simulator_Construction`

```bash
cd Simulator_Construction
sh run.sh
```

### Step 2: Personalized Headline Synthesis
Synthesize the personalized headline dataset using the multi-agent simulation.

**Directory**: `Stage1_Personalized_Headline_Synthesis`

```bash
cd Stage1_Personalized_Headline_Synthesis
sh run.sh
```

### Step 3: Generator Tuning
Fine-tune the generator model using the synthesized dataset to produce personalized headlines.

**Directory**: `Stage2_Generator_Tuning`

```bash
cd Stage2_Generator_Tuning
sh run.sh
```
