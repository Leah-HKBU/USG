# USG: Personalized News Headline Generation via Multi-Agent User Simulation


## Dataset
Please download the following dataset and place them in the `data` folder:

- **PENS**: A News-based dataset containing personalized news headline.



## Requirements
- Python 3.12.7
- Install dependencies:
```bash
pip install -r requirements.txt
```
## Pipeline

### Step 1: Simulator Construction
In the first step, we conduct 3 simulators for step 2. 

**Directory**: `Simulator_Construction`

To conduct the simulators:
```bash
cd Simulator_Construction
sh run.sh
```

### Step 2: Personalized Headline Synthesis
This step synthesizes personalized headline dataset.

**Directory**: `Stage1_Personalized_Headline_Synthesis`

To synthesize personalized headline:
```bash
cd Stage1_Personalized_Headline_Synthesis
sh run.sh
```

### Step 3: Generator Tuning
In the final step, we fine-tune a generator for generating personalized headlines.

**Directory**: `Stage2_Generator_Tuning`

To tune generator:
```bash
cd Stage2_Generator_Tuning
sh run.sh
```

