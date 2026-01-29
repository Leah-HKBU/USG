import json
from tqdm import tqdm
import random


# Preprocess the data for model training, then train the model with LLaMA Factory.
with open("data/reason.txt","r") as f, \
     open("../prompts.json", "r", encoding="utf-8") as f_prompts, \
    open("data/sft_user_sim.json","w") as sft_us:
    prompts = json.load(f_prompts)
    user_prompts = prompts["User"]
    result = []
    for line in tqdm(f):
        line = json.loads(line)
        user_profile = line["user_profile"]
        clk_news_str = line["clk_news_str"]
        candi_new_str = line["original_headline"]
        analysis = line["analysis"]
        if analysis == "None":
            continue
        analysis["Analysis"] = analysis["Analysis"].replace("the user's","my").replace("this user's","my").replace("the user’s","my").replace("this user’s","my").replace("the user","i").replace("this user","i")
        analysis["Critique"] = analysis["Critique"].replace("the user's","my").replace("this user's","my").replace("the user’s","my").replace("this user’s","my").replace("the user","i").replace("this user","i")
        analysis = json.dumps(analysis,ensure_ascii=False)
        instruction = user_prompts["user_train_prompt"]["user"]
        instruction = instruction.format(user_profile=user_profile,clk_news_str=clk_news_str,candi_new_str=candi_new_str)
        system = user_prompts["user_train_prompt"]["system"]
        sample_json = {"instruction": instruction, "output": analysis, "system": system}
        result.append(sample_json)
    random.shuffle(result)
    json.dump(result,sft_us,ensure_ascii=False,indent=2)