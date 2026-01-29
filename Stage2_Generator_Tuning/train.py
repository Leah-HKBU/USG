import json
from tqdm import tqdm
import random


# Preprocess the data for model training, then train the model with LLaMA Factory.
with open("data/reason.txt","r") as f, \
    open("../prompts.json", "r", encoding="utf-8") as f_prompts, \
    open("data/sft_gen.json","w") as sft_us:
    prompts = json.load(f_prompts)
    gen_prompts = prompts["Generator"]
    result = []
    for line in tqdm(f):
        line = json.loads(line)
        clk_news_str = line["clk_news_str"]
        new_body = line["ori_body"]
        new_head = line["candi_new_str"]
        user_profile = line["user_profile"]
        reasoning = line["reasoning"]

        new_body = line["original body"]
        new_head = line["original headline"]
        analysis = line["analysis"]
        output= {"Analysis": reasoning, "Personalized Headline": new_head}
        instruction = gen_prompts["train_prompt"]["user"]
        instruction = instruction.format(user_profile=user_profile,new_body=new_body,clk_news_str=clk_news_str)
        system = gen_prompts["train_prompt"]["system"]
        sample_json = {"instruction": instruction, "output": output, "system": system}
        result.append(sample_json)
    random.shuffle(result)
    json.dump(result,sft_us,ensure_ascii=False,indent=2)
