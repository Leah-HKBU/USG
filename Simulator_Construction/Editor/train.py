import json
from tqdm import tqdm
import random


# Preprocess the data for model training, then train the model with LLaMA Factory.
with open("data/headline.txt","r") as f, \
    open("../prompts.json", "r", encoding="utf-8") as f_prompts, \
    open("data/sft_edit.json","w") as sft_us:
    prompts = json.load(f_prompts)
    editor_prompts = prompts["Editor"]
    result = []
    for line in tqdm(f):
        line = json.loads(line)
        new_body = line["original body"]
        new_head = line["original headline"]
        analysis = line["analysis"]
        j += 1
        if not isinstance(analysis,list):
            continue
        for i in range(len(analysis)):
            cur_exp = analysis[i]
            if i == 0 and cur_exp["Decision"] != "yes":
                print(new_head)
                print(cur_exp)
                continue
            if i == 0 and cur_exp["Headline"] != new_head:
                print(new_head)
                print(cur_exp)
                continue
            cur_head = cur_exp["Headline"]
            del cur_exp["Headline"]
            cur_exp = json.dumps(cur_exp,ensure_ascii=False)
            instruction = editor_prompts["train_prompt"]["user"]
            instruction = instruction.format(new_body=new_body,new_head=cur_head)
            system = editor_prompts["train_prompt"]["system"]
            sample_json = {"instruction": instruction, "output": cur_exp, "system": system}
            result.append(sample_json)
    random.shuffle(result)
    json.dump(result,sft_us,ensure_ascii=False,indent=2)
