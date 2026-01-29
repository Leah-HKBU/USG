from ast import main
import pickle
import random
import time
import json
from openai import OpenAI

client = OpenAI(api_key=gen_api_key, base_url="")
class Distill_Data:
    def __init__(self, user_profile,clk_news_str,new_head,new_body,gen_prompts):
        self.user_profile = user_profile
        self.clk_news_str = clk_news_str
        self.new_head = new_head
        self.new_body = new_body
        self.gen_prompts = gen_prompts
    
    def get_prompt(self):
        system = self.gen_prompts["reasoning_prompt"]["system"]
        instruction = self.gen_prompts["reasoning_prompt"]["user"]
        instruction = instruction.format(
            user_profile=self.user_profile,
            clk_news_str=self.clk_news_str,
            headline=self.new_head,
            new_body=self.new_body
        )
            
        return system, instruction
    def gen_data(self, system_message, user_message):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
        model="gpt-4.1",
        stream=False,
        messages=messages)
                result = response.choices[0].message.content
                result = json.loads(result)
                return result
            except:
                time.sleep(30)
                print(f"Attempt {attempt + 1} timed out. Retrying...")
            return "None"
    def output(self):
        system, instruction = self.get_prompt()
        self.reasoning = self.gen_data(system, instruction)
        return self.reasoning




def main():
    with open("../Stage1_Personalized_Headline_Synthesis/data/personalized_headline_train.txt", "r", encoding="utf-8") as f, \
        open("prompts.json", "r", encoding="utf-8") as f2, \
        open("data/reason.txt", "a", encoding="utf-8", buffering=1) as f1:
        prompts = json.load(f2)
        gen_prompts = prompts["Generator"]
        for line in f:
            line = line.strip()
            line = json.loads(line)
            clk_news_str = line["clk_news_str"]
            new_body = line["ori_body"]
            new_head = line["candi_new_str"]
            user_profile = line["user_profile"]
            distill_cls = Distill_Data(user_profile,clk_news_str,new_head,new_body,gen_prompts)
            line["reasoning"] = distill_cls.output()
            f1.write(json.dumps(line, ensure_ascii=False) + "\n")
            

if __name__ == "__main__":
    main()
