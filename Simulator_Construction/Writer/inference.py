from openai import OpenAI
import json
import time

gen_client = OpenAI(api_key=gen_api_key, base_url="")
with open("../prompts.json", "r", encoding="utf-8") as f_prompts:
    prompts = json.load(f_prompts)
    writer_prompts = prompts["Writer"]

class WriterAgent:
    def __init__(self, user_profile,clk_news_str, new_body, new_head, prior_feedback):
        self.new_body = new_body
        self.user_profile = user_profile
        self.clk_news_str = clk_news_str
        self.memory = prior_feedback
    def get_prompt(self):
        system = writer_prompts["train_prompt"]["system"]
        instruction = writer_prompts["train_prompt"]["user"]
        instruction = instruction.format(user_profile=self.user_profile, clk_news_str=self.clk_news_str, new_body=self.new_body, prior_feedback=self.memory)
        return system, instruction

    def output(self):
        for attempt in range(5):
            try:
                system, instruction = self.get_prompt()
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": instruction},
                ]
                result = gen_client.chat.completions.create(
            model="gpt-4.1",
            stream=False,
            messages=messages
        )
                pred_gen = result.choices[0].message.content
                pred_gen = json.loads(pred_gen)
                pred_gen = pred_gen["Headline"]
                return pred_gen
            except:
                time.sleep(30)
                print(f"Attempt {attempt + 1} timed out. Retrying...")
        return "None"

