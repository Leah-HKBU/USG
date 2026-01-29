from openai import OpenAI
import json
import time
with open("../prompts.json", "r", encoding="utf-8") as f_prompts:
    prompts = json.load(f_prompts)
    user_prompts = prompts["User"]

user_client = OpenAI(api_key="0",base_url="http://localhost:8000/v1")
class UserAgent:
    def __init__(self, user_profile, clk_news_str,candi_new_str):
        self.memory = {"long-term memory": user_profile, "short-term memory": clk_news_str}
        self.candi_new_str = candi_new_str
    def get_prompt(self):
        instruction = user_prompts["user_train_prompt"]["user"]
        instruction = instruction.format(user_profile=self.memory["long-term memory"],clk_news_str=self.memory["short-term memory"],candi_new_str=self.candi_new_str)
        system = user_prompts["user_train_prompt"]["system"]
        return system, instruction

    def output(self):
        for attempt in range(5):
            try:
                system, instruction = self.get_prompt()
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": instruction},
                ]
                result = user_client.chat.completions.create(messages=messages, model="Qwen/Qwen3-4B")
                pred_user_sim = result.choices[0].message.content
                content = json.loads(pred_user_sim)
                pred_analysis = content["Analysis"]
                prob_clk = content["Decision_prob"]
                return prob_clk, pred_analysis
            except:
                time.sleep(30)
                print(f"Attempt {attempt + 1} timed out. Retrying...")
        return "None", "None"