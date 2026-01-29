from ast import main
import pickle
import random
import time
import json
from openai import OpenAI

client = OpenAI(api_key=gen_api_key, base_url="")
class Distill_Data:
    def __init__(self, user_profile, data_area,long_clks_news_str,clk_news_str,label,news_j,user_prompts):
        self.user_profile = user_profile
        self.data_area = data_area
        self.clk_news_str = clk_news_str
        self.candi_new_str = news_j[2]
        self.long_clks_news_str = long_clks_news_str
        self.label = label
        self.user_prompts = user_prompts
    
    def get_prompt_p(self):
        system_p = self.user_prompts["user_profile_prompt"]["system"]
        instruction_p = self.user_prompts["user_profile_prompt"]["user"]
        instruction_p = instruction_p.format(
            long_clks_news_str=self.long_clks_news_str
        )
            
        return system_p, instruction_p
    def get_prompt_r(self):
        system_r = self.user_prompts["user_reasoning_prompt"]["system"]
        instruction_r = self.user_prompts["user_reasoning_prompt"]["user"]
        click_not = "yes" if self.label == 1 else "no"
        instruction_r = instruction_r.format(
            user_profile=self.user_profile,
            clk_news_str=self.clk_news_str,
            candi_new_str=self.candi_new_str,
            click_not=click_not
        )
        return system_r, instruction_r
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
        if "profile" in self.data_area:
            system_p, instruction_p = self.get_prompt_p()
            self.user_profile = self.gen_data(system_p, instruction_p)
        if "reason" in self.data_area:
            system_r, instruction_r = self.get_prompt_r()
            self.user_reasoning = self.gen_data(system_r, instruction_r)
            return self.user_profile, self.user_reasoning
        return self.user_profile




def main():
    with open('../../data/TrainUsers.pkl', 'rb') as f:
        TrainUsers = pickle.load(f)
    with open('../../data/TrainSamples_2w.pkl', 'rb') as f:
        TrainSamples = pickle.load(f)
    with open('../../data/news.pkl', 'rb') as f:
        news = pickle.load(f)
    with open("../prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    user_prompts = prompts["User"]
    user_id_set = set()

    user_profiles = {}
    with open("data/reason.txt", "a", encoding="utf-8", buffering=1) as f, \
        open("data/profile.txt", "a", encoding="utf-8", buffering=1) as f1:
        for i in range(len(TrainSamples)):
            row = TrainSamples[i]
            userindex = row[0]
            if userindex in user_id_set:
                continue
            long_clks = TrainUsers[userindex][0].split(" ")[-200:]
            clks = TrainUsers[userindex][0].split(" ")[-20:]
            clk_news = [news[clk] for clk in clks]
            clk_news_str = '; '.join([f'new_{i}: headline: {clk_news[i][2]}, category: ({clk_news[i][0]},{clk_news[i][1]})' for i in range(len(clk_news))])
            long_clks_news = [news[clk] for clk in long_clks]
            long_clks_news_str = '; '.join([f'new_{i}: headline: {long_clks_news[i][2]}, category: ({long_clks_news[i][0]},{long_clks_news[i][1]})' for i in range(len(long_clks_news))])
            
            pos_neg = row[1]
            for j in range(len(pos_neg)):
                new_id = pos_neg[j]
                news_j = news[new_id]
                if j == 0:
                    label = 1
                else:
                    label = 0
                    
                if userindex in user_profiles:
                    user_profile = user_profiles[userindex]
                    data_area = ["reason"]
                else:
                    user_profile = ""
                    data_area = ["profile", "reason"]
                Distill_cls = Distill_Data(user_profile, data_area,long_clks_news_str,clk_news_str,label,news_j,user_prompts)
                user_profile, reason = Distill_cls.output()
                write_json = {"userindex":userindex, "clk_news_str": clk_news_str,"user_profile": user_profile, "original_headline": news_j[2],"candi_new_id": new_id,"label":str(label), "analysis":reason}
                write_dump = json.dumps(write_json, ensure_ascii=False)
                f.write(write_dump)
                f.write("\n")
                if userindex not in user_profiles:
                    f1.write(json.dumps({"userindex":userindex, "user_profile": user_profile}, ensure_ascii=False) + "\n")
                    user_profiles[userindex] = user_profile


if __name__ == "__main__":
    main()
