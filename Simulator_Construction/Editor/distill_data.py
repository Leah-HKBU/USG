import pickle
import time
from openai import OpenAI
import json

client = OpenAI(api_key=gen_api_key, base_url="")

class Distill_Data:
    def __init__(self, content, title, editor_prompts):
        self.content = content
        self.title = title
        self.editor_prompts = editor_prompts
    
    def get_prompt(self):
        system = self.editor_prompts["contrastive_prompt"]["system"]
        instruction = self.editor_prompts["contrastive_prompt"]["user"]
        instruction = instruction.format(
            content=self.content,
            title=self.title
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
    with open('../../data/TrainUsers.pkl', 'rb') as f:
        TrainUsers = pickle.load(f)
    with open('../../data/TrainSamples_2w.pkl', 'rb') as f:
        TrainSamples = pickle.load(f)
    with open('../../data/news.pkl', 'rb') as f:
        news = pickle.load(f)
    with open("../prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    editor_prompts = prompts["Editor"]
    gen_news_id = set()
    with open("data/headline.txt", "a", encoding="utf-8", buffering=1) as f:
        for i in range(len(TrainSamples)):
            row = TrainSamples[i]
            pos_neg = row[1]
            for j in range(len(pos_neg)):
                new_id = pos_neg[j]
                news_j = news[new_id]
                if new_id in gen_news_id:
                    continue
                gen_news_id.add(new_id)
                content = ' '.join(news_j[3].split(" ")[:500])
                title = news_j[2]
                distill_cls = Distill_Data(content, title,editor_prompts)
                reason = distill_cls.output()
                write_json = {"new_id": new_id,"original body":content,"original headline":title, "analysis": reason}
                write_dump = json.dumps(write_json, ensure_ascii=False)
                f.write(write_dump)
                f.write("\n")

if __name__ == "__main__":
    main()