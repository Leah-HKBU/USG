import pickle
import random
import time
import json
from openai import OpenAI



client = OpenAI(api_key=gen_api_key, base_url="")


def model_predict(user_message: str):
    system_message = f"""You are an expert User Behavior Analyst. 
"""
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

            return messages, result
        except:
            time.sleep(30)
            print(f"Attempt {attempt + 1} timed out. Retrying...")
        return "None","None"

with open('data/TrainUsers.pkl', 'rb') as f:
    TrainUsers = pickle.load(f)
with open('data/TrainSamples_2w.pkl', 'rb') as f:
    TrainSamples = pickle.load(f)
with open('data/news.pkl', 'rb') as f:
    news = pickle.load(f)



def combine_prompt(clk_news_str,candi_new, label=0):
    candi_new_str = candi_new[2]

    click_not = 'not click' if label == 0 else 'click'
    prompt = f""" I will provide a User's Click History, a Candidate Headline, and the User's Reaction ("click" or "not click").

Your goal is to explain the user's reaction logic, focusing on two dimensions:
1. **Topic Relevance:** Is the news content interesting to this user?
2. **Style Alignment:** Does the headline match the user's preferred reading style (e.g., Concise vs. Verbose, Casual vs. Formal, Extractive vs. Abstractive)?

Task: Output a JSON object with:
- "Analysis": A step-by-step reasoning. First check Topic, then check Style (length/tone).
- "Critique": Direct feedback to the headline writer. 
    * If User's Reaction is "click": Confirm it's a good match.
    * If User's Reaction is "not click": Explain specifically WHAT to fix (e.g., "Topic is good, but headline is too long (18 words). Shorten to <15 words.").
- "Decision": "[click]" or "[not click]" (Must match the User's Reaction).

Input:
- User Click History (oldest → newest): {clk_news_str}
- Candidate Headline: {candi_new_str}
- User's Reaction: {click_not} 

Output JSON:
{{
    "Analysis": "First, the topic ",
    "Critique": "The topic ",
    "Decision": "{click_not}" 
}}"""
    
    return prompt

user_id_set = set()
with open("data/reason.txt") as f:
    for line in f:
        line = json.loads(line)
        user_id = line["userindex"]
        user_id_set.add(user_id)

for i in range(len(TrainSamples)):
    row = TrainSamples[i]
    userindex = row[0]
    if userindex in user_id_set:
        continue
    pos = row[1][0]
    neg = row[1][1]
    clks = TrainUsers[userindex][0].split(" ")[-20:]

    pos_news = news[pos]
    neg_news = news[neg]
    clk_news = [news[clk] for clk in clks]
    clk_news_str = '; '.join([f'new_{i}: headline: {clk_news[i][2]}, category: ({clk_news[i][0]},{clk_news[i][1]})' for i in range(len(clk_news))])
    prompt = combine_prompt(clk_news_str,pos_news,label=1)
    messages, pos_reason = model_predict(prompt)
    with open("data/reason.txt", "a", encoding="utf-8", buffering=1) as f:
        candi_new_str = pos_news[2]
        write_json = {"userindex":userindex, "clk_news_str": clk_news_str, "candi_new_str": candi_new_str,"candi_new_id": pos,"label":"1", "analysis":pos_reason}
        write_dump = json.dumps(write_json, ensure_ascii=False)
        f.write(write_dump)
        f.write("\n")

    prompt = combine_prompt(clk_news_str,neg_news,label=0)
    messages,neg_reason = model_predict(prompt)    
    with open("data/reason.txt", "a", encoding="utf-8", buffering=1) as f:
        candi_new_str = neg_news[2]
        write_json = {"userindex":userindex, "clk_news_str": clk_news_str, "candi_new_str": candi_new_str,"candi_new_id": neg,"label":"0", "analysis":neg_reason}
        write_dump = json.dumps(write_json, ensure_ascii=False)
        f.write(write_dump)
        f.write("\n")



