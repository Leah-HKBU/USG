import pickle
import random
import time
from openai import OpenAI
import json

from openai import OpenAI

client = OpenAI(api_key=gen_api_key, base_url="")


def model_predict(user_message: str):
    system_message = f"""You are a Senior News Editor responsible for auditing headline quality."""


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
with open('data/TrainUsers.pkl', 'rb') as f:
    TrainUsers = pickle.load(f)
with open('data/TrainSamples_2w.pkl', 'rb') as f:
    TrainSamples = pickle.load(f)
with open('data/news.pkl', 'rb') as f:
    news = pickle.load(f)

def combine_prompt(content,title, label=0):
    prompt = f"""**Quality Standards:**
1. **Factual Consistency:** Matches the event/outcome in the Article Body.
2. **Strict Extraction:** Content words must be from Article Body (exact forms).
3. **Entity Check:** Main subject/person must be correct.
4. **Conciseness:** Simple and under 20 words**.
5. **Style Preference:** Casual, Direct, and Practical (like a user's note). Avoid overly formal journalistic jargon.

**Task:** 
Based on the provided Article Body and Original Headline, output a JSON list of 4 items:

1. **The Original Headline (Correct):**
   - Provide an **Analysis** verifying it meets all standards (Entity, Fact, Extraction, Conciseness, Style).
   - **Critique:** "None."
   - **Decision:** "[correct]"

2. Three Distinct Type Negative Cases**
- **Instruction:** Randomly select **3 DIFFERENT standards** from the list above to violate.
- **Headline:** Generate a headline that clearly violates the chosen standard.
- **Analysis:** Verify the headline against the body and identify the flaw.
- **Critique:** Explicitly state: "Violates [Standard Name]: [Reason]" (e.g., "Violates Strict Extraction: Hallucinated number").
- **Decision:** "[incorrect]""

**Input:**
- Article Body: {content}
- Original Headline: {title}

**Output JSON**
[
  {{
    "Headline": {title}",
    "Analysis": "Step-by-step verification...",
    "Critique": "None.",
    "Decision": "[correct]"
  }},
  {{ "Headline": "Negative Case",
    "Analysis": "Step-by-step verification...",
    "Critique": "...",
    "Decision": "[incorrect]"}},
    ...
]"""
    # prompt = f'{clk_news_str}\n{candi_new_str}\nlabel: {label}'
    return prompt

gen_news_id = set()
with open("data/headline.txt") as f:
    for line in f:
        line = json.loads(line)
        new_id = line["new_id"]
        gen_news_id.add(new_id)
for i in range(len(TrainSamples)):
    row = TrainSamples[i]
    userindex = row[0]
    pos = row[1][0]
    neg = row[1][1]
    if pos in gen_news_id:
        continue
    if neg in gen_news_id:
        continue
    gen_news_id.add(pos)
    gen_news_id.add(neg)
    clks = TrainUsers[userindex][0].split(" ")[-20:]
    pos_news = news[pos]
    neg_news = news[neg]
    clk_news = [news[clk] for clk in clks]
    content = ' '.join(pos_news[3].split(" ")[:200])
    title = pos_news[2]
    prompt = combine_prompt(content,title)
    pos_reason = model_predict(prompt)
    with open("data/headline.txt", "a", encoding="utf-8", buffering=1) as f:
        write_json = {"new_id": pos,"original body":content,"original headline":title, "analysis": pos_reason}
        write_dump = json.dumps(write_json, ensure_ascii=False)
        f.write(write_dump)
        f.write("\n")
    content = ' '.join(neg_news[3].split(" ")[:200])
    title = neg_news[2]
    prompt = combine_prompt(content,title)
    neg_reason = model_predict(prompt) 
    with open("data/headline.txt", "a", encoding="utf-8", buffering=1) as f:
        write_json = {"new_id": neg,"original body":content,"original headline":title, "analysis": neg_reason}
        write_dump = json.dumps(write_json, ensure_ascii=False)
        f.write(write_dump)
        f.write("\n")
