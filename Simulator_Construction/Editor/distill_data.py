import pickle
import time
from openai import OpenAI
import json

client = OpenAI(api_key=gen_api_key, base_url="")

SYSTEM = "You are a senior news editor responsible for auditing headline quality."

USER_TEMPLATE = """\
Headline quality requirements:
1) Factual consistency: Matches the event/outcome in the article body.
2) Strict extraction: Use only content words that appear verbatim in the article body.
3) Entity check: The main subject/person must be correct.
4) Conciseness: Simple and under 20 words.
5) No misleading/Clickbait content: Do not imply, exaggerate, or omit facts in ways that distort the article body.
6) Style preference: Casual, direct, practical. Avoid overly formal journalistic jargon.

Task:
Based on the provided article body and original headline, return a JSON array with 4 items:

1) Original headline (correct)
• Analysis: Step-by-step verification against all requirements.
• Critique: "None."
• Decision: "correct"

2) Three distinct negative cases
• Instruction: Randomly select 3 different requirements from the list above to violate (one per case).
• Headline: A headline that clearly violates the chosen requirement.
• Analysis: Verify the headline against the article body and identify the flaw.
• Critique: Explicitly state "Violates [Requirement Name]: [Reason]" (e.g., "Violates Conciseness: too long").
• Decision: "incorrect"

Input:
• Article body: {content}
• Original headline: {title}

Output only the JSON array:
[
  {{
    "Headline": "{title}",
    "Analysis": "Step-by-step verification...",
    "Critique": "None.",
    "Decision": "correct"
  }},
  {{
    "Headline": "Negative case 1",
    "Analysis": "Step-by-step verification...",
    "Critique": "Violates ...: ...",
    "Decision": "incorrect"
  }},
  {{
    "Headline": "Negative case 2",
    "Analysis": "Step-by-step verification...",
    "Critique": "Violates ...: ...",
    "Decision": "incorrect"
  }},
  {{
    "Headline": "Negative case 3",
    "Analysis": "Step-by-step verification...",
    "Critique": "Violates ...: ...",
    "Decision": "incorrect"
  }}
]"""


class Distill_Data:
    def __init__(self, content, title):
        self.content = content
        self.title   = title

    def get_prompt(self):
        system      = SYSTEM
        instruction = USER_TEMPLATE.format(content=self.content, title=self.title)
        return system, instruction

    def gen_data(self, system_message, user_message):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ]
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1",
                    stream=False,
                    messages=messages,
                )
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
        TrainSamples = pickle.load(f)
    with open('../../data/news.pkl', 'rb') as f:
        news = pickle.load(f)
    gen_news_id = set()
    with open("data/headline.txt", "a", encoding="utf-8", buffering=1) as f:
        for i in range(len(TrainSamples)):
            row    = TrainSamples[i]
            pos_neg = row[1]
            for j in range(len(pos_neg)):
                new_id = pos_neg[j]
                news_j = news[new_id]
                if new_id in gen_news_id:
                    continue
                gen_news_id.add(new_id)
                content      = ' '.join(news_j[3].split(" ")[:500])
                title        = news_j[2]
                distill_cls  = Distill_Data(content, title)
                reason       = distill_cls.output()
                write_json   = {
                    "new_id": new_id,
                    "original body": content,
                    "original headline": title,
                    "analysis": reason,
                }
                f.write(json.dumps(write_json, ensure_ascii=False))
                f.write("\n")

if __name__ == "__main__":
    main()
