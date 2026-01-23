from openai import OpenAI
import json
import time


gen_client = OpenAI(api_key=gen_api_key, base_url="")




class GenAgent:
    def __init__(self, clk_news_str, new_body, decision):
        self.new_body = new_body
        self.clknewsstr = clk_news_str
        self.decision = decision

    def get_prompt(self):

        system = """You are a Personalized Headline Writer.
Your goal is to generate a news headline from scratch that aligns with a **User-Generated Style**.

**TASK:**
Write the **"Analysis"** string (1-2 sentences, first person "I") that explains your generation process.

**LOGIC FLOW:**
1.  **Analyze User:** Infer the user's preferred style (Length, Tone, Specificity) and Topic Interests from User Click History.
2.  **Explain Generation:** Explain **HOW** you constructed the Headline to match this user.
    - Mention how you selected the **Salient Point** (Verdict/Emotion/Event) from the Body.
    - Mention how you satisfied the **CONSTRAINTS**:
        1. **Core Style:** MUST look like a **User-Generated Style** (Casual, Direct, Personal Note).
        2. **Factual Consistency:** Matches the event/outcome in the Article Body.
        3. **Strict Extraction:** Content words must be from Article Body (exact forms).
        4. **Entity Check:** Main subject/person must be correct.
        5. **Conciseness:** Simple and under 20 words.
"""

        instruction = f"""
**INPUTS:**
• **User Click History (oldest → newest):** {self.clknewsstr}
• **Article Body:** {self.new_body}
• **Personalized Headline:** {self.decision}

**OUTPUT (Analysis String Only):**
"""
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

                return pred_gen
            except:
                time.sleep(30)
                print(f"Attempt {attempt + 1} timed out. Retrying...")
        return "None"

