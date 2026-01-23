from openai import OpenAI
import json
import time
user_client = OpenAI(api_key="0",base_url="http://localhost:8000/v1")
edit_client = OpenAI(api_key="0",base_url="http://localhost:8001/v1")
gen_client = OpenAI(api_key=gen_api_key, base_url="")


class UserAgent:
    def __init__(self, clk_news_str,candi_new_str):
        self.clk_news_str = clk_news_str
        self.candi_new_str = candi_new_str
    def get_prompt(self):
        system = system = "You are a regular user of a news website."
        instruction = f"""**Your Profile:**
You have a specific click history provided below. This history defines your interests (Topics) and your preferred reading style (Length, Tone, Vocabulary).

Your goal is to act as this user and evaluate the candidate headline, focusing on two dimensions:
1. **Topic Relevance Check:** Is the news content interesting to you?
2. **Style Alignment Check:** Does the headline match the your preferred reading style (e.g., Concise vs. Verbose, Casual vs. Formal, Extractive vs. Abstractive)?

Task: Output a JSON object with:
- "Analysis": A step-by-step reasoning. First check Topic, then check Style (length/tone).
- "Critique": Direct feedback to headline writer explaining why you liked or disliked it (e.g., \"It's too long,\" \"I don't follow this sport,\" \"Perfect length\")
- "Decision": "[click]" or "[not click]".

Input:
- Your Click History (oldest → newest): {self.clk_news_str}
- New Candidate Headline: {self.candi_new_str}

Output JSON:
{{
    "Analysis": "...",
    "Critique": "...",
    "Decision": "..." 
}}"""
        return system, instruction
    def output(self):
        for attempt in range(5):
            try:
                system, instruction = self.get_prompt()
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": instruction},
                ]
                result = user_client.chat.completions.create(messages=messages, model="Qwen/Qwen3-8B")
                pred_user_sim = result.choices[0].message.content
                content = json.loads(pred_user_sim)
                pred_analysis = content["Analysis"]
                pred_user_decision = content["Decision"]
                if pred_user_decision == "click":
                    prob_clk = 1.0
                elif pred_user_decision == "not click":
                    prob_clk = 0.0
                return prob_clk, pred_user_decision, pred_analysis
            except:
                time.sleep(30)
                print(f"Attempt {attempt + 1} timed out. Retrying...")
        return "None", "None", "None"

class EditorAgent:
    def __init__(self, new_body, new_head):
        self.new_body = new_body
        self.new_head = new_head
    def get_prompt(self):
        system = "You are a Senior News Editor responsible for auditing headline quality."
        instruction = f"""**Task:** Verify the headline against these **Quality Standards**:
1. **Factual Consistency:** Matches the event/outcome in the Article Body.
2. **Strict Extraction:** Content words must be from Article Body (exact forms).
3. **Entity Check:** Main subject/person must be correct.
4. **Conciseness:** Simple and under 20 words**.
5. **Style Preference:** Casual, Direct, and Practical (like a user's note). Avoid overly formal journalistic jargon.

**Input:**
- Article Body: {self.new_body}
- Headline: {self.new_head}

**Output:**
Provide a JSON response with:
- "Analysis": Step-by-step verification against the standards above.
- "Critique": 
    - If it meets **ALL** standards: "None". 
    - If it violates **ANY** standard: State "Violates [Standard Name]" and explain (e.g., "Violates Conciseness: Too long").
- "Decision": "[correct]" or "[incorrect]"

**Output JSON:**"""
        return system, instruction

    def output(self):
        for attempt in range(5):
            try:
                system, instruction = self.get_prompt()
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": instruction},
                ]
                result = edit_client.chat.completions.create(messages=messages, model="Qwen/Qwen3-8B")
                pred_edit = result.choices[0].message.content
                # print(f" this is EditorAgent messages: {messages}")
                # print(f"this is EditorAgent output: {pred_edit}" )
                res = json.loads(pred_edit)
                pred_editor_decision = res["Decision"]
                pred_editor_feedback = res["Analysis"]

                return pred_editor_decision, pred_editor_feedback
            except:
                time.sleep(30)
                print(f"Attempt {attempt + 1} timed out. Retrying...")
        return "None"
class GenAgent:
    def __init__(self, clk_news_str, new_body, new_head, prior_feedback):
        self.new_body = new_body
        self.new_head = new_head
        self.clknewsstr = clk_news_str
        self.prior_feedback = prior_feedback
    def get_prompt(self):
        system = f"""You are a Personalized Headline Writer.
Your goal is to rewrite news headlines into a **User-Generated Style** (like a personal note or search query), NOT a journalistic style.

**CORE OBJECTIVE:**
Mimic the user's "Handwritten" style inferred from their history. The goal is to match how *this specific user* would summarize the news for themselves.

**STRICT CONSTRAINTS:**
1. **Extractive:** Use vocabulary strictly from the Article Body.
2. **Concise:** Keep it under 10-15 words.
3. **Factual:** Do not hallucinate details not in the text.
4. **Anti-Expert Style:** No journalistic jargon (e.g., "Amid", "Reportedly"). Use casual structure (e.g., Noun phrases, Simplicity).

**REFINEMENT STRATEGY:**
If `Prior Feedback` is provided, you must **FIX** the specific issues raised in the last round:
- **If Editor rejected (Fact/Extraction error/Too Long):** Re-read the Body and correct.
- **If User Simulator rejected (Style/Topic):** Change the wording to be more casual, punchy, or focus on a different detail the user cares about.

**Tasks:**
1.  **Analyze User Style:** Infer preferences from User Click History.
2.  **Analyze Feedback:** Look at the **LATEST** feedback in `Prior Feedback`. Identify exactly why the previous headline failed.
3.  **Refine:** Rewrite the headline to address the specific feedback while maintaining User Style.
    - *Do not repeat the exact same rejected headline.*"""

        instruction = f"""Inputs:
• User Click History (oldest → newest): {self.clknewsstr}
• Article Body: {self.new_body}
• Prior Headlines and Feedback: {self.prior_feedback}

Output (JSON):
{{"Analysis": "Briefly explain what went wrong in the last round (if any) and how this new version fixes it while matching user style.", "Headline": "The refined headline text."}}"""
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
                print(f"this is pred_gen: {pred_gen}")

                pred_gen = json.loads(pred_gen)
                pred_gen = pred_gen["Headline"]
                
                return pred_gen
            except:
                time.sleep(30)
                print(f"Attempt {attempt + 1} timed out. Retrying...")
        return "None"

