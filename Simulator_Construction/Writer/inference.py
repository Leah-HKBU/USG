from openai import OpenAI
import json
import re
import time
import logging

log = logging.getLogger(__name__)

gen_client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find('{')
    end   = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found in response: {text[:200]!r}")


class WriterAgent:
    """Writer Agent (W) - Generates and refines personalized headlines.

    Formula: h ~ P_W(h | c, u, M)
    """
    def __init__(self, u_prof, u_hist, article_body, memory_buffer=None, initial_headline=""):
        self.u_prof          = u_prof
        self.u_hist          = u_hist
        self.article_body    = article_body
        self.memory_buffer   = memory_buffer if memory_buffer is not None else []
        self.initial_headline = initial_headline

    def format_memory(self):
        if not self.memory_buffer:
            return "No prior interaction history."
        lines = []
        for entry in self.memory_buffer:
            k   = entry.get("iteration", "?")
            h   = entry.get("headline", "")
            i_e = entry.get("editor_analysis", "")
            i_u = entry.get("user_analysis", "")
            lines.append(f"[Round {k}] Proposed Headline: {h}")
            if i_e and i_e != "None":
                lines.append(f"  Editor Analysis (I_E): {i_e}")
            if i_u and i_u != "None":
                lines.append(f"  User Analysis (I_U): {i_u}")
        return "\n".join(lines)

    def get_prompt(self):
        memory_str = self.format_memory()

        system = "You are a Personalized Headline Writer balancing editorial standards and user preferences."

        instruction = f"""Task:
Generate a headline that simultaneously satisfies two objectives:
1. Editorial Quality: factually accurate, no misleading/clickbait, stylistically appropriate, entity-accurate
2. User Preference: aligned with the user's topic interests and reading style

Generation Constraints:
1. Extractive: use vocabulary from the article body only (no hallucinations)
2. Concise: under 20 words
3. User-written style: avoid journalistic jargon.
4. Address ALL critique instructions from the Memory Module

Memory Module (Accumulated Interaction History): {memory_str}

User Context:
[Long-term Profile]: {self.u_prof}
[Short-term Click History]: {self.u_hist}

Article Body: {self.article_body}

Now reason step by step:
Step 1 -- Fact Extraction: identify the most important event, outcome, and entity from the article body
Step 2 -- Preference Inference: what topics and styles does this user prefer based on profile and history?
Step 3 -- Critique Integration: for each entry in Memory, ensure the new headline addresses editor and user feedback
Step 4 -- Generation: produce a headline that maximizes editorial validity and user preference

Output JSON:
{{
    "reasoning": "(1) key facts extracted, (2) inferred user preferences, (3) how Memory critiques are addressed",
    "headline": "the generated personalized headline (max 20 words)"
}}"""
        return system, instruction

    def output(self):
        for attempt in range(5):
            try:
                system, instruction = self.get_prompt()
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": instruction},
                ]
                result = gen_client.chat.completions.create(
                    messages=messages,
                    model="qwen3-8b"
                )
                pred = result.choices[0].message.content
                log.info(f"      [WriterAgent raw] {pred[:300]!r}")
                pred_json = extract_json(pred)

                headline  = pred_json.get("headline", "").strip()
                reasoning = pred_json.get("reasoning", "")

                return (headline, reasoning) if headline else ("None", "None")
            except Exception as e:
                log.info(f"      [WriterAgent attempt {attempt+1} failed] {e}")
                time.sleep(30)
        return "None", "None"
