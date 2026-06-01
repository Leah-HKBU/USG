from openai import OpenAI
import json
import re
import time
import logging

log = logging.getLogger(__name__)

edit_client = OpenAI(api_key="0", base_url="http://localhost:8001/v1")


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


class EditorAgent:
    """Editor Agent (E) - Validates editorial standards."""

    def __init__(self, article_body, headline):
        self.article_body = article_body
        self.headline     = headline

    def get_prompt(self):
        system = "You are a Senior News Editor responsible for auditing headline quality and ensuring adherence to editorial standards."

        instruction = f"""Task:
Validate the headline against the Quality Standards below and provide constructive critique to guide the Writer's refinement.

Quality Standards:
1. Factual Consistency: the headline must faithfully convey core article information without hallucination
2. No Misleading/Clickbait: avoid exaggerated or sensationalist language that distorts content
3. Stylistic Appropriateness: concise (under 20 words) and engaging, not a generic narrative sentence
4. Entity Accuracy: main subject/person must be correctly identified

Article Body: {self.article_body}
Headline: {self.headline}

Output JSON:
{{
    "analysis": "step-by-step verification against all standards",
    "critique": "If meets ALL standards: None. If violates ANY standard: state Violates [Standard Name] and explain (e.g., Violates Factual Consistency: contains unsubstantiated claim)",
    "decision": "[correct] or [incorrect]"
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
                result = edit_client.chat.completions.create(
                    messages=messages,
                    model="qwen3-8b"
                )
                pred = result.choices[0].message.content
                log.info(f"      [EditorAgent raw] {pred[:300]!r}")
                res = extract_json(pred)

                pred_editor_decision = res.get("decision", "[incorrect]")
                pred_editor_critique = res.get("critique", "")
                pred_editor_analysis = res.get("analysis", "")

                return pred_editor_decision, pred_editor_analysis, pred_editor_critique
            except Exception as e:
                log.info(f"      [EditorAgent attempt {attempt+1} failed] {e}")
                time.sleep(30)
        return "[incorrect]", "None", "None"
