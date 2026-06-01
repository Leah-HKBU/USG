from openai import OpenAI
import json
import re
import time
import logging

log = logging.getLogger(__name__)

user_client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")


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


class UserAgent:
    """User Agent (U) - Evaluates alignment with user preferences.

    Formula: (R_U(h), r) <- LLM(M_u || h || S || rho_U)
    """
    def __init__(self, u_prof, u_hist, candidate_headline, nrms_score=0.5):
        self.u_prof = u_prof
        self.u_hist = u_hist
        self.candidate_headline = candidate_headline
        self.nrms_score = nrms_score

    def get_prompt(self):
        nrms_label = "strong" if self.nrms_score >= 0.6 else "moderate" if self.nrms_score >= 0.4 else "weak"

        system = "You are a news website user evaluating whether a headline matches your preferences."

        instruction = f"""Task:
Evaluate the candidate headline by reasoning through:
1. Topic Relevance: alignment with long-term profile and short-term click history
2. Style Alignment: match with preferred tone, formality, and style
3. Holistic Judgment: synthesize a preference score and actionable critique for the Writer to improve

Memory Module:
[Long-term Profile]: {self.u_prof}
[Short-term Click History]: {self.u_hist}

[Behavioral Signal] NRMS score: {self.nrms_score:.3f}
(0.0 = irrelevant, 1.0 = highly relevant based on large-scale interaction patterns)

Candidate Headline: {self.candidate_headline}

Now reason step by step:
Step 1 -- Topic Relevance: Does this topic align with the long-term profile and click history?
The NRMS score of {self.nrms_score:.3f} indicates {nrms_label} behavioral relevance.
Step 2 -- Style Alignment: Does the headline match the preferred tone, formality, and style?
Step 3 -- Holistic Judgment: Combine the above into a preference score from 0.0 to 1.0.
Step 4 -- Critique: Provide specific, actionable feedback that tells the Writer exactly what to improve.

Output JSON:
{{
    "analysis": "step-by-step reasoning covering topic relevance, style alignment, and holistic judgment",
    "critique": "actionable feedback for the Writer (e.g., topic aligns but tone is too formal, use casual phrasing)",
    "preference_score": float 0.0-1.0
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
                result = user_client.chat.completions.create(
                    messages=messages,
                    model="qwen3-8b"
                )
                pred = result.choices[0].message.content
                log.info(f"      [UserAgent raw] {pred[:300]!r}")
                content = extract_json(pred)

                preference_score = float(content.get("preference_score", 0.0))
                pred_analysis    = content.get("analysis", "")
                pred_critique    = content.get("critique", "")

                prob_clk = min(max(preference_score, 0.0), 1.0)
                return prob_clk, pred_analysis, pred_critique
            except Exception as e:
                log.info(f"      [UserAgent attempt {attempt+1} failed] {e}")
                time.sleep(30)
        return 0.0, "None", "None"
