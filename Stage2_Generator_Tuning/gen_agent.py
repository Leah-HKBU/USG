"""
Stage 2: GeneratorAgent — inference only.

The Generator G_theta is fine-tuned externally on D_syn from Stage 1.
This file provides:
  - GeneratorAgent: calls the fine-tuned model via vLLM at port 8002
  - run_inference: runs G_theta on the full test set


"""

import os
import sys
import json
import pickle
import time
import logging
import re
from tqdm import tqdm
from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMIT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(SUBMIT_DIR, 'data')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# Fine-tuned Generator served via vLLM
gen_client = OpenAI(api_key="0", base_url="http://localhost:8002/v1")

# ── NRMS Scorer ──────────────────────────────────────────────────────────────
_nrms_scorer = None
try:
    sys.path.insert(0, os.path.join(SUBMIT_DIR, 'Simulator_Construction', 'User'))
    from nrms_scorer import NRMSScorer
    _nrms_scorer = NRMSScorer.get_instance()
    log.info("[NRMS] Scorer loaded successfully.")
except Exception as e:
    log.info(f"[NRMS] Not available (S=0.5 fallback): {e}")


def get_nrms_score(click_str_ids, candidate_str_id):
    if _nrms_scorer is None:
        return 0.5
    return _nrms_scorer.get_score(click_str_ids, candidate_str_id)


# ── Must match SYSTEM in gen_personalized_headline.py ────────────────────────
SYSTEM = "You are a Personalized Headline Writer balancing editorial standards and user preferences."


def build_instruction(article_body, u_prof, u_hist):
    """Construct instruction X = (c, u_prof, u_hist).

    Must be identical to build_output_record() in gen_personalized_headline.py.
    """
    return (
        "Task:\n"
        "Generate a headline that simultaneously satisfies two objectives:\n"
        "1. Editorial Quality: adherence to the Editorial Standards below\n"
        "2. User Preference: aligned with the user's topic interests and reading style\n\n"
        "Generation Constraints:\n"
        "1. Extractive: use vocabulary from the article body only (no hallucinations)\n"
        "2. Concise: under 20 words\n"
        "3. User-written style: avoid journalistic jargon\n\n"
        "Editorial Standards:\n"
        "1. Factual Consistency: faithfully convey core article information without hallucination\n"
        "2. No Misleading/Clickbait: avoid exaggerated or sensationalist language that distorts content\n"
        "3. Stylistic Appropriateness: concise and engaging, not a generic narrative sentence\n"
        "4. Entity Accuracy: main subject/person must be correctly identified\n\n"
        "Memory Module:\n"
        f"[User Long-term Profile]: {u_prof}\n"
        f"[User Short-term Click History]: {u_hist}\n\n"
        f"Article Body: {article_body}\n\n"
        "Now reason step by step:\n"
        "Step 1 -- Fact Extraction: identify the most important event, outcome, and entity from the article body\n"
        "Step 2 -- Preference Inference: what topics and styles does this user prefer based on profile and history?\n"
        "Step 3 -- Generation: produce a headline that maximizes editorial validity and user preference\n\n"
        "Output JSON:\n"
        "{\n"
        '    "reasoning": "(1) key facts extracted, (2) inferred user preferences, (3) brief explanation of the generation process",\n'
        '    "personalized_headline": "the generated personalized headline (max 20 words)"\n'
        "}"
    )


def parse_output(text):
    """Extract headline from model output.

    Training output format: {rationale}\\n\\nHeadline: {h*}
    Also handles JSON output with "personalized_headline" key.
    Returns (rationale, headline).
    """
    text = text.strip()
    # Try JSON with "personalized_headline" key
    try:
        obj = json.loads(text)
        headline  = obj.get("personalized_headline", "").strip()
        rationale = obj.get("reasoning", "")
        if headline:
            return rationale, headline
    except (json.JSONDecodeError, AttributeError):
        pass
    # Try markdown JSON block
    m = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            headline  = obj.get("personalized_headline", "").strip()
            rationale = obj.get("reasoning", "")
            if headline:
                return rationale, headline
        except (json.JSONDecodeError, AttributeError):
            pass
    # Find "Headline:" marker (case-insensitive, legacy fallback)
    m = re.search(r'(?i)headline\s*:\s*(.+)', text)
    if m:
        headline  = m.group(1).strip()
        rationale = text[:m.start()].strip()
        return rationale, headline
    # Last resort: last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return text, lines[-1] if lines else ""


# ── GeneratorAgent ────────────────────────────────────────────────────────────

class GeneratorAgent:
    """End-to-end personalized headline generator (G_theta inference).

    Given (c, u_prof, u_hist), calls the fine-tuned model and returns
    (reasoning, headline) providing interpretability alongside generation.
    """

    def __init__(self, article_body, u_prof, u_hist):
        self.article_body = article_body
        self.u_prof       = u_prof
        self.u_hist       = u_hist

    def generate(self):
        """Return (reasoning, headline). Falls back to ('None', 'None') on failure."""
        instruction = build_instruction(self.article_body, self.u_prof, self.u_hist)
        for attempt in range(5):
            try:
                result = gen_client.chat.completions.create(
                    model="qwen3-8b",
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user",   "content": instruction},
                    ]
                )
                raw = result.choices[0].message.content
                reasoning, headline = parse_output(raw)
                if headline:
                    return reasoning, headline
            except Exception as e:
                log.info(f"  [GeneratorAgent] attempt {attempt+1} failed: {e}")
                time.sleep(30)
        return "None", "None"


# ── Helper: build u_hist and u_prof ──────────────────────────────────────────

def build_u_hist_str(click_news):
    parts = []
    for i, item in enumerate(click_news):
        cat  = item[0] if len(item) > 0 else "unknown"
        sub  = item[1] if len(item) > 1 else "unknown"
        head = item[2] if len(item) > 2 else ""
        parts.append(f'news_{i}: headline: {head}, category: ({cat},{sub})')
    return '; '.join(parts)


def synthesize_user_profile(u_hist_str):
    """Synthesize u_prof from click history (same as Stage 1)."""
    prompt = f"""Given a user's news click history (oldest to newest), synthesize a concise user profile:
1. Topic Interests: What categories and topics does this user engage with?
2. Reading Style: What headline styles do they prefer (length, tone, formality)?
3. Persona: A 1-2 sentence summary.

User Click History:
{u_hist_str}

Output JSON:
{{
    "topic_interests": "...",
    "style_preferences": "...",
    "persona_summary": "..."
}}"""
    for attempt in range(3):
        try:
            resp = gen_client.chat.completions.create(
                model="qwen3-8b",
                messages=[
                    {"role": "system", "content": "You are a user behavior analyst."},
                    {"role": "user",   "content": prompt}
                ]
            )
            raw = resp.choices[0].message.content.strip()
            try:
                content = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', raw)
                if m:
                    content = json.loads(m.group(1).strip())
                else:
                    s, e2 = raw.find('{'), raw.rfind('}')
                    content = json.loads(raw[s:e2+1])
            return (f"Topic Interests: {content.get('topic_interests', '')}\n"
                    f"Style Preferences: {content.get('style_preferences', '')}\n"
                    f"Persona: {content.get('persona_summary', '')}")
        except Exception as e:
            log.info(f"  [u_prof] attempt {attempt+1} failed: {e}")
            time.sleep(15)
    return u_hist_str[:200]


# ── run_inference ─────────────────────────────────────────────────────────────

def run_inference(output_path=None):
    """Run G_theta on the full test set.

    TestSamples row : [userindex, news_id, label_headline]
    TestUsers       : TestUsers[userindex] = list of news_id strings
    """
    if output_path is None:
        output_path = os.path.join(DATA_DIR, 'pred_generator.txt')

    with open(os.path.join(DATA_DIR, 'news.pkl'), 'rb') as f:
        news = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'TestUsers.pkl'), 'rb') as f:
        TestUsers = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'TestSamples.pkl'), 'rb') as f:
        TestSamples = pickle.load(f)

    log.info(f"[Inference] TestSamples={len(TestSamples)}, output={output_path}")

    # Resume
    done_set = set()
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_set.add((rec['userindex'], rec['new_id']))
                except Exception:
                    continue
        log.info(f"[Resume] {len(done_set)} already done.")
    except FileNotFoundError:
        log.info("[Resume] Starting fresh.")

    pbar = tqdm(enumerate(TestSamples), total=len(TestSamples),
                desc="Generator-Inference", unit="sample", dynamic_ncols=True)

    for i, row in pbar:
        userindex = row[0]
        news_id   = row[1]       # direct string ID
        label     = row[2]       # ground-truth headline

        if (userindex, news_id) in done_set:
            continue

        if news_id not in news or userindex not in TestUsers:
            continue

        news_item        = news[news_id]
        original_headline = news_item[2] if len(news_item) > 2 else ""
        article_body     = ' '.join(news_item[3].split()[:200]) if len(news_item) > 3 else ""

        # Build user context — identical pattern to test_multiagent_editor_orig.py
        click_ids  = TestUsers[userindex][-200:]
        click_news = [news[nid] for nid in click_ids if nid in news]
        if not click_news:
            continue

        u_hist_str = build_u_hist_str(click_news[-20:]) if len(click_news) > 20 else build_u_hist_str(click_news)
        u_prof     = synthesize_user_profile(build_u_hist_str(click_news))

        try:
            reasoning, headline = GeneratorAgent(article_body, u_prof, u_hist_str).generate()
        except Exception as e:
            log.info(f"  [ERROR] user={userindex}, news={news_id}: {e}")
            continue

        record = {
            "userindex":    userindex,
            "new_id":       news_id,
            "pred_output":  headline,
            "ori_headline": original_headline,
            "label":        label,
            "reasoning":    reasoning,
        }

        with open(output_path, 'a', encoding='utf-8', buffering=1) as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        done_set.add((userindex, news_id))

        pbar.set_postfix(done=len(done_set))
        log.info(f"  [{i+1}] pred='{headline[:60]}'")

    log.info(f"[Done] total={len(done_set)}, output={output_path}")


if __name__ == "__main__":
    run_inference()
