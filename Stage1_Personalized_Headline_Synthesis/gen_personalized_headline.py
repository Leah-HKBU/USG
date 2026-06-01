"""
Step 2: Generate personalized headline fine-tuning data (One-to-Many).

Source  : TrainSamples_2w.pkl  (pos + neg samples)
Strategy: For each news ID, assign up to USERS_PER_NEWS=3 users
          (positive first, negative to supplement).

Each output record:
  system      : personalized headline generation expert
  instruction : u_prof + u_hist + article body
  output      : rationale r* + "Headline: h*"

Run from submit_code/:
    python Stage1_Personalized_Headline_Synthesis/gen_personalized_headline.py
"""

import os
import sys
import json
import pickle
import random
import time
import traceback
import logging
import re
import argparse
from collections import defaultdict
from tqdm import tqdm

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
SUBMIT_DIR      = os.path.dirname(SCRIPT_DIR)
DATA_DIR        = os.path.join(SUBMIT_DIR, 'data')
# Paths are finalised after argparse (see below); placeholders here.
OUTPUT_FILE  = None
DETAIL_FILE  = None
LOG_FILE     = os.path.join(DATA_DIR, 'gen_personalized_headline_debug.log')

# ─── Logging (identical pattern to test_multiagent_editor_orig.py) ────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stderr),
    ]
)
log = logging.getLogger(__name__)

# ─── Agent imports ────────────────────────────────────────────────────────────
import importlib.util

def _load_agent(agent_dir, class_name):
    path = os.path.join(SUBMIT_DIR, 'Simulator_Construction', agent_dir, 'inference.py')
    path = os.path.normpath(path)
    spec = importlib.util.spec_from_file_location(f"{agent_dir}_inference", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)

UserAgent   = _load_agent('User',   'UserAgent')
EditorAgent = _load_agent('Editor', 'EditorAgent')
WriterAgent = _load_agent('Writer', 'WriterAgent')

from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
# Shared with test_multiagent_editor_orig.py
TAU_HIGH       = 0.85
MAX_ITERATIONS = 3

# Gen-specific
TAU_LOW      = 0.3
TARGET_TOTAL = 30000   # total (news, user) pairs to generate


parser = argparse.ArgumentParser()
parser.add_argument('users_per_news', type=int, choices=[3],
                    help='One-to-Many M: 3 (One-to-Many)')
args = parser.parse_args()

USERS_PER_NEWS = args.users_per_news


# Output files are keyed by M to avoid overwriting across runs
OUTPUT_FILE = os.path.join(DATA_DIR, f'personalized_headline_train_m{USERS_PER_NEWS}.jsonl')
DETAIL_FILE = os.path.join(DATA_DIR, f'personalized_headline_train_m{USERS_PER_NEWS}_detail.jsonl')

random.seed(42)

gen_client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")

SYSTEM = "You are a Personalized Headline Writer balancing editorial standards and user preferences."

log.info("=" * 60)
log.info("USG Stage 1: Personalized Headline Synthesis (One-to-Many)")
log.info("=" * 60)

# ─── NRMS Scorer (identical to test_multiagent_editor_orig.py) ───────────────
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


# ─── Data Loading ─────────────────────────────────────────────────────────────
log.info("[Data] Loading TrainUsers, TrainSamples_2w, news...")
with open(os.path.join(DATA_DIR, 'TrainUsers.pkl'), 'rb') as f:
    TrainUsers = pickle.load(f)
with open(os.path.join(DATA_DIR, 'TrainSamples_2w.pkl'), 'rb') as f:
    TrainSamples = pickle.load(f)
with open(os.path.join(DATA_DIR, 'news.pkl'), 'rb') as f:
    news = pickle.load(f)
log.info(f"[Data] TrainSamples={len(TrainSamples)}, TrainUsers={len(TrainUsers)}, news={len(news)}")

# ─── Resume ──────────────────────────────────────────────────────────────────
# news_index_dict[news_id] = number of users already saved for that article.
processed_pairs = set()
news_index_dict = {}   # news_id → int count
try:
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                processed_pairs.add((rec['_userindex'], rec['_news_id']))
                nid = rec['_news_id']
                news_index_dict[nid] = news_index_dict.get(nid, 0) + 1
            except Exception:
                continue
    log.info(f"[Resume] {len(processed_pairs)} pairs done.")
except FileNotFoundError:
    log.info("[Resume] Starting fresh.")


# ─── Helper Functions (identical to test_multiagent_editor_orig.py) ──────────

def build_u_hist_str(click_news):
    parts = []
    for i, item in enumerate(click_news):
        cat  = item[0] if len(item) > 0 else "unknown"
        sub  = item[1] if len(item) > 1 else "unknown"
        head = item[2] if len(item) > 2 else ""
        parts.append(f'news_{i}: headline: {head}, category: ({cat},{sub})')
    return '; '.join(parts)


def synthesize_user_profile(u_hist_str):
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
            log.info(f"    [u_prof raw] {raw[:300]!r}")
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
            log.info(f"    [u_prof] attempt {attempt+1} failed: {e}")
            time.sleep(15)
    log.info("    [u_prof] All attempts failed, using raw u_hist as fallback")
    return u_hist_str[:200]


# ─── Core pipeline (identical to test_multiagent_editor_orig.py) ──────────────

def run_multiagent(u_prof, u_hist_str, article_body, original_headline, nrms_score=0.5):
    """Critique-and-Refine where original headline also goes through Editor first."""
    traj = {
        "org_editor_decision":  "",
        "org_editor_analysis":  "",
        "org_valid":            False,
        "org_score":            None,
        "org_analysis":         "",
        "refinement_triggered": False,
        "iterations":           [],
        "final_decision":       "",
    }

    # Step 0: Editor validates original headline
    log.info(f"    [Editor] Validating original headline...")
    orig_editor = EditorAgent(article_body, original_headline)
    orig_ed_decision, orig_ed_analysis, orig_ed_critique = orig_editor.output()
    orig_valid = "[correct]" in orig_ed_decision

    traj["org_editor_decision"] = orig_ed_decision
    traj["org_editor_analysis"] = orig_ed_analysis
    traj["org_valid"]           = orig_valid
    log.info(f"    [Editor] Original: {orig_ed_decision}")

    best_valid_headline = None
    best_valid_score    = -1.0

    if orig_valid:
        log.info(f"    [UserAgent] Scoring original headline...")
        ua = UserAgent(u_prof, u_hist_str, original_headline, nrms_score)
        org_score, org_analysis, org_critique = ua.output()

        if not isinstance(org_score, float):
            log.info(f"    [UserAgent] ERROR non-float: {org_score!r}")
            traj["final_decision"] = "error"
            return original_headline, 0.0, None, orig_valid, traj

        traj["org_score"]    = round(org_score, 4)
        traj["org_analysis"] = org_analysis
        log.info(f"    [UserAgent] org_score={org_score:.3f}")

        best_valid_headline = original_headline
        best_valid_score    = org_score

        if org_score >= TAU_HIGH:
            log.info(f"    [Decision] Keep original (Editor ✓, score={org_score:.3f} >= {TAU_HIGH})")
            traj["final_decision"] = "kept_high"
            return original_headline, org_score, org_score, orig_valid, traj

        memory_buffer = [{
            "iteration":       0,
            "headline":        original_headline,
            "editor_analysis": orig_ed_critique,
            "user_analysis":   org_critique
        }]
    else:
        log.info(f"    [Editor] Original rejected: {orig_ed_critique[:80]}")
        org_score = None
        memory_buffer = [{
            "iteration":       0,
            "headline":        original_headline,
            "editor_analysis": orig_ed_critique,
            "user_analysis":   ""
        }]

    traj["refinement_triggered"] = True
    log.info(f"    [Refinement] Starting, best_so_far={best_valid_score:.3f}...")

    for iteration in range(1, MAX_ITERATIONS + 1):
        log.info(f"    [Iter {iteration}/{MAX_ITERATIONS}] Writer generating...")
        iter_record = {"iteration": iteration}

        writer = WriterAgent(u_prof, u_hist_str, article_body,
                             memory_buffer, original_headline)
        candidate, writer_reasoning = writer.output()

        if not candidate or candidate == "None":
            log.info(f"    [Iter {iteration}] Writer empty, stopping.")
            iter_record["writer"] = "empty"
            traj["iterations"].append(iter_record)
            break

        log.info(f"    [Iter {iteration}] Writer: '{candidate[:60]}'")
        iter_record["writer_headline"]  = candidate
        iter_record["writer_reasoning"] = writer_reasoning

        editor = EditorAgent(article_body, candidate)
        editor_decision, editor_analysis, editor_critique = editor.output()
        is_valid = "[correct]" in editor_decision
        log.info(f"    [Iter {iteration}] Editor: {editor_decision}")

        iter_record["editor_decision"] = editor_decision
        iter_record["editor_analysis"] = editor_analysis

        if not is_valid:
            log.info(f"    [Iter {iteration}] Editor rejected: {editor_critique[:80]}")
            iter_record["user_score"]    = None
            iter_record["user_analysis"] = ""
            traj["iterations"].append(iter_record)
            memory_buffer.append({
                "iteration":       iteration,
                "headline":        candidate,
                "editor_analysis": editor_critique,
                "user_analysis":   ""
            })
            continue

        log.info(f"    [Iter {iteration}] UserAgent scoring candidate...")
        ua = UserAgent(u_prof, u_hist_str, candidate, nrms_score)
        user_score, user_analysis, user_critique = ua.output()

        if not isinstance(user_score, float):
            log.info(f"    [Iter {iteration}] UserAgent ERROR: {user_score!r}")
            iter_record["user_score"]    = "error"
            iter_record["user_analysis"] = ""
            traj["iterations"].append(iter_record)
            break

        log.info(f"    [Iter {iteration}] UserAgent: score={user_score:.3f}")
        iter_record["user_score"]    = round(user_score, 4)
        iter_record["user_analysis"] = user_analysis
        traj["iterations"].append(iter_record)

        if user_score > best_valid_score:
            best_valid_score    = user_score
            best_valid_headline = candidate
            log.info(f"    [Iter {iteration}] New best: {best_valid_score:.3f}")

        if user_score >= TAU_HIGH:
            log.info(f"    [Early exit] score={user_score:.3f} >= {TAU_HIGH}")
            break

        memory_buffer.append({
            "iteration":       iteration,
            "headline":        candidate,
            "editor_analysis": editor_critique,
            "user_analysis":   user_critique
        })

    if best_valid_headline is not None:
        log.info(f"    [Decision] Best Editor-valid: score={best_valid_score:.3f}, "
                 f"'{best_valid_headline[:60]}'")
        traj["final_decision"] = (
            "kept_original" if best_valid_headline == original_headline else "replaced"
        )
        return best_valid_headline, best_valid_score, org_score, orig_valid, traj

    log.info(f"    [Decision] Fallback: no Editor-valid candidate, return original.")
    traj["final_decision"] = "fallback_no_valid"
    return original_headline, 0.0, org_score, orig_valid, traj


# ─── Gen-specific: Rejection Sampling ────────────────────────────────────────

def rejection_sampling(user_label, org_score, org_valid, best_score):
    """Two criteria from the paper."""
    if best_score < TAU_LOW:
        return False, f"best_score={best_score:.2f} < TAU_LOW={TAU_LOW}"
    if org_valid and org_score is not None:
        if user_label == 1 and org_score < TAU_LOW:
            return False, f"Label=click but org_score={org_score:.2f} < TAU_LOW"
        if user_label == 0 and org_score > TAU_HIGH:
            return False, f"Label=no-click but org_score={org_score:.2f} > TAU_HIGH"
    return True, "Accepted"


# ─── Gen-specific: Rationale Synthesis ───────────────────────────────────────

def synthesize_rationale(traj_dict, article_body, u_prof, best_headline):
    """r* <- LLM(T || c || u || h* || rho_rat)"""
    iterations = traj_dict.get("iterations", [])
    traj_parts = []
    if traj_dict.get("org_score") is not None:
        traj_parts.append(
            f"Round 0: headline='{traj_dict.get('org_editor_analysis', '')}', "
            f"user_score={traj_dict['org_score']:.2f}"
        )
    for entry in iterations:
        k, h = entry.get("iteration", "?"), entry.get("writer_headline", "")
        i_e  = entry.get("editor_analysis", "")
        i_u  = entry.get("user_analysis", "")
        s    = entry.get("user_score", None)
        part = f"Round {k}: headline='{h}'"
        if i_e and i_e not in ("None", ""):
            part += f", editor_analysis='{i_e}'"
        if i_u and i_u not in ("None", ""):
            part += f", user_analysis='{i_u}'"
        if s is not None and isinstance(s, float):
            part += f", user_score={s:.2f}"
        traj_parts.append(part)
    traj_str = "\n".join(traj_parts) if traj_parts else "(no trajectory)"

    prompt = f"""You are a reasoning synthesizer for personalized headline generation. Given a multi-agent interaction trajectory, synthesize a structured rationale explaining HOW the final headline was constructed by following two explicit reasoning steps.

**Multi-Agent Interaction Trajectory (T):**
{traj_str}

**Article Body (c):**
{article_body[:200]}

**User Profile (u_prof):**
{u_prof}

**Final Selected Headline (h*):**
{best_headline}

**Task:** Write a coherent reasoning chain r* that mirrors the Generator's two-step pipeline:

Step 1 — Editorial Grounding (extract salient article facts):
  Identify the key facts, entities, and events from the article body that the headline must faithfully convey. Explain which facts were selected and why they ensure factual accuracy, conciseness, and no clickbait.

Step 2 — User Preference Alignment (infer latent topic/style preferences from u_prof and u_hist):
  Analyse the user's topic interests and headline style preferences (length, tone, formality) inferred from the profile and click history. Explain how the headline's framing, wording, and focus were adjusted to match these preferences.

Synthesis — How the two steps combine into h*:
  Briefly state how the editorial facts (Step 1) and user preferences (Step 2) were jointly satisfied in h*, referencing any critique signals from the trajectory that guided refinement.

Output 3-5 sentences covering all three parts above."""

    for attempt in range(3):
        try:
            result = gen_client.chat.completions.create(
                model="qwen3-8b",
                messages=[
                    {"role": "system", "content": "You are a reasoning synthesizer for personalized headline generation."},
                    {"role": "user",   "content": prompt}
                ]
            )
            return result.choices[0].message.content.strip()
        except Exception as e:
            log.info(f"    [rationale] attempt {attempt+1} failed: {e}")
            time.sleep(15)
    return (f"The headline '{best_headline}' satisfies editorial accuracy by grounding "
            f"key facts from the article, while aligning with the user's topic interests "
            f"and preferred reading style.")


# ─── Gen-specific: build formatted output record ─────────────────────────────

def build_output_record(u_prof, u_hist_str, article_body, rationale, best_headline,
                        userindex, news_id):
    """Build the {system, instruction, output} record for Generator fine-tuning."""
    instruction = (
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
        f"[User Short-term Click History]: {u_hist_str}\n\n"
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
    output = f"{rationale}\n\nHeadline: {best_headline}"
    return {
        "system":      SYSTEM,
        "instruction": instruction,
        "output":      output,
        # hidden keys for resume tracking (prefixed _ to distinguish)
        "_userindex":  userindex,
        "_news_id":    news_id,
    }


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    saved_count = 0
    error_count = 0

    log.info(f"[Main] TrainSamples={len(TrainSamples)}, USERS_PER_NEWS={USERS_PER_NEWS}, "
             f"TARGET_TOTAL={TARGET_TOTAL}")

    pbar = tqdm(enumerate(TrainSamples), total=len(TrainSamples),
                desc="Stage1-OneToMany", unit="row", dynamic_ncols=True)

    for i, row in pbar:
        userindex = row[0]
        pos_neg   = row[1]

        for j, new_id in enumerate(pos_neg):
            user_label = 1 if j == 0 else 0

            # One-to-Many: skip if this news already has USERS_PER_NEWS users saved
            if news_index_dict.get(new_id, 0) >= USERS_PER_NEWS:
                continue

            # Skip already-done (user, news) pair
            if (userindex, new_id) in processed_pairs:
                continue

            if new_id not in news:
                log.info(f"  [SKIP] {new_id} not in news dict")
                continue

            news_item         = news[new_id]
            original_headline = news_item[2] if len(news_item) > 2 else ""
            article_body      = ' '.join(news_item[3].split()[:200]) if len(news_item) > 3 else ""

            log.info(f"[{i}] user={userindex}, news={new_id}, label={user_label}")

            # Click history
            click_str_ids = TrainUsers[userindex][0].split()[-200:]
            click_news    = [news[s] for s in click_str_ids if s in news]
            if not click_news:
                log.info(f"  [SKIP] no click history for user={userindex}")
                continue

            u_hist_str = build_u_hist_str(click_news[-20:]) if len(click_news) > 20 else build_u_hist_str(click_news)
            u_prof     = synthesize_user_profile(build_u_hist_str(click_news))
            nrms_score = get_nrms_score(click_str_ids, new_id)
            log.info(f"  [NRMS] S={nrms_score:.3f}")

            try:
                best_headline, best_score, org_score, org_valid, traj = run_multiagent(
                    u_prof, u_hist_str, article_body, original_headline, nrms_score
                )
            except Exception as e:
                error_count += 1
                log.info(f"  [ERROR] {e}\n{traceback.format_exc()}")
                continue

            if not best_headline:
                continue

            # Rejection sampling
            is_valid, reason = rejection_sampling(user_label, org_score, org_valid, best_score)
            if not is_valid:
                log.info(f"  [Rejected] {reason}")
                continue

            # Rationale synthesis
            rationale = synthesize_rationale(traj, article_body, u_prof, best_headline)

            # Build output record
            out_rec = build_output_record(
                u_prof, u_hist_str, article_body, rationale, best_headline,
                userindex, new_id
            )

            # Save formatted output
            try:
                with open(OUTPUT_FILE, 'a', encoding='utf-8', buffering=1) as f:
                    f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                processed_pairs.add((userindex, new_id))
                news_index_dict[new_id] = news_index_dict.get(new_id, 0) + 1
                saved_count += 1
                log.info(f"  [Saved #{saved_count}/{TARGET_TOTAL}] best='{best_headline[:60]}'")
                pbar.set_postfix(saved=saved_count, errors=error_count)
                if saved_count >= TARGET_TOTAL:
                    log.info(f"[Main] Reached TARGET_TOTAL={TARGET_TOTAL}, stopping.")
                    pbar.close()
                    return
            except Exception as e:
                log.info(f"  [ERROR] write failed: {e}")
                error_count += 1
                continue

            # Save detail record
            detail_rec = {
                "userindex":         userindex,
                "news_id":           new_id,
                "user_label":        user_label,
                "u_prof":            u_prof,
                "nrms_score":        round(nrms_score, 4),
                "original_headline": original_headline,
                "best_headline":     best_headline,
                "best_score":        round(best_score, 4),
                "org_score":         round(org_score, 4) if org_score is not None else None,
                "replaced":          best_headline != original_headline,
                **traj,
            }
            try:
                with open(DETAIL_FILE, 'a', encoding='utf-8', buffering=1) as f:
                    f.write(json.dumps(detail_rec, ensure_ascii=False) + "\n")
            except Exception as e:
                log.info(f"  [ERROR] detail write failed: {e}")

    log.info("=" * 60)
    log.info(f"[Done] Saved={saved_count}, Errors={error_count}")
    log.info(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
