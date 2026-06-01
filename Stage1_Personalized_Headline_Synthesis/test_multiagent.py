"""
Multi-Agent Test: Original Headline Also Goes Through Editor

Variant of test_multiagent.py where the original headline undergoes
the same Editor → User pipeline as Writer-generated candidates.

Key difference from test_multiagent.py:
  Original:  UserAgent scores original directly (Editor skipped)
  This file: Editor validates original first → if rejected, org_score=0.0
             and refinement starts immediately; if accepted, UserAgent scores.

Everything else (TAU_HIGH, IMPROVEMENT_MARGIN, MAX_ITERATIONS, replacement
logic, trajectory saving) is identical to test_multiagent.py.

Output: data/pred_multiagent_editor_orig.txt
        data/pred_multiagent_editor_orig_trajectory.jsonl

Run from submit_code/:
    python test_multiagent.py
"""

import os
import sys
import json
import pickle
import time
import traceback
import logging
from tqdm import tqdm

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_FILE      = os.path.join(DATA_DIR, 'pred_multiagent_editor_orig.txt')
TRAJECTORY_FILE  = os.path.join(DATA_DIR, 'pred_multiagent_editor_orig_trajectory.jsonl')
LOG_FILE         = os.path.join(DATA_DIR, 'test_multiagent_editor_orig_debug.log')

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',   # no timestamp (overhead negligible, log is cleaner)
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stderr),
    ]
)
log = logging.getLogger(__name__)

# ─── Agent imports ───────────────────────────────────────────────────────────
import importlib.util

def _load_agent(agent_dir, class_name):
    path = os.path.join(SCRIPT_DIR, '..', 'Simulator_Construction', agent_dir, 'inference.py')
    path = os.path.normpath(path)
    spec = importlib.util.spec_from_file_location(f"{agent_dir}_inference", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)

UserAgent   = _load_agent('User',   'UserAgent')
EditorAgent = _load_agent('Editor', 'EditorAgent')
WriterAgent = _load_agent('Writer', 'WriterAgent')

from openai import OpenAI

# ─── Configuration (identical to test_multiagent.py) ─────────────────────────
TAU_HIGH           = 0.85
IMPROVEMENT_MARGIN = 0.05
MAX_ITERATIONS     = 3

gen_client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")

log.info("=" * 60)
log.info("USG Multi-Agent (Editor validates original headline too)")
log.info("=" * 60)

# ─── NRMS Scorer ─────────────────────────────────────────────────────────────
_nrms_scorer = None
try:
    sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'Simulator_Construction', 'User'))
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
log.info("[Data] Loading TestUsers, TestSamples, news...")
with open(os.path.join(DATA_DIR, 'TestUsers.pkl'), 'rb') as f:
    TestUsers = pickle.load(f)
with open(os.path.join(DATA_DIR, 'TestSamples.pkl'), 'rb') as f:
    TestSamples = pickle.load(f)
with open(os.path.join(DATA_DIR, 'news.pkl'), 'rb') as f:
    news = pickle.load(f)
log.info(f"[Data] TestSamples={len(TestSamples)}, TestUsers={len(TestUsers)}, news={len(news)}")

# ─── Resume ──────────────────────────────────────────────────────────────────
done_set = set()
try:
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                done_set.add((rec['userindex'], rec['new_id']))
            except Exception:
                continue
    log.info(f"[Resume] {len(done_set)} samples already done.")
except FileNotFoundError:
    log.info("[Resume] Starting fresh.")


# ─── Helper Functions (identical to test_multiagent.py) ──────────────────────

def build_u_hist_str(click_news):
    parts = []
    for i, item in enumerate(click_news):
        cat  = item[0] if len(item) > 0 else "unknown"
        sub  = item[1] if len(item) > 1 else "unknown"
        head = item[2] if len(item) > 2 else ""
        parts.append(f'news_{i}: headline: {head}, category: ({cat},{sub})')
    return '; '.join(parts)


def synthesize_user_profile(u_hist_str):
    import re
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
            raw = resp.choices[0].message.content
            log.info(f"    [u_prof raw] {raw[:300]!r}")
            raw = raw.strip()
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


# ─── Core pipeline ────────────────────────────────────────────────────────────

def run_multiagent(u_prof, u_hist_str, article_body, original_headline, nrms_score=0.5):
    """Critique-and-Refine where original headline also goes through Editor first.

    Step 0a: Editor validates original headline
      - Rejected → org_valid=False, org_score=0.0, start refinement immediately
      - Accepted → proceed to UserAgent scoring

    Step 0b: UserAgent scores editor-valid original
      - org_score >= TAU_HIGH → keep original
      - org_score < TAU_HIGH  → start refinement

    Refinement loop and replacement logic identical to test_multiagent.py.
    """
    _t_start = time.time()   # run_multiagent start time
    traj = {
        "org_editor_decision":  "",
        "org_editor_analysis":  "",
        "org_valid":            False,
        "org_score":            None,
        "org_analysis":         "",
        "refinement_triggered": False,
        "iterations":           [],
        "final_decision":       "",
        "writer_times":         [],   # seconds per writer.output() call
        "total_time":           0.0,  # seconds for entire run_multiagent
    }

    # ── Step 0: Editor validates original headline ────────────────────────────
    log.info(f"    [Editor] Validating original headline...")
    orig_editor = EditorAgent(article_body, original_headline)
    orig_ed_decision, orig_ed_analysis, orig_ed_critique = orig_editor.output()
    orig_valid = "[correct]" in orig_ed_decision

    traj["org_editor_decision"] = orig_ed_decision
    traj["org_editor_analysis"] = orig_ed_analysis
    traj["org_valid"]           = orig_valid
    log.info(f"    [Editor] Original: {orig_ed_decision}")

    # org_score: float baseline for comparison (0.0 if Editor rejects original)
    # best_gen: tracks best Editor-valid GENERATED candidate (original excluded)
    best_gen_headline = None
    best_gen_score    = -1.0

    if orig_valid:
        log.info(f"    [UserAgent] Scoring original headline...")
        ua = UserAgent(u_prof, u_hist_str, original_headline, nrms_score)
        org_score, org_analysis, org_critique = ua.output()

        if not isinstance(org_score, float):
            log.info(f"    [UserAgent] ERROR non-float: {org_score!r}")
            traj["final_decision"] = "error"
            traj["total_time"] = time.time() - _t_start
            return original_headline, 0.0, traj

        traj["org_score"]    = round(org_score, 4)
        traj["org_analysis"] = org_analysis
        log.info(f"    [UserAgent] org_score={org_score:.3f}")

        if org_score >= TAU_HIGH:
            log.info(f"    [Decision] Keep original (Editor ✓, score={org_score:.3f} >= {TAU_HIGH})")
            traj["final_decision"] = "kept_high"
            traj["total_time"] = time.time() - _t_start
            return original_headline, org_score, traj

        memory_buffer = [{
            "iteration":       0,
            "headline":        original_headline,
            "editor_analysis": orig_ed_critique,
            "user_analysis":   org_critique
        }]
    else:
        log.info(f"    [Editor] Original rejected: {orig_ed_critique[:80]}")
        org_score = 0.0   # treat as zero so any improvement is measurable
        memory_buffer = [{
            "iteration":       0,
            "headline":        original_headline,
            "editor_analysis": orig_ed_critique,
            "user_analysis":   ""
        }]

    traj["refinement_triggered"] = True
    log.info(f"    [Refinement] Starting, org_score={org_score:.3f}, "
             f"need gen > org + {IMPROVEMENT_MARGIN}...")

    # ── Refinement loop ───────────────────────────────────────────────────────
    for iteration in range(1, MAX_ITERATIONS + 1):
        log.info(f"    [Iter {iteration}/{MAX_ITERATIONS}] Writer generating...")
        iter_record = {"iteration": iteration}

        writer = WriterAgent(u_prof, u_hist_str, article_body,
                             memory_buffer, original_headline)
        _tw = time.time()
        candidate, writer_reasoning = writer.output()
        traj["writer_times"].append(time.time() - _tw)

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

        log.info(f"    [Iter {iteration}] UserAgent: score={user_score:.3f}, "
                 f"delta={user_score - org_score:.3f}")
        iter_record["user_score"]    = round(user_score, 4)
        iter_record["user_analysis"] = user_analysis
        traj["iterations"].append(iter_record)

        # Update best Editor-valid GENERATED candidate
        if user_score > best_gen_score:
            best_gen_score    = user_score
            best_gen_headline = candidate
            log.info(f"    [Iter {iteration}] New best gen: {best_gen_score:.3f}")

        # Early exit: high quality AND meaningful improvement over original
        if user_score >= TAU_HIGH and (user_score - org_score) > IMPROVEMENT_MARGIN:
            log.info(f"    [Early exit] score={user_score:.3f} >= {TAU_HIGH} and "
                     f"delta={user_score - org_score:.3f} > {IMPROVEMENT_MARGIN}")
            break

        memory_buffer.append({
            "iteration":       iteration,
            "headline":        candidate,
            "editor_analysis": editor_critique,
            "user_analysis":   user_critique
        })

    # ── Final: replace original only when best_gen beats it by > IMPROVEMENT_MARGIN
    if best_gen_headline is not None:
        delta = best_gen_score - org_score
        if delta > IMPROVEMENT_MARGIN:
            log.info(f"    [Decision] REPLACE: {org_score:.3f} -> {best_gen_score:.3f} "
                     f"(delta={delta:.3f} > {IMPROVEMENT_MARGIN})")
            traj["final_decision"] = "replaced"
            traj["total_time"] = time.time() - _t_start
            return best_gen_headline, best_gen_score, traj
        else:
            log.info(f"    [Decision] Keep original: best_gen={best_gen_score:.3f}, "
                     f"delta={delta:.3f} not > {IMPROVEMENT_MARGIN}")
            traj["final_decision"] = "kept_no_improvement"
    else:
        log.info(f"    [Decision] Keep original: no Editor-valid generated candidate.")
        traj["final_decision"] = "kept_no_valid_candidate"

    traj["total_time"] = time.time() - _t_start
    return original_headline, org_score, traj


# ─── Main Loop (identical to test_multiagent.py) ─────────────────────────────

def main():
    total  = len(TestSamples)
    saved  = len(done_set)
    errors = 0

    # ── Timing accumulators ───────────────────────────────────────────────────
    total_ma_time     = 0.0   # sum of run_multiagent durations
    total_ma_count    = 0     # number of completed run_multiagent calls
    total_wr_time     = 0.0   # sum of writer.output() durations
    total_wr_count    = 0     # number of writer.output() calls

    log.info(f"[Main] total={total}, done={saved}, output={OUTPUT_FILE}")

    pbar = tqdm(enumerate(TestSamples), total=total,
                desc="Multi-Agent(+EditorOrig)", unit="sample", dynamic_ncols=True)

    for i, row in pbar:
        userindex = row[0]
        new_id    = row[1]
        label     = row[2]

        if (userindex, new_id) in done_set:
            continue

        log.info(f"[{i+1}/{total}] user={userindex}, news={new_id}")
        log.info(f"  label: {label[:80]}")

        if new_id not in news:
            log.info(f"  [SKIP] news_id not found")
            continue

        news_item    = news[new_id]
        ori_headline = news_item[2] if len(news_item) > 2 else ""
        article_body = ' '.join(news_item[3].split()[:200]) if len(news_item) > 3 else ""
        log.info(f"  ori_headline: {ori_headline[:80]}")

        click_ids  = TestUsers[userindex][-200:]
        click_news = [news[nid] for nid in click_ids if nid in news]
        if not click_news:
            log.info(f"  [SKIP] no click history")
            continue
        log.info(f"  click_history: {len(click_ids)} IDs, {len(click_news)} found")

        u_hist_str = build_u_hist_str(click_news[-20:])
        u_prof     = synthesize_user_profile(build_u_hist_str(click_news))
        nrms_score = get_nrms_score(click_ids, new_id)
        log.info(f"  [NRMS] S={nrms_score:.3f}")

        try:
            pred_headline, pred_score, traj = run_multiagent(
                u_prof, u_hist_str, article_body, ori_headline, nrms_score
            )
        except Exception as e:
            errors += 1
            log.info(f"  [ERROR] {e}")
            log.info(traceback.format_exc())
            continue

        # ── Accumulate timing stats ───────────────────────────────────────────
        total_ma_time  += traj.get("total_time", 0.0)
        total_ma_count += 1
        wt = traj.get("writer_times", [])
        total_wr_time  += sum(wt)
        total_wr_count += len(wt)

        record = {
            "userindex":    userindex,
            "new_id":       new_id,
            "pred_output":  pred_headline,
            "ori_headline": ori_headline,
            "label":        label,
            "pred_score":   round(pred_score, 4),
            "replaced":     pred_headline != ori_headline
        }
        try:
            with open(OUTPUT_FILE, 'a', encoding='utf-8', buffering=1) as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            done_set.add((userindex, new_id))
            saved += 1

            avg_ma = total_ma_time / total_ma_count if total_ma_count else 0.0
            avg_wr = total_wr_time / total_wr_count if total_wr_count else 0.0
            log.info(f"  [Saved #{saved}] replaced={record['replaced']}, "
                     f"avg_run={avg_ma:.1f}s, avg_writer={avg_wr:.1f}s, "
                     f"pred='{pred_headline[:60]}'")
            pbar.set_postfix(saved=saved, errors=errors,
                             avg_run=f"{avg_ma:.0f}s",
                             avg_writer=f"{avg_wr:.0f}s")
        except Exception as e:
            log.info(f"  [ERROR] write failed: {e}")
            errors += 1

        traj_record = {
            "userindex":     userindex,
            "new_id":        new_id,
            "ori_headline":  ori_headline,
            "label":         label,
            "u_prof":        u_prof,
            "nrms_score":    round(nrms_score, 4),
            **traj,
            "pred_headline": pred_headline,
            "pred_score":    round(pred_score, 4),
            "replaced":      pred_headline != ori_headline,
        }
        try:
            with open(TRAJECTORY_FILE, 'a', encoding='utf-8', buffering=1) as f:
                f.write(json.dumps(traj_record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.info(f"  [ERROR] trajectory write failed: {e}")

    avg_ma = total_ma_time / total_ma_count if total_ma_count else 0.0
    avg_wr = total_wr_time / total_wr_count if total_wr_count else 0.0
    log.info("=" * 60)
    log.info(f"[Done] Saved={saved}, Errors={errors}, Total={total}")
    log.info(f"[Timing] avg run_multiagent = {avg_ma:.2f}s  "
             f"(over {total_ma_count} calls)")
    log.info(f"[Timing] avg writer.output  = {avg_wr:.2f}s  "
             f"(over {total_wr_count} calls)")
    log.info(f"Output: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
