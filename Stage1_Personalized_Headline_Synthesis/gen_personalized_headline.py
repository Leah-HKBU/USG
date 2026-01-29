import pickle
import random
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Simulator_Construction.Writer.inference import WriterAgent
from Simulator_Construction.User.inference import UserAgent
from Simulator_Construction.User.distill_data import Distill_Data
from Simulator_Construction.Editor.inference import EditorAgent


def multi_agent_data_collaboration(user_profile, clk_news_str, new_body, new_head,prob_clk,user_feedback):
    prior_feedback = f"""Round1:
Editor Writting Headline: {new_head}
User Feedback: {user_feedback}
"""
    normal_label = "0"
    attempt_j = 0
    attempts = 4
    user_clk_scores = []
    pred_heads = []
    preb_clk_max = -1
    preb_gen_max = ''
    for j in range(attempts):
        Gen = WriterAgent(user_profile, clk_news_str, new_body, new_head, prior_feedback)
        pred_gen = Gen.output()
        if pred_gen == "None" or pred_gen.strip() == "":
            normal_label = "1"
            break
        Editor = EditorAgent(new_body, pred_gen)
        pred_editor_decision, pred_editor_feedback = Editor.output()
        if ("yes" not in pred_editor_decision) and ("no" not in pred_editor_decision):
            normal_label = "1"
            break
        if "no" in pred_editor_decision:
            prior_feedback = prior_feedback + f"""Round{j+2}:
You Writting Headline: {pred_gen}
Editor Feedback: {pred_editor_feedback}
"""
            continue
        User = UserAgent(user_profile, clk_news_str,pred_gen)
        prob_clk_agent, pred_user_feedback = User.output()
        if isinstance(prob_clk_agent, str):
            normal_label = "1"
            break
        if int(prob_clk_agent) > 0.8:
            preb_clk_max = prob_clk_agent
            preb_gen_max = pred_gen
            break
        else:
            user_clk_scores.append(prob_clk_agent)
            pred_heads.append(pred_gen)
            if preb_clk_max < prob_clk_agent:
                preb_clk_max = prob_clk_agent
                preb_gen_max = pred_gen
            prior_feedback = prior_feedback + f"""Round{j+2}:
You Writting Headline: {pred_gen}
Editor Feedback: {pred_editor_feedback}
User Feedback: {pred_user_feedback}
"""
        attempt_j += 1
    if preb_clk_max - prob_clk < 0.1:
        preb_gen_max = new_head
        preb_clk_max = prob_clk
    if preb_clk_max < 0.1:
        normal_label = "1"
    return pred_gen, normal_label, prior_feedback, attempt_j,preb_clk_max, preb_gen_max


def main():
    news_index_dict = {}
    user_samp = 3
    with open('../data/TrainUsers.pkl', 'rb') as f:
        TrainUsers = pickle.load(f)
    with open('../data/TrainSamples.pkl', 'rb') as f:
        TrainSamples = pickle.load(f)
    with open('../data/news.pkl', 'rb') as f:
        news = pickle.load(f)
    with open("../Simulator_Construction/User/data/profile.txt", 'rb') as f:
        user_profiles = {}
        for line in f:
            line = line.strip()
            line_dict = json.loads(line)
            user_profiles[line_dict["userindex"]] = line_dict["user_profile"]
    with open("../Simulator_Construction/prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
        user_prompts = prompts["User"]
    for i in range(len(TrainSamples)):
        normal_label = "0"
        row = TrainSamples[i]
        userindex = row[0]
        pos_neg = row[1]
        for j in range(len(pos_neg)):
            new_id = pos_neg[j]
            if new_id not in news_index_dict:
                news_index_dict[new_id] = 0
            news_index_dict[new_id] += 1
            if news_index_dict[new_id] > user_samp:
                continue
            if j==0:
                label = 1
            else:
                label = 0
            clks = TrainUsers[userindex][0].split(" ")[-20:]
            clk_news = [news[clk] for clk in clks]
            clk_news_str = '; '.join([f'new_{i}: headline: {clk_news[i][2]}, category: ({clk_news[i][0]},{clk_news[i][1]})' for i in range(len(clk_news))])
            long_clks = TrainUsers[userindex][0].split(" ")[-200:]
            long_clks_news = [news[clk] for clk in long_clks]
            long_clks_news_str = '; '.join([f'new_{i}: headline: {long_clks_news[i][2]}, category: ({long_clks_news[i][0]},{long_clks_news[i][1]})' for i in range(len(long_clks_news))])
            news_j = news[new_id]
            if userindex not in user_profiles:
                Distill_cls = Distill_Data("", ["profile"],long_clks_news_str,clk_news_str,label,news_j,user_prompts)
                user_profiles[userindex] = Distill_cls.output()
            user_profile = user_profiles[userindex]
            new_head = news_j[2]
            new_body = ' '.join(news_j[3].split(" ")[:500])
            User = UserAgent(user_profile, clk_news_str,new_head)
            prob_clk, user_feedback= User.output()
            if isinstance(prob_clk, str):
                continue
            attempt_j = -1
            if label == 1 and int(prob_clk) > 0.8:
                preb_gen_max = new_head
                pred_gen = new_head
                prior_feedback = ""
            elif label == 1 and int(prob_clk) < 0.1:
                continue
            elif label == 0 and int(prob_clk) > 0.8:
                continue
            else:
                pred_gen, normal_label, prior_feedback, attempt_j,preb_clk_max, preb_gen_max = multi_agent_collaboration(user_profile, clk_news_str, new_body, new_head,prob_clk,user_feedback)
            if normal_label == "0":
                with open("data/personalized_headline_train.txt", "a", encoding="utf-8", buffering=1) as f:
                    write_json = {"userindex":userindex, "clk_news_str": clk_news_str,"ori_headline":new_head,"ori_body":new_body, "candi_new_str": preb_gen_max,"candi_new_id": new_id, "multi_agent":prior_feedback,"pos_neg_label":label,"pred_gen":pred_gen,"attempt_j":attempt_j,"user_profile":user_profile}
                    write_dump = json.dumps(write_json, ensure_ascii=False)
                    f.write(write_dump)
                    f.write("\n")


if __name__ == "__main__":
    main()