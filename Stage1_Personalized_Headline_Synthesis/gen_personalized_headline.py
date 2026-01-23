import pickle
import random
from agents import GenAgent, UserAgent, EditorAgent
import json

user_index_set = set()
with open('data/TrainUsers.pkl', 'rb') as f:
    TrainUsers = pickle.load(f)
with open('data/TrainSamples_2w.pkl', 'rb') as f:
    TrainSamples = pickle.load(f)
with open('data/news.pkl', 'rb') as f:
    news = pickle.load(f)
with open('data/personalized_headline_train.txt', 'rb') as f:
    for line in f:
        line = json.loads(line)
        user_index_set.add(line["userindex"])
attempts = 4

def multi_agent_data(clk_news_str, new_body, new_head,prob_clk,user_feedback):
    prior_feedback = f"""Round1:
Editor Writting Headline: {new_head}
User Feedback: {user_feedback}
"""
    normal_label = "0"
    attempt_j = 0
    user_clk_scores = []
    user_decisions = []
    pred_heads = []
    preb_clk_max = -1
    preb_gen_max = ''
    for j in range(attempts):
        Gen = GenAgent(clk_news_str, new_body, new_head, prior_feedback)
        pred_gen = Gen.output()
        if pred_gen == "None" or pred_gen.strip() == "":
            normal_label = "1"
            break
        Editor = EditorAgent(new_body, pred_gen)
        pred_editor_decision, pred_editor_feedback = Editor.output()
        if ("[incorrect]" not in pred_editor_decision) and ("[correct]" not in pred_editor_decision):
            normal_label = "1"
            break
        if "[incorrect]" in pred_editor_decision:
            prior_feedback = prior_feedback + f"""Round{j+2}:
You Writting Headline: {pred_gen}
Editor Feedback: {pred_editor_feedback}
"""
            continue
        User = UserAgent(clk_news_str,pred_gen)
        prob_clk_agent, pred_user_decision, pred_user_feedback = User.output()
        if isinstance(prob_clk_agent, str):
            normal_label = "1"
            break
        if ("click" not in pred_user_decision) and ("not click" not in pred_user_decision):
            normal_label = "1"
            break
        if int(prob_clk_agent) == 1:
            preb_clk_max = prob_clk_agent
            preb_gen_max = pred_gen
            break
        else:
            user_clk_scores.append(prob_clk_agent)
            user_decisions.append(pred_user_decision)
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
        print(f"current prior_feedback: {prior_feedback}")
    if preb_clk_max - prob_clk < 0.01:
        preb_gen_max = new_head
    return pred_gen, normal_label, prior_feedback, attempt_j,preb_clk_max, preb_gen_max


for i in range(len(TrainSamples)):
    normal_label = "0"
    row = TrainSamples[i]
    userindex = row[0]
    if userindex in user_index_set:
        continue
    user_index_set.add(userindex)
    pos = row[1][0]
    neg = row[1][1]
    clks = TrainUsers[userindex][0].split(" ")[-20:]
    pos_news = news[pos]
    neg_news = news[neg]
    clk_news = [news[clk] for clk in clks]
    clk_news_str = '; '.join([f'new_{i}: headline: {clk_news[i][2]}, category: ({clk_news[i][0]},{clk_news[i][1]})' for i in range(len(clk_news))])
    pos_head = pos_news[2]
    pos_body = ' '.join(pos_news[3].split(" ")[:200])
    User = UserAgent(clk_news_str,pos_head)
    prob_clk,user_decision, user_feedback= User.output()
    if isinstance(prob_clk, str):
        continue
    pos_neg_label = "1"
    attempt_j = -1
    if int(prob_clk) == 1:
        label = "1"
        preb_gen_max = pos_head
        pred_gen = pos_head
        prior_feedback = ""
    else:
        pred_gen, normal_label, prior_feedback, attempt_j,preb_clk_max, preb_gen_max = multi_agent_data(clk_news_str, pos_body, pos_head,prob_clk,user_feedback)
        label = "0"
        if preb_gen_max == pos_head:
            label = "3"
    pos_neg_label = "1"
    if normal_label == "0":
        with open("data/personalized_headline_train.txt", "a", encoding="utf-8", buffering=1) as f:
            write_json = {"userindex":userindex, "clk_news_str": clk_news_str,"ori_headline":pos_head,"ori_body":pos_body, "candi_new_str": preb_gen_max,"candi_new_id": pos,"label":label, "multi_agent":prior_feedback,"pos_neg_label":pos_neg_label,"pred_gen":pred_gen,"attempt_j":attempt_j}
            write_dump = json.dumps(write_json, ensure_ascii=False)
            f.write(write_dump)
            f.write("\n")
    neg_body = ' '.join(neg_news[3].split(" ")[:100])
    neg_head = neg_news[2]
    User = UserAgent(clk_news_str,neg_head)
    prob_clk,user_decision, user_feedback= User.output()
    if isinstance(prob_clk, str):
        continue
    normal_label = "0"
    if int(prob_clk) == 1:
        continue
    else:
        number_neg = random.random()
        if number_neg > 0.5:
            pred_gen, normal_label, prior_feedback, attempt_j,preb_clk_max, preb_gen_max = multi_agent_data(clk_news_str, neg_body, neg_head,prob_clk,user_feedback)
            label = "0"
            if preb_gen_max == neg_head:
                label = "3"
            pos_neg_label = "0"
            if normal_label == "0":
                with open("data/personalized_headline_train.txt", "a", encoding="utf-8", buffering=1) as f:
                    write_json = {"userindex":userindex, "clk_news_str": clk_news_str,"ori_headline":neg_head,"ori_body":neg_body, "candi_new_str": preb_gen_max,"candi_new_id": neg,"label":label, "multi_agent":prior_feedback,"pos_neg_label":pos_neg_label,"pred_gen":pred_gen,"attempt_j":attempt_j}
                    write_dump = json.dumps(write_json, ensure_ascii=False)
                    f.write(write_dump)
                    f.write("\n")


