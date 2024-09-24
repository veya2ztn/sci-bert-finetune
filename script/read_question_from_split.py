from tqdm import tqdm
import multiprocessing
import json
import re,os
import concurrent.futures
import pandas as pd
# assuming your function find_subsentence is defined somewhere
ROOTDIR = 'data/unarXive_quantum_physics/'

print("loading csv files...........")
sentense_ids = pd.read_csv(os.path.join(ROOTDIR,"unarXive_quantum_physics.clear.sections.id.csv"))
sentense_ids = list(sentense_ids.groupby('paper_id'))
print("done~!")
paper_unique_id_to_index={}
for paper_id, group in sentense_ids:
    if paper_id not in paper_unique_id_to_index:
        paper_unique_id_to_index[paper_id] = len(paper_unique_id_to_index)


def find_subsentence(sentence):
    match = re.search(r'(In|What|How|Why|Where|Can|Could|Would|Is|Will).*\?', sentence)
    if match:
        return match.group(0)
    else:
        return None


def deal_with_json_file(path):
    try:
        with open(path, 'r') as f:data = json.load(f)
    except:
        print(f"fail at {path}")
        return [],[]
    good_questions = []
    bad_questions = []

    for sample_id, metadata in data.items():
        #print(question_id)
        paper_id = metadata['paper_id']
        for key,val in metadata.items():
            if key in ['outlines','paper_id']:continue
            question_type = key.replace('question_for_','')
            for question in val.strip().split('\n'):
                if len(question) == 0:continue
                if not question:continue
                true_question = find_subsentence(question.strip())
                #print(f"{question_type}==>{true_question}")
                if true_question is None:
                    bad_questions.append([paper_unique_id_to_index[paper_id], paper_id, question_type, question])
                else:
                    good_questions.append([paper_unique_id_to_index[paper_id], paper_id, question_type, true_question])

    return good_questions, bad_questions


def multiprocessing_handler(path_list, max_workers):
    good_questions = []
    bad_questions = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(
            deal_with_json_file, path): path for path in path_list}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            good_questions.extend(result[0])
            bad_questions.extend(result[1])

    return good_questions, bad_questions


ROOTDIR = 'data/unarXive_quantum_physics/'

ROOTPATH = os.path.join(ROOTDIR,'full_paper_question_results/')
path_list = [os.path.join(ROOTPATH,p) for p in os.listdir(ROOTPATH)]
deal_with_json_file(path_list[0])

good_questions, bad_questions = multiprocessing_handler(path_list,64)
print(len(good_questions))
print(len(bad_questions))

import pandas as pd
good_questions = pd.DataFrame(good_questions,
columns=['question_id','paper_id','question_type','question'])
good_questions.to_csv(os.path.join(ROOTDIR, 'query_full_paper.question.good_questions.csv'))

import json
with open(os.path.join(ROOTDIR, 'query_full_paper.question_answer_map.json'),'w') as f:
    json.dump(good_questions['question_id'].to_dict(),f)


bad_questions= pd.DataFrame(bad_questions,
columns=['question_id','paper_id','question_type','question'])
bad_questions.to_csv(os.path.join(ROOTDIR, 'query_full_paper.question.bad_questions.csv'))
