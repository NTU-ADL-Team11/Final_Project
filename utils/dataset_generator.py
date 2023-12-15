from openai import OpenAI
import pandas as pd
import yaml
import uuid
from tqdm import tqdm
from context import *
import random
import chinese_converter
import json

client = OpenAI(api_key="")
MODEL = "gpt-3.5-turbo"
# FIXME: Add diversity
INSTRUCTION = {"question_answering": question_answering_context,
               "preach": preach_context,
               "pray": pray_context,
               "consult": consult_context}
INSTRUCTION_NO_INPUT = {"pray": "請根據聖經為使用者的處境禱告。", "consult": "請根據聖經幫助使用者面對他的處境。"}

def inference(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=MODEL,
    )
    return chat_completion.choices[0].message.content

def prompt_input_generator(tp=None, context=None, **kwargs):
    if tp == "pray" or tp == "consult":
        return f"## 處境: {kwargs['situation']} ## 經文: {context}" if context != None else kwargs['situation']
    elif tp == "question_answering" or "preach":
        return context
    else:
        return ""
    
def situation_prompt_input_generator():
    print("******* Generate Situations *******")
    situation = []
    progress_bar = tqdm(range(len(challenges)*2), position=0, leave=True)
    for i in range(2):
        for challenge in challenges:
            situation_prompt = f"你是使用者，請簡短敘述一個你所面對到{challenge}的困難，並且你完全不知道該怎麼辦"
            situation_output = chinese_converter.to_traditional(inference(situation_prompt))\
                                        .replace("作為一個使用者，","")\
                                        .replace("作為使用者，", "")\
                                        .replace("\n", "").replace("麵", "面")
            situation.append(situation_output)
            progress_bar.update(1)
    return situation

def random_split_scope(scope, task_type=None):
    if task_type == "question_answering" or task_type == "preach":
        return ["".join(scope["context"].tolist())]
    
    slen = len(scope)
    try:
        splitat = random.sample(range(slen), 2)
    except:
        splitat = []

    splitat.sort()
    splitat.insert(0, 0)

    for i in range(1, len(splitat)):
        if splitat[i] - splitat[i-1] < 5:
            splitat[i] = splitat[i-1] + 5

    splitat = list(filter(lambda x: x<slen, splitat))
    splitat.append(slen)
    scope_ls = ["".join(scope[splitat[i]:splitat[i+1]]["context"].tolist()) for i in range(len(splitat)-1)]
    scope_ls.append("".join(scope_ls))
    return scope_ls


def main():
    situation = situation_prompt_input_generator()

    bible = pd.read_json(path_or_buf="./dataset/processed_data/raw_bible.jsonl", lines=True)
    with open("./dataset/bible.yaml", "r") as f:
        bible_metadata = yaml.safe_load(f)

    tuning_data = []
    counter = {"question_answering":0, "preach":0, "pray":0, "consult":0}

    # Data with scripture input
    print("******* Data with scripture input *******")
    progress_bar = tqdm(range(3136), position=0, leave=True)
    for task_type in ["question_answering", "preach", "pray", "consult"]:
        book_scope = bible_metadata["books"][bible_metadata["books"].index("Matthew"):] \
                        if task_type == "pray" or task_type == "consult" else bible_metadata["books"]
        if task_type == "pray" or task_type == "consult":
            book_scope.append("Psalms")

        for book in book_scope:
            for chap in range(1, bible_metadata["num_chapters"][book]+1):
                scope = bible.loc[bible.apply(lambda x: f"{book}:{chap}:" in x["id"], axis=1)]
                scope_ls = random_split_scope(scope, task_type=task_type)
                for input_context in scope_ls:
                    sits = random.sample(situation, 2) if task_type == "pray" or task_type == "consult" else [None]

                    for sit in sits:
                        prompt_input = prompt_input_generator(tp=task_type, context=input_context, situation=sit)
                        instr = random.choice(INSTRUCTION[task_type])
                        prompt = f"### INSTRUCTION: {instr} ### INPUT: {prompt_input}"
                        output = chinese_converter.to_traditional(inference(prompt).replace("\n", "")).replace("麵", "面")
                        
                        if task_type == "question_answering":
                            qa_pairs = json.loads(output)
                            for qa in qa_pairs:
                                instr = qa["question"]
                                output = qa["answer"]
                                prompt = f"### INSTRUCTION: {instr} ### INPUT: {prompt_input}"
                                tuning_data.append({"id": str(uuid.uuid4()), "instruction": instr, "input": prompt_input, "output": output})
                                counter[task_type] += 1
                        else:
                            tuning_data.append({"id": str(uuid.uuid4()), "instruction": instr, "input": prompt_input, "output": output})
                            counter[task_type] += 1
                
                progress_bar.update(1)

    # data without input prompt (pray and consult)
    print("******* Data without scripture input *******")
    progress_bar = tqdm(range(len(situation)*2), position=0, leave=True)
    for task_type in ["pray", "consult"]:
        for sit in situation:
            prompt_input = prompt_input_generator(tp=task_type, context=None, situation=sit)
            instr = INSTRUCTION_NO_INPUT[task_type]
            prompt = f"### INSTRUCTION: {instr} ### INPUT: {prompt_input}"
            output = chinese_converter.to_traditional(inference(prompt).replace("\n", "")).replace("麵", "面")
            tuning_data.append({"id": str(uuid.uuid4()), "instruction": instr, "input": prompt_input, "output": output})
            counter[task_type] += 1
            progress_bar.update(1)

    # write to file
    print("******* Writing to File *******")
    with open("./dataset/processed_data/train.json", "w") as f:
        json.dump(tuning_data, f, indent=4)
    with open("./dataset/processed_data/train_metadata.json", "w") as f:
        json.dump(counter, f, indent=4)

if __name__ == "__main__":
    main()