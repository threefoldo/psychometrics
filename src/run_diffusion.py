import os
import json
from partial_json_parser import loads
from dotenv import load_dotenv
from openai import OpenAI
from jinja2 import Template
import random
import numpy as np
import pandas as pd
import logging
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

total_samples = 500
seed = 42
temperature = 0
model = 'meta-llama/llama-2-13b-chat'
timestamp = datetime.now().strftime('%Y%m%d%H%M')
output_file = f'output/{model.replace("/", "-")}_s{seed}t{temperature}_{timestamp}'
log_file = f'output/{model.replace("/", "-")}_{timestamp}.log'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file)])

client = OpenAI(base_url="https://openrouter.ai/api/v1", 
               api_key=os.getenv("OPENROUTER_API_KEY"))
data = json.load(open('../data/questions_21.json'))
questions, g2q = data['questions'], data['g2q']
labels = list(questions.keys())


prompt_system = '''
Given a list of questions and their corresponding answers about a user's Amazon shopping experience, imagine a detailed persona of a user who would provide those exact answers. Be sure to include key demographic information, relevant personality traits, shopping habits and preferences, and any other pertinent details that can be inferred from the provided answers.
Then, carefully consider the target question that is asking for this user's likely response. Think step-by-step about how the imagined user, based on their established persona, would approach and answer this target question. Explain the logical reasoning of how their unique traits and shopping behaviors would lead them to select a particular answer from the given options.
Output the full result in the following JSON format:
{
  "user_description": "A paragraph providing a comprehensive, concrete description of the imagined user persona, including all relevant demographic info, personality traits, shopping preferences and habits that can be gleaned from their provided answers. Avoid relying on implicit assumptions and aim to create a detailed, fleshed-out user profile.",
  "reasoning": "A paragraph outlining the step-by-step logical reasoning and thought process of how this specific user persona would approach the target question based on their established traits and preferences. Explain how these factors would influence them to ultimately select a particular answer from the given options.", 
  "answer": "A single digit representing their selected answer, ensuring it is consistent with both their persona and their answers to the other questions, and falls within the scope of the provided options."
}

'''

param_exp = {
    'model': model,
    'seed': seed,
    'temperature': temperature,
    'system': prompt_system
}

user_message = Template('''
Here are a list of questions along with numerical answers:
<question_and_answers>
{{question_and_answers}}
</question_and_answers>

And here is a target question:
<target_question>
{{target_question}}
</target_question>

Please answer the target question, providing your full output in the JSON format specified previously.
''')


def ask_llm(question, params):
    completion = client.chat.completions.create(
      model=params['model'],
      seed=params['seed'],
      temperature=params['temperature'],
      response_format={ "type": "json_object" },
      messages=[
        # {"role": "system", "content": params['system']},
        {"role": "user", "content": question}
      ]
    )

    content = completion.choices[0].message.content
    # print(content)
    return content


def output_qa(questions, answered):
    text = []
    for label in answered:
        q = questions[label]
        answer = answered[label]
        content = f'question: {q["question"]}\n options: {q["options"]}\n answer: {answer}\n'
        text.append(content)
    return '\n'.join(text)

def output_target(questions, label):
    q = questions[label]
    return f'question: {q["question"]}\n options: {q["options"]}\n'


answer_pattern = re.compile(r'answer[\'"\s]*[:=][\s\'"]*(\d+)')
                     
def answer_by_llm(answered, target):
    txt_q = output_qa(questions, answered)
    txt_t = output_target(questions, target)
    message = user_message.render(question_and_answers=txt_q, target_question=txt_t)
    answer = {}
    prompt = f'<s>[INST]<<SYS>>{prompt_system}<</SYS>>\n{message} [/INST]'
    # for k in range(3):
    try:
        r = ask_llm(prompt, param_exp)
        match = answer_pattern.search(r)
        if match:
            answer['answer'] = match.group(1)
            answer['raw'] = r
        else:
            logging.error(f'Failed to answer: {target} {prompt} {r}')
    except Exception as e:
        logging.error(f'Failed to call API: {target}')
        logging.error(f'Error message: {str(e)}')
        # pos1, pos2 = r.find('{'), r.find('}')
        # answer = loads(r[pos1 : pos2+1])
    
    answer['label'] = target
    # if 'user_description' not in answer: answer['user_description'] = ''
    # if 'reasoning' not in answer: answer['reasoning'] = ''
    if 'answer' not in answer: answer['answer'] = 0
    return answer


def generate_sample(i, labels, questions, g2q):
    answered = {}
    logging.info(f'---> sample {i}')
    sample_matrix = np.zeros(len(labels), dtype=int)
    
    while len(answered) < len(labels):
        # Select a random question that hasn't been answered yet
        remaining_questions = list(set(labels) - set(answered.keys()))
        random_label = random.choice(remaining_questions)
        col_index = labels.index(random_label)
        
        random_question = questions[random_label]
        # Generate a random answer (1-7) for the selected question
        answer = random.choice(random_question['values'])
        answered[random_label] = answer
        sample_matrix[col_index] = answer
        
        # Find the group of the selected question
        group = random_question['group']
        logging.info(f'Sample {i}: {random_label} {group} {answer}')
        
        # Answer all questions in the same group
        group_labels = g2q[group].copy()
        random.shuffle(group_labels)
        for label in group_labels:
            if label not in answered:
                answer = answer_by_llm(answered, label)
                try:
                    answer_value = int(answer['answer'])
                except:
                    logging.error(f'invalid answer: {answer["answer"]}')
                    answer_value = 0
                answered[label] = answer_value
                logging.info(f'Sample {i}: {label} {answer}')
                col_index = labels.index(label)
                sample_matrix[col_index] = answer_value
                
    return i, answered, sample_matrix

def main(total_samples, labels, questions, g2q, output_file):
    matrix = np.zeros((total_samples, len(labels)), dtype=int)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(generate_sample, i, labels, questions, g2q) for i in range(total_samples)]
        
        with open(output_file + '_b.jsonl', 'w') as fp:
            for future in as_completed(futures):
                i, answered, sample_matrix = future.result()
                matrix[i] = sample_matrix
                print(sample_matrix, answered)
                fp.write(json.dumps(answered) + '\n')
                fp.flush()
    df = pd.DataFrame(matrix, columns=labels)
    df.to_csv(output_file + '_a.csv', index=False)

# Assuming labels, questions, g2q, total_samples, and output_file are already defined
main(total_samples, labels, questions, g2q, output_file)


