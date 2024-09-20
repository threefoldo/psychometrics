import re, csv
import shutil
import time, random
import gradio as gr
import pandas as pd
from partial_json_parser import loads
from openai import OpenAI
from jinja2 import Template
import logging
from datetime import datetime

seed = 42
timestamp = datetime.now().strftime('%Y%m%d%H%M')
log_file = f'output_{seed}_{timestamp}.log'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file)])


prompt_system = '''
You are an AI assistant specialized in user behavior analysis and prediction. Your task is to create a detailed user persona 
based on provided survey responses, then predict how this persona would respond to a new question. Provide your analysis 
and prediction in a structured JSON format.
'''

question_prompt = Template('''
Your task is to analyze survey responses, create a user persona, and predict a response to a new question. Follow these steps:

1. Review the provided survey questions and responses:
<survey_responses>
{{survey_responses}}
</survey_responses>

2. Based on these responses, construct or update a comprehensive user persona. Consider:
   - Demographics (age, gender, occupation, location, etc.)
   - Psychographics (values, attitudes, interests, lifestyle)
   - Behavioral patterns and preferences
   - Potential pain points or motivations
   - Any other relevant characteristics that can be inferred
<user_persona>
{{user_persona}}
</user_persona>

3. Analyze the new question and its options:
<new_question>
{{new_question}}
</new_question>
<options>
{{options}}
</options>

4. Consider how your constructed persona would approach this new question:
   - Which aspects of the question would resonate most with them?
   - How do their established traits and preferences influence their perspective?
   - What factors would be most important in their decision-making process?

5. Predict the most likely response the persona would give, ensuring consistency with their established profile 
and previous answers. If the survey responses are empty, generate a random answer for the question. 
The final answer shoule be only one digit chosen from provided options. 

6. Provide your full analysis and prediction in the following JSON format:
```json
{
  "user_persona": "A paragraph providing a comprehensive, concrete description of the imagined user persona, including all relevant demographic info, personality traits, shopping preferences and habits that can be gleaned from their provided answers. Avoid relying on implicit assumptions and aim to create a detailed, fleshed-out user profile.",
  "reasoning": "A paragraph outlining the step-by-step logical reasoning and thought process of how this specific user persona would approach the target question based on their established traits and preferences. Explain how these factors would influence them to ultimately select a particular answer from the given options.", 
  "answer": "A single digit representing their selected answer, ensuring it is consistent with both their persona and their answers to the other questions, and falls within the scope of the provided options."
}
```

Ensure that your analysis is well-reasoned, detailed, and consistently aligned with the information provided in the survey responses. 
If there's ambiguity or lack of information in certain areas, acknowledge this and explain how it affects your prediction.
''')

API_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_CHOICES = [
    "openai/gpt-3.5-turbo",
    "openai/gpt-4o",
    "meta-llama/llama-3.1-70b-instruct",
    "anthropic/claude-3.5-sonnet"
]
DEFAULT_MODEL = "openai/gpt-4o"

json_pattern = re.compile(r'```json\n(.*?)```', re.DOTALL)
answer_pattern = re.compile(r'answer[\'"\s]*[:=][\s\'"]*(\d+)')
csv_columns = ['contract_id', 'name', 'question_id', 'question', 'options']

def get_response_text(questions, answered):
    text = []
    for question in answered:
        answer = answered[question]
        group_name, options = questions[question]['name'], questions[question]['options']
        content = f'group: {group_name}\nquestion: {question} \noptions: {options} \t response: {answer}\n'
        text.append(content)
    return '\n'.join(text)

def validate_csv(filename):
    required_columns = csv_columns
    df = pd.read_csv(filename)
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file is missing required columns: contract_id, name, question_id, question, options")
    return df


def ask_llm(client, model, content, temperature=0.7):
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=float(temperature),
            seed=seed,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": content}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        return None


def prepare_question_prompt(all_questions, answered, new_question, user_persona):
    txt_response = get_response_text(all_questions, answered)
    message = question_prompt.render(survey_responses=txt_response, user_persona=user_persona, 
                                        new_question=new_question['question'], options=new_question['options'])
    prompt = f'<s>[INST]<<SYS>>{prompt_system}<</SYS>>\n{message} [/INST]'
    log = f"\n--> Processing question: {new_question['contract_id']} - {new_question['question_id']} {new_question['question']}\n"
    return prompt, log

def parse_response(answered, new_question, response):
    log, answer_value = '', ''
    user_persona = None
    if response is None:
        log = f"No response received for question {new_question['question_id']}\n"
        return user_persona, log
    try:
        json_text = response
        match_json = json_pattern.search(response)
        if match_json:
            json_text = match_json.group(1)
            print(f'matched: {json_text}')
        response_json = loads(json_text)
        user_persona = response_json.get('user_persona', '')
        answer_value = response_json.get('answer', '')
    except:
        match = answer_pattern.search(response)
        if match:
            answer_value = match.group(1)
        print(f'failed to parse: {response}')
    answered[new_question['question']] = answer_value
    log += f"Answer generated: {answer_value}\n"
    if user_persona is not None:
        log += f"User persona: {user_persona}\n"
    # else:
    #     log += f"Full response: {response}\n"
    return user_persona, log


def process_csv(filename, api_key, model, temperature, total_numbers):
    if api_key == '': 
        yield 'API key is required', 'result.csv'
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    logging.info(f"Processing with model: {model}, temperature: {temperature}\n")
    logging.info(f"Generate {total_numbers} cases")

    df = validate_csv(filename)
    questions_list = df.to_dict('records')
    qid_ordered = [q['question_id'] for q in questions_list]
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    output_filename = f"results_{timestamp}_0.csv"
    
    # Create the empty CSV file
    empty_df = pd.DataFrame(columns=qid_ordered)
    empty_df.to_csv(output_filename, index=False)
    
    # Group questions by contract_id
    grouped_questions = {}
    all_questions = {}
    for question in questions_list:
        all_questions[question['question']] = question
        contract_id = question['contract_id']
        if contract_id not in grouped_questions:
            grouped_questions[contract_id] = []
        grouped_questions[contract_id].append(question)

    # Process groups in random order
    group_order = list(grouped_questions.keys())
    random.shuffle(group_order)
    logging.info(f"Ordered groups: {group_order}")

    log = ''
    for i in range(total_numbers):
        logging.info(f"--> sample: {i}")
        
        answered = {}
        user_persona = {}
        for contract_id in group_order:
            group = grouped_questions[contract_id]
            yield f"Processing group: {contract_id}", None
            
            # Randomly shuffle questions within the group
            random.shuffle(group)
            
            for new_question in group:
                # # first question will be answered randomly
                # if len(answered) < 1:
                #     answer = random.choice(range(1, 7))
                #     answered[new_question['question']] = answer
                #     continue
                prompt, msg = prepare_question_prompt(all_questions, answered, new_question, user_persona)
                log += msg
                yield log, output_filename
                
                logging.info(f"prompt: {prompt}\n")
                response = ask_llm(client, model, prompt, temperature)
                logging.info(f"response: {response}")
                new_persona, msg = parse_response(answered, new_question, response)
                if new_persona is not None:
                    user_persona = new_persona
                log += msg
                yield log, output_filename
                    
                time.sleep(0.1)
                
        logging.info(f"** answered **: {answered}")
        log += "Processing complete.\n"
        
        report = {}
        # convert answered to a matrix
        for q in answered:
            question = all_questions[q]
            report[question['question_id']] = answered[q]
        new_output_file = f"results_{timestamp}_{i+1}.csv"
        shutil.copy(output_filename, new_output_file)
        with open(new_output_file, 'a', newline='\n') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([str(report[qid]) for qid in qid_ordered])
            
        output_filename = new_output_file
        yield log, new_output_file
    
    # return log, output_filename

iface = gr.Interface(
    fn=process_csv,
    inputs=[
        gr.File(label="Upload CSV file"),
        gr.Textbox(label="API Key (OpenRouter)", type="password"),
        gr.Dropdown(
            label="LLM Model",
            choices=MODEL_CHOICES,
            value=DEFAULT_MODEL
        ),
        gr.Slider(label="Temperature", minimum=0, maximum=1, step=0.1, value=0),
        gr.Slider(label="Total cases", minimum=1, maximum=500, step=1, value=1)
    ],
    outputs=[
        gr.Textbox(label="Execution Log", lines=30),
        gr.File(label="Download Results in CSV")
    ],
    examples = [
        ["data/questions.csv", "", "openai/gpt-4o", 0.1, 3]
    ],
    title="User Survey Generator",
    description="Upload a CSV file with questions (contract_id, name, question_id, question). Provide API key and model settings. The AI will generate persona-based responses for each question."
)

iface.queue()
iface.launch(share=True)
