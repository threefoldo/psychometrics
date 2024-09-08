import gradio as gr
import pandas as pd
from partial_json_parser import loads
from openai import OpenAI
import time, random
from jinja2 import Template

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

question_prompt = Template('''
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


def output_qa(answered):
    text = []
    for question in answered:
        answer = answered[question]
        content = f'question: {question}\n answer: {answer}\n'
        text.append(content)
    return '\n'.join(text)

def output_target(question):
    return f'question: {question}\n'



def process_csv(file, api_key, model, temperature):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    log = f"Processing with model: {model}, temperature: {temperature}\n"
    yield log

    def ask_llm(question):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=float(temperature),
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": question}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            return None

    df = pd.read_csv(file.name)
    questions = df.to_dict('records')
    
    # Group questions by ContractID
    grouped_questions = {}
    for question in questions:
        contract_id = question['contract_id']
        if contract_id not in grouped_questions:
            grouped_questions[contract_id] = []
        grouped_questions[contract_id].append(question)

    # Process groups in random order
    group_order = list(grouped_questions.keys())
    random.shuffle(group_order)

    answered = {}
    for contract_id in group_order:
        group = grouped_questions[contract_id]
        yield f"Processing group: {contract_id}"
        
        # Randomly shuffle questions within the group
        random.shuffle(group)
        
        for question in group:
            txt_q = output_qa(answered)
            txt_t = output_target(question)
            message = question_prompt.render(question_and_answers=txt_q, target_question=txt_t)
            message = question_prompt.render(question_and_answers = question['question'])
            prompt = f'<s>[INST]<<SYS>>{prompt_system}<</SYS>>\n{message} [/INST]'
            log += f"Processing question: {contract_id} - {question['question_id']}\n"
            yield log

            response = ask_llm(prompt)
            
            if response:
                try:
                    response_json = loads(response)
                    answer_value = response_json.get('answer', 'N/A')
                    answered[question['question']] = answer_value
                    log += f"Answer generated: {answer_value}\n"
                except:
                    log += f"Error parsing JSON for question {question['question_id']}: {response}\n"
            else:
                log += f"No response received for question {question['question_id']}\n"
            yield log
                
            time.sleep(0.1)

    log += "Processing complete.\n"
    yield log

iface = gr.Interface(
    fn=process_csv,
    inputs=[
        gr.File(label="Upload CSV file"),
        gr.Textbox(label="API Key", type="password"),
        gr.Dropdown(
            label="LLM Model",
            choices=["openai/gpt-3.5-turbo", "openai/gpt-4o", "meta-llama/llama-2-13b-chat", "anthropic/claude-3.5"],
            value="meta-llama/llama-2-13b-chat"
        ),
        gr.Slider(label="Temperature", minimum=0, maximum=1, step=0.1, value=0)
    ],
    outputs=gr.Textbox(label="Execution Log", lines=30),
    title="User Survey Generator",
    description="Upload a CSV file with questions (contract_id, name, question_id, question). Provide API key and model settings. The AI will generate persona-based responses for each question."
)

iface.queue()
iface.launch()
