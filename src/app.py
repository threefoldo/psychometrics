import gradio as gr
import pandas as pd
from partial_json_parser import loads
from openai import OpenAI
import time, random
from jinja2 import Template


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

3. Analyze the new question:
<new_question>
{{new_question}}
</new_question>

4. Consider how your constructed persona would approach this new question:
   - Which aspects of the question would resonate most with them?
   - How do their established traits and preferences influence their perspective?
   - What factors would be most important in their decision-making process?

5. Predict the most likely response the persona would give, ensuring consistency with their established profile and previous answers.

6. Provide your full analysis and prediction in the following JSON format:

{
  "user_persona": {
    "demographics": "Key demographic information inferred from survey responses",
    "psychographics": "Relevant attitudes, values, and lifestyle factors",
    "behaviors": "Notable behavioral patterns or preferences",
    "motivations": "Primary motivations or pain points influencing decisions"
  },
  "persona_analysis": "A detailed paragraph describing the user persona, synthesizing the information from the above categories. Explain how you've drawn these conclusions from the survey responses.",
  "question_analysis": "A paragraph examining the new question from the persona's perspective. Discuss which aspects would be most relevant or impactful for this user.",
  "response_prediction": {
    "predicted_response": "The predicted response to the new question",
    "confidence_level": "High/Medium/Low, based on how well the persona aligns with the new question",
    "reasoning": "A thorough explanation of why this response was predicted, referencing specific aspects of the persona and their previous responses"
  }
}

Ensure that your analysis is well-reasoned, detailed, and consistently aligned with the information provided in the survey responses. 
If there's ambiguity or lack of information in certain areas, acknowledge this and explain how it affects your prediction.
''')

API_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_CHOICES = [
    "openai/gpt-3.5-turbo",
    "openai/gpt-4o",
    "meta-llama/llama-2-13b-chat",
    "anthropic/claude-3.5"
]
DEFAULT_MODEL = "meta-llama/llama-2-13b-chat"


def get_response_text(answered):
    text = []
    for question in answered:
        answer = answered[question]
        content = f'question: {question} \t response: {answer}\n'
        text.append(content)
    return '\n'.join(text)

def validate_csv(filename):
    required_columns = ['contract_id', 'name', 'question_id', 'question']
    df = pd.read_csv(filename)
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file is missing required columns")
    return df

def process_csv(filename, api_key, model, temperature):
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    log = f"Processing with model: {model}, temperature: {temperature}\n"
    yield log

    def ask_llm(content):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=float(temperature),
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

    df = validate_csv(filename)
    questions = df.to_dict('records')
    
    # Group questions by contract_id
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
    user_persona = {}
    for contract_id in group_order:
        group = grouped_questions[contract_id]
        yield f"Processing group: {contract_id}"
        
        # Randomly shuffle questions within the group
        random.shuffle(group)
        
        for question in group:
            
            # first question will be answered randomly
            if len(answered) < 1:
                answer = random.choice(range(1, 7))
                answered[question['question']] = answer
                continue
            
            txt_response = get_response_text(answered)
            message = question_prompt.render(survey_responses=txt_response, user_persona=user_persona, new_question=question['question'])
            prompt = f'<s>[INST]<<SYS>>{prompt_system}<</SYS>>\n{message} [/INST]'
            log += f"\n--> Processing question: {contract_id} - {question['question_id']} {question['question']}\n"
            yield log

            response = ask_llm(prompt)
            
            if response:
                try:
                    response_json = loads(response)
                    user_persona = response_json.get('user_persona', {})
                    response_prediction = response_json.get('response_prediction', {})
                    answer_value = response_prediction.get('predicted_response', '')
                    answered[question['question']] = answer_value
                    log += f"Answer generated: {answer_value}\n"
                    log += f"User persona: {user_persona}\n"
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
            choices=MODEL_CHOICES,
            value=DEFAULT_MODEL
        ),
        gr.Slider(label="Temperature", minimum=0, maximum=1, step=0.1, value=0)
    ],
    outputs=gr.Textbox(label="Execution Log", lines=30),
    title="User Survey Generator",
    description="Upload a CSV file with questions (contract_id, name, question_id, question). Provide API key and model settings. The AI will generate persona-based responses for each question."
)

iface.queue()
iface.launch()
