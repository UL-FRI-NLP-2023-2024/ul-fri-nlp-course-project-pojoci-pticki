import openai
from dotenv import load_dotenv
import os
import re
import pandas as pd
import time

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_chatgpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",  # Use the appropriate model
        messages=[
            {"role": "system", "content": "You are an academic researcher specializing in linguistics."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,             # Adjust max tokens according to your needs
        temperature=0.0             # Adjust the creativity level of the response
    )

    return response['choices'][0]['message']['content'].strip()

valid_categories = ["UX", "Social", "Procedure", "Deliberation", "Seminar", "Imaginative Entry", "Others"]
def extract_info(response_text):
    # Regular expression to find category from a list of valid categories
    category_match = re.search(r'category\s*:\s*(' + '|'.join(re.escape(cat) for cat in valid_categories) + r')', response_text, re.IGNORECASE)

    # Extract and clean up the category
    if category_match:
        category = category_match.group(1).strip()
    else:
        category = "NOLABEL"

    return category

# def save_to_file(filename, data_list):
#     with open(filename, 'w') as file:
#         for item in data_list:
#             file.write(item + '\n')


def sample_and_concatenate(df, categories, num_samples=2):
    sampled_lines = []
    
    for category in categories:
        # Sample rows for the current category
        num_samples = num_samples_dict.get(category)
        sampled_df = df[df['Discussion Type'] == category].sample(n=num_samples, replace=True, random_state=33)
        for _, row in sampled_df.iterrows():
            sampled_lines.append(f"Message: {row['Message']}; Category: {row['Discussion Type']}")
    
    # Concatenate sampled lines into a single string
    concatenated_output = "\n".join(sampled_lines)
    return concatenated_output

num_samples_dict = {
    "UX": 1,
    "Social": 2,
    "Procedure": 2,
    "Deliberation": 2,
    "Seminar": 4,
    "Imaginative Entry": 2,
    "Others": 1
}

a=os.getcwd()
df = pd.read_csv('./data/final_data.csv')
df = df.dropna()
print(df.iloc[:, 1].unique())

with open('./prompts/text_categorization_prompt_fewshot.md', 'r', encoding='utf-8') as file:
    prompt = file.read()

fewshot_examples = sample_and_concatenate(df, valid_categories)

with open('./data/chat_gpt3.5_fewshot_categories.txt', 'w', encoding='utf-8') as file_cat:
    categories = []
    for index, row in df.iterrows():
        input_text = str(row['Message'])
        give_chat = prompt.format(input=input_text, examples=fewshot_examples)
        response = get_chatgpt_response(give_chat)
        category = extract_info(response)
        file_cat.write(str(category) + '\n')
        print(category)
        time.sleep(0.1)


#save_to_file('./data/chat_gpt3.5_zeroshot_explanations.txt', explanations)
#save_to_file('./data/chat_gpt3.5_zeroshot_categories.txt', categories)
