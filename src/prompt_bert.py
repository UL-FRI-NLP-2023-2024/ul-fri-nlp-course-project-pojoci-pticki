from transformers import BertForQuestionAnswering, BertTokenizer
import torch

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

context = "You are an asshole!"
question = "You must answer with only one of the following emotions that best represents the sentence: Emotions are positive, neutral, negative."

inputs = tokenizer(question, context, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

start_scores = outputs.start_logits
end_scores = outputs.end_logits

start_idx = torch.argmax(start_scores)
end_idx = torch.argmax(end_scores) + 1  # +1 to include the end token

answer_tokens = inputs['input_ids'][0][start_idx:end_idx]
answer = tokenizer.decode(answer_tokens)

print(f"Answer: {answer}")
