from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "ramsrigouthamg/t5_paraphraser"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

input_text = """
Artificial intelligence has become one of the most important technologies in modern computer science. It focuses on developing systems that can perform tasks such as learning, reasoning, problem-solving, and decision-making, which traditionally required human intelligence. With the rapid growth of data and computational power, machine learning and deep learning techniques have enabled significant advancements in fields like healthcare, finance, education, and transportation. Despite its benefits, artificial intelligence also raises concerns related to ethics, data privacy, bias, and the impact of automation on employment. Therefore, responsible development and deployment of AI systems are essential to ensure they benefit society as a whole.
"""

text = "paraphrase: " + input_text + " </s>"

input_ids = tokenizer.encode(
    text,
    return_tensors="pt",
    max_length=128,
    truncation=True
).to(device)

outputs = model.generate(
    input_ids,
    max_length=128,
    num_beams=5,
    num_return_sequences=1,
    temperature=1.0,
    repetition_penalty=1.5
)

paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Original:", input_text)
print("Paraphrased:", paraphrased_text)

