# https://huggingface.co/timpal0l/mdeberta-v3-base-squad2
"""
DeBERTa improves the BERT and RoBERTa models using disentangled attention and enhanced mask decoder. With those two improvements, DeBERTa out perform RoBERTa on a majority of NLU tasks 
"""
from transformers import pipeline

qa_model = pipeline("question-answering", "timpal0l/mdeberta-v3-base-squad2")
question = "Hol élek?"
context = "A nevem Seres Sándor, és Budapesti vagyok"
resp= qa_model(question = question, context = context)
print(resp)
# {'score': 0.4942767918109894, 'start': 24, 'end': 34, 'answer': ' Budapesti'}

context = "A trigonometria egy matematikai részhalmaz"
resp= qa_model(question = question, context = context)
print(resp)
#{'score': 1.1617652262430056e-07, 'start': 0, 'end': 15, 'answer': 'A trigonometria'}