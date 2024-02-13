import logging
import sys
import os
import warnings
from langchain.embeddings import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import json
import torch
from pathlib import Path
import pandas as pd

NUMEXPR_MAX_THREADS = 8
from copy import deepcopy
# transformers
from transformers import BitsAndBytesConfig
# llama_index
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index import download_loader, VectorStoreIndex, ServiceContext
from llama_index.node_parser import SentenceSplitter
from langchain.embeddings import HuggingFaceEmbeddings

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


torch.cuda.empty_cache()
################################################
#Load Data
################################################
data_path = Path('./data')
contents = os.listdir(data_path)
# create chunks
node_parser = SentenceSplitter(chunk_size=512)
PDFReader = download_loader("PDFReader")
loader = PDFReader()
nodes = []
for item in contents:
    item_path = data_path / item
    if item_path.is_file():  # Ellenőrizzük, hogy az elem fájl-e
        doc = loader.load_data(file=item_path)
        nodes.append(node_parser.get_nodes_from_documents(doc))
    else:
        print(f"Skipping directory: {item_path}")  # Mappák kihagyása

flat_list = [item for sublist in nodes for item in sublist]
################################################

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def messages_to_prompt(messages):
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt

MODEL= "mistralai/Mistral-7B-Instruct-v0.1"

llm = HuggingFaceLLM(
    model_name=MODEL,
    tokenizer_name=MODEL,
    query_wrapper_prompt=PromptTemplate("<|system|>\n\n<|user|>\n{query_str}\n<|assistant|>\n"),
    context_window=4096,
    max_new_tokens=2048,
    model_kwargs={"quantization_config": quantization_config},
    generate_kwargs={"temperature": 0.1, "do_sample":False},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",
)

#Load Open Embedding 
# multilingual sentence transformer for embedding.
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ServiceContext
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model )

# vector vectorstore
vector_index = VectorStoreIndex(
    flat_list, service_context=service_context
)

query_str = ["melyik dokumentumban említik a Da Vinci szót? Magyarul!",
             "Említik a genetikus algoritmust? Magyarul!",
             "Mivel foglalkozik az LP Solution? Magyarul!"
            ]

query_engine = vector_index.as_query_engine()
for q in query_str:
  response = query_engine.query(q)
  print(response)
  for i in response.metadata.keys():
    print(response.metadata[i])
    # print(response.source_nodes.score)
