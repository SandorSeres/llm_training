#https://pyimagesearch.com/2024/01/01/introduction-to-gemini-pro-vision/

import pathlib
import textwrap
#!pip install -q -U google-generativeai
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import PIL.Image
import urllib.request 
from PIL import Image

GOOGLE_API_KEY= ""
genai.configure(api_key=GOOGLE_API_KEY)

#lists and prints the names of available models
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)

def to_markdown(text):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))

# Opening the image for Image Understanding
urllib.request.urlretrieve('https://i.imgur.com/RxsIzEy.png', "comic.png") 
image = PIL.Image.open('comic.png')

model = genai.GenerativeModel("gemini-pro-vision")
response = model.generate_content(image)
to_markdown(response.text)


'''
Generate content with a specific prompt (“Write an explanation based on the image, give the name of the author and the book that it is from”) and the image. The stream=True
parameter indicates that the response is streamed, and response.resolve()
waits for the completion of this streaming response. 
'''
response = model.generate_content(
    ["Write an explanation based on the image, give the name of the author and the book that it is from", image],
    stream=True
)
response.resolve()
to_markdown(response.text)
