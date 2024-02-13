import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = 'sk-NjuQfd.....'
os.environ["SERPAPI_API_KEY"] = 'dd738e1b43b41eea8906......7'

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True)

agent.run("Analyze OTP.BD stock and craft investment recommendations. Lets do it step by step")

