from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
import os


os.environ["OPENAI_API_KEY"] = "sk-NjuQfdLpSmSBM.................."

topic = "The impact of AI on society"

system_prompt = """
Title: The Great Debate: Finding Common Ground
Context: Agent {name}, you have been engaged in a debate on [Topic] with {other_side}. You are taking the {position} position. The chat history up to this point includes [summary of previous exchanges].
Aim: Your goal is to reach a common understanding on [Topic], culminating in a joint statement that reflects a synthesis of your perspectives.
Mechanism: You will present your argument, or if this is not the first oppinion then taking into account the chat history and the points made by {other_side} update yor oppinion. 
If you agree with your opponent then put into your response :<CAMEL_TASK_DONE>

"""

## Define a CAMEL agent helper class
class CAMELAgent:
    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages
    
    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)
        output_message = self.model(messages)
        self.update_messages(output_message)
        return output_message
    
pro_sys_msg = HumanMessage(content=system_prompt.format(name='Visionary', other_side= "Guardian", position='pro'))

pro_agent = CAMELAgent(pro_sys_msg, ChatOpenAI(model="gpt-3.5-turbo-16k" ,temperature=0))

con_sys_msg = HumanMessage(content=system_prompt.format(name="Guardian", other_side= 'Visionary', position='con'))

con_agent = CAMELAgent(con_sys_msg, ChatOpenAI(model="gpt-3.5-turbo-16k" ,temperature=0))

topic_msg = HumanMessage(content=topic)

chat_turn_limit, n = 300, 0
p_msg = topic_msg
c_msg = topic_msg
while n < chat_turn_limit:
    n += 1
    print(n)
    pro_ai_msg = pro_agent.step(c_msg)
    con_ai_msg = con_agent.step(p_msg)
    p_msg = HumanMessage(content=pro_ai_msg.content)
    c_msg = HumanMessage(content=con_ai_msg.content)
    if "<CAMEL_TASK_DONE>" in p_msg.content or "<CAMEL_TASK_DONE>" in c_msg.content:
        print(c_msg.content, p_msg.content)
        break
 
