import os
from dotenv import load_dotenv
from langchain.agents import create_agent


load_dotenv()


agent = create_agent("gpt-5")

result = agent.invoke({"messages": [{"role": "user", "content": "Hi"}]})

print(result)
