import os, getpass
import asyncio
import nest_asyncio
from typing import List
from dotenv import load_dotenv
import logging

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool as langchain_tool
from langgraph.prebuilt import create_react_agent

# Load environment variables from .env file
load_dotenv()

# UNCOMMENT if you want to enter the API key manually
# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

try:
   # A model with function/tool calling capabilities is required.
   # 需要一个具有函数调用能力的模型，这里使用 Gemini 2.0 Flash。
   llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
   print(f"✅ Language model initialized: {llm.model_name}")
except Exception as e:
   print(f"🛑 Error initializing language model: {e}")
   llm = None

# --- Define a Tool ---
# --- 定义模拟的搜索工具 ---
@langchain_tool
def search_information(query: str) -> str:
   """
   Provides factual information on a given topic. Use this tool to find answers to phrases
   like 'capital of France' or 'weather in London?'.
   # 模拟提供关于特定查询的输出。使用此工具查找类似「法国的首都是哪里？」或「伦敦的天气如何？」这类问题的答案。
   """
   print(f"\n--- 🛠️ Tool Called: search_information with query: '{query}' ---")
   # Simulate a search tool with a dictionary of predefined results.
   # 通过一个字典预定义的结果来模拟搜索工具。
   simulated_results = {
       "weather in london": "The weather in London is currently cloudy with a temperature of 15°C.",
       "capital of france": "The capital of France is Paris.",
       "population of earth": "The estimated population of Earth is around 8 billion people.",
       "tallest mountain": "Mount Everest is the tallest mountain above sea level.",
       "default": f"Simulated search result for '{query}': No specific information found, but the topic seems interesting."
   }
   result = simulated_results.get(query.lower(), simulated_results["default"])
   print(f"--- TOOL RESULT: {result} ---")
   return result

tools = [search_information]

# --- Create a Tool-Calling Agent ---
# --- 创建一个使用工具的智能体 ---
if llm:
   # Create the agent using langgraph's create_react_agent.
   # This returns a compiled graph that can be invoked directly.
   # 使用 langgraph 的 create_react_agent 创建智能体。
   # 这将返回一个可以直接调用的编译图。
   agent_executor = create_react_agent(llm, tools)

async def run_agent_with_tool(query: str):
   """
   Invokes the agent executor with a query and prints the final response.
   执行智能体并打印最终输出信息。
   """
   print(f"\n--- 🏃 Running Agent with Query: '{query}' ---")
   try:
       # Langgraph agents expect 'messages' as input
       # Langgraph 智能体期望 'messages' 作为输入
       response = await agent_executor.ainvoke({"messages": [("user", query)]})
       print("\n--- ✅ Final Agent Response ---")
       # Get the last message content from the response
       # 从响应中获取最后一条消息的内容
       final_message = response["messages"][-1].content
       print(final_message)
   except Exception as e:
       print(f"\n🛑 An error occurred during agent execution: {e}")

async def main():
   """
   Runs all agent queries concurrently.
   并发运行所有智能体查询任务。
   """
   tasks = [
       run_agent_with_tool("What is the capital of France?"),
       run_agent_with_tool("What's the weather like in London?"),
       run_agent_with_tool("Tell me something about dogs.") # Should trigger the default tool response
   ]
   await asyncio.gather(*tasks)

if llm:
   nest_asyncio.apply()
   asyncio.run(main())
else:
   print("\n❌ Cannot run agent: LLM was not initialized. Please set OPENAI_API_KEY in your .env file.")