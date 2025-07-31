from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.chains.llm import LLMChain
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool

from langchain_community.tools import WikipediaQueryRun, YouTubeSearchTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch

import gradio as gr

import os
os.environ['TAVILY_API_KEY']="Your_Tavily_API_Key"

# ---------------------------- LLM Setup ----------------------------
LLM_MODEL_NAME = "llama3.1:8b"
llm = ChatOllama(model=LLM_MODEL_NAME)

# ---------------------------- Tools ----------------------------
calculator_tool = PythonREPLTool()

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wikipedia = WikipediaQueryRun(
    description="A tool to explain things in text format.",
    api_wrapper=wiki_api_wrapper
)

tavily_raw = TavilySearch(max_results=2)
def simple_tavily_search(query: str) -> str:
    return tavily_raw.run(query)

tavily_tool = Tool(
    name="tavily_search",
    func=simple_tavily_search,
    description="Use this to search the internet for current information."
)

youtube = YouTubeSearchTool(
   description="A tool to search YouTube videos and return video links."
)

tools = [calculator_tool, wikipedia, tavily_tool, youtube]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. "
            "You must use the provided tools to answer user questions when appropriate. "
            "If a tool is not suitable, respond directly. "
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), 
    ]
)

def _format_chat_history(chat_history):
    formatted_messages = []
    for chat_message in chat_history:
        if chat_message["role"] == "user":
            formatted_messages.append(HumanMessage(content=chat_message["content"]))
        elif chat_message["role"] == "assistant":
            formatted_messages.append(AIMessage(content=chat_message["content"]))
    return formatted_messages 

# Create the LangChain Agent
# `create_tool_calling_agent` is the recommended way for tool-calling models
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the AgentExecutor
# This handles the full agentic loop: planning, tool calling, responding.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def stream_agentic_response(message, history):
    """
    Processes user input using the LangChain AgentExecutor and streams the response
    back to the Gradio ChatInterface.
    """
    formatted_history = _format_chat_history(history)
    print(formatted_history, "formatted_history")
    full_response_content = ""

    try:
        for s in agent_executor.stream(
            {"input": message, "chat_history": formatted_history}
        ):
            print(s)
            if "output" in s:
                for char in s["output"]:
                    full_response_content += char
                    yield full_response_content

            
    except Exception as e:
        yield f"An error occurred: {e}"
        print(f"Error during agent streaming: {e}")

    # --- 5. Create Gradio ChatInterface ----------
demo = gr.ChatInterface(
    fn=stream_agentic_response,
    chatbot=gr.Chatbot(height=400, type='messages'), 
    textbox=gr.Textbox(placeholder="Ask me anything!", container=False, scale=7),
    title="Agentic Chatbot using Llama3.1 with Tools",
    description=(
        "This chatbot can leverage tools for specific tasks like performing calculations, searching the internet for real time information, finding helpful youtube videos etc"
    ),
    examples=[
        "What is weather today in Visakhapatnam?",
        "Calculate 123 + (45 * 8) / 2.",
        "What is the capital of Japan?",
        "Tell me interesting facts about India",
        "What is Artificial Intelligence?",
        "Find a youtube video about Agentic AI"
    ],
    cache_examples=False, 
    theme="soft",
)

# --- 6. Launch Gradio App ----------
demo.launch()