---
title: AI Agent
description: Let's explore AI Agents.
author: DS2Man
date: 2025-05-16 11:00:00 +0000
categories: [LLM&RAG, L&R-Agent]
tags:
  - LangChain
  - Tools
  - Agent
math: true
pin: true
---

I understand an AI Agent, simply put, as binding the Tools we learned in the previous post to an LLM. I don’t think we need to view Agents as something overly grand or complicated. In this post, we’ll take some time to implement an Agent ourselves and evaluate its performance.

<!--
AI Agent는 쉽게 이야기해 이전 포스트에서 배웠던 Tool을 LLM에 Binding 시킨 거라고 나는 이해하고 있다. 거창하게 볼 필요가 없는 것이 Agent 가 아닌가 싶다. 이번 포스트에서는 Agent를 직접 구현해 보고, 평가해 보는 시간을 가지도록 하겠다.
-->

## *Steps to Build a Tool Calling Agent*

In this post, we'll walk through how to build a Tool Calling Agent from scratch using a modular approach. A Tool Calling Agent enables an LLM to intelligently determine **when to call a tool** and **how to call it**, allowing it to complete tasks that go beyond pure text generation.

Here's the step-by-step breakdown:

1. Tool Creation
2. Prompt Creation
3. Agent Creation
4. AgentExecutor Setup

### *1. Tool Creation*

First, define the tools that your agent can use. These tools act as extensions that the agent can call to perform specific functions such as web search, math computation, PDF search, or code execution.
Make sure each tool is annotated with `@tool` and has a clear docstring so the LLM can understand its purpose and how to use it.

~~~python
#############################################################################
# python_repl_tool
#############################################################################
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool

python_repl_tool_instance = PythonREPLTool()

@tool
def python_repl_tool(
    query: str,
) -> str:
    """
    Executes Python code in a REPL. 

    To show the result to the user, use `print(...)`. 
    Only printed output will be returned and visible.
    """
    try:
        return python_repl_tool_instance.invoke(query)
    except Exception as e:
        error_msg = f"[Execution Error] {type(e).__name__}: {str(e)}"
        print(error_msg)
        return error_msg
    
#############################################################################
# tavily_search_tool
#############################################################################
from langchain_tavily.tavily_search import TavilySearch
from langchain_core.tools import tool
import os

os.environ["TAVILY_API_KEY"] = "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"

# Tavily 도구 인스턴스 생성
tavily_search_instance = TavilySearch(
    max_results=6,  # 더 많은 결과 확보
    include_answer=True, # 요약된 답변 활성화
    include_raw_content=False, # 너무 긴 원문은 비활성화
    # include_images=True,
    search_depth="advanced", # or "basic"  # 더 깊은 검색 (추천)
    # include_domains=["github.io", "wikidocs.net"],
    # exclude_domains = []
)

@tool
def tavily_search_tool(query: str):
    """
    Search the web using Tavily and return a list of relevant documents.

    Provide a search query as input in Korean.
    Returns a list of documents containing raw content from search results.
    Use this tool to retrieve up-to-date web information.
    """
    try:
        return tavily_search_instance.invoke(query)
    except Exception as e:
        error_msg = f"[Tavily Search Error] {type(e).__name__}: {str(e)}"
        print(error_msg)
        return [{"error": error_msg}]

#############################################################################
# googlenews_search_tool
#############################################################################
from langchain_teddynote.tools import GoogleNews
from langchain.tools import tool
from typing import List, Dict

googlenews_instance = GoogleNews()

@tool
def googlenews_search_tool(query: str) -> List[Dict[str, str]]:
    """
    Searches Google News for recent articles related to the given keyword.

    Returns a list of dictionaries, each containing article metadata such as title, link, and content.
    """
    try:
        return googlenews_instance.search_by_keyword(query, k=3)
        # return "\n".join(
        #     [f'- {news["content"]}' for news in googlenews_instance.search_by_keyword(query, k=3)]
        # )
    except Exception as e:
        error_msg = f"[GoogleNews Error] {type(e).__name__}: {str(e)}"
        print(error_msg)
        return [{"error": error_msg}]

#############################################################################
# get_current_datetime_tool
#############################################################################
from datetime import datetime
from langchain_core.tools import tool

@tool
def get_current_datetime_tool() -> str:
    """
    Returns the current date and time in a human-readable format.

    Format: 'Today is Month Day, Year at HH:MM.'
    Example: 'Today is July 11, 2025 at 16:45.'
    """
    try:
        now = datetime.now()
        return now.strftime("Today is %B %d, %Y at %H:%M.")
    except Exception as e:
        error_msg = f"[Datetime Error] {type(e).__name__}: {str(e)}"
        print(error_msg)
        return [{"error": error_msg}]

#############################################################################
# pdf_retriever_tool
#############################################################################
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

loader = PyMuPDFLoader("zData/SPRI_AI_Brief.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = loader.load_and_split(text_splitter)
embeddings = HuggingFaceEmbeddings(model_name="./Pretrained_byGit/bge-m3")
vector = FAISS.from_documents(split_docs, embeddings)

retriever = vector.as_retriever()

@tool
def pdf_retriever_tool(query: str) -> str:
    """
    Search information from the PDF document using semantic similarity.

    Provide a question or keyword, and this tool will return relevant chunks from the document.
    """
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])
~~~

### *2. Prompt Creation*

Design a prompt that guides the agent's behavior and instructs it how to interact with the tools. This includes defining the system message and the context in which the tools should be used.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

my_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("placeholder", "{chat_history}"), 
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
```

### *3. Agent Creation*

Now combine the LLM, the list of tools, and the prompt to create the Tool Calling Agent.
This agent will now be able to choose which tool to call and with what inputs, based on the user query.

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent

openai_api_key = os.environ["OPENROUTER_API_KEY"]
openai_api_base = os.environ["OPENROUTER_ENDPOINT"]

my_llm = ChatOpenAI(
    model="meta-llama/llama-4-maverick",
    openai_api_base=openai_api_base,
    openai_api_key=openai_api_key
)

my_agent = create_tool_calling_agent(llm=my_llm, tools=my_tools, prompt=my_prompt)
```

### *4. AgentExecutor Setup*

Wrap your agent inside an `AgentExecutor` to handle the interaction loop - executing tools, processing outputs, and keeping track of intermediate steps.
The agent will automatically pick the most relevant tool and return a structured, intelligent response.

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=my_agent,
    tools=my_tools,
    verbose=True,
    max_iterations=10,
    max_execution_time=10,
    handle_parsing_errors=True,
)
```

#### Key Attributes

- **`agent`**: The agent instance that plans and decides the next action at each step of the execution loop.    
- **`tools`**: A list of valid tools that the agent can use during execution.    
- **`verbose`**: If set to `True`, prints detailed logs of the agent’s reasoning process, tool calls, and intermediate steps.  Useful for debugging and understanding how the agent makes decisions during execution.
- **`max_iterations`**: Maximum number of steps allowed before forcefully terminating the loop.
- **`max_execution_time`**: Maximum amount of time (in seconds) allowed for the loop to run.    
- **`handle_parsing_errors`**: How to handle errors raised by the agent’s output parser. Can be `True`, `False`, or a custom error handler function.    

#### Core Methods

- **`invoke`** – Executes the agent and returns the final output.  
    Use this method for standard execution where only the final answer is needed.    
- **`stream`** – Streams the agent’s thought process step-by-step until the final answer.  
    This method yields each step of the agent's reasoning process — including thought, action, observation, and intermediate decisions — in sequence.   
     Note: `stream()` does **not** provide real-time, token-by-token output like an LLM with `streaming=True`.

### *5-1. Invoke*

Executes the agent and returns the final output. 

```python
result = agent_executor.invoke({"input": "2010년부터 2024년까지의 연도별 대한민국의 1인당 GDP는?"})  

print("Agent Result:")
print(result["output"])
```

- The output when `verbose=True` is as follows.

```bash
[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m
Invoking: `tavily_search_tool` with `{'query': 'South Korea GDP per capita 2010-2024'}`
responded: The function definitions you've provided do not encompass this task. Please revise your function definitions. 
To find the GDP per capita of South Korea from 2010 to 2024, we can use the following steps:

1. Search for the GDP per capita data for South Korea.
2. Extract the data for the specified years (2010-2024).

Let's start by searching for the GDP per capita data for South Korea.

[0m[33;1m[1;3m{'query': 'South Korea GDP per capita 2010-2024', 'follow_up_questions': None, 'answer': "South Korea's GDP per capita was $33,121 in 2023 and $32,395 in 2022. It increased by 2.24% in 2023. The data covers 2010-2024.", 'images': [], 'results': [{'url': 'https://www.macrotrends.net/global-metrics/countries/kor/south-korea/gdp-gross-domestic-product', 'title': 'South Korea GDP | Historical Chart & Data - Macrotrends', 'content': 'South Korea GDP 1960-2025 | MacroTrends South Korea GDP 1960-2025 ##### South Korea GDP for 2023 was 1.713 trillion US dollars, a 2.32% increase from 2022. South Korea GDP for 2022 was 1.674 trillion US dollars, a 7.95% decline from 2021. South Korea GDP for 2021 was 1.818 trillion US dollars, a 10.59% increase from 2020. South Korea GDP for 2020 was 1.644 trillion US dollars, a 0.43% decline from 2019. Image 2: Click on chart icon to view chart Image 3: Click on table icon to view table Image 4: Click on chart icon to view chart Image 5: Click on table icon to view table Image 7: Click on table icon to view table', 'score': 0.98557, 'raw_content': None}, {'url': 'https://www.macrotrends.net/global-metrics/countries/kor/south-korea/gdp-per-capita', 'title': 'South Korea GDP Per Capita | Historical Chart & Data - Macrotrends', 'content': '## South Korea GDP Per Capita GDP GDP Per Capita ##### South Korea GDP per capita for 2023 was **$33,121**, a **2.24% increase** from 2022. * South Korea GDP per capita for 2022 was **$32,395**, a **7.77% decline** from 2021. * South Korea GDP per capita for 2021 was **$35,126**, a **10.73% increase** from 2020. * South Korea GDP per capita for 2020 was **$31,721**, a **0.57% decline** from 2019. ## GDP Per Capita (US $) | South Korea GDP Per Capita  GDP Per Capita (US $) | | | Name | GDP Per Capita (US $) | | North America | $79,640.43 | | Name | GDP Per Capita (US $) | | Guinea | $1,541.041 |', 'score': 0.98527, 'raw_content': None}, {'url': 'https://en.wikipedia.org/wiki/Economy_of_South_Korea', 'title': 'Economy of South Korea - Wikipedia', 'content': 'By nominal GDP, the economy was worth ₩2.61 quadrillion (US$1.87 trillion). It has the 4th largest economy in Asia and the 13th largest in the world as of 2025.', 'score': 0.98454, 'raw_content': None}, {'url': 'https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG?locations=KR', 'title': 'GDP growth (annual %) - World Bank Open Data', 'content': 'No information is available for this page. · Learn why', 'score': 0.98122, 'raw_content': None}, {'url': 'https://www.statista.com/statistics/263758/gross-domestic-product-gdp-growth-in-south-korea/', 'title': 'South Korea: GDP growth 1954-2030 - Statista', 'content': "Success stories Live webinars & recordings Prices & Access Business Solutions Academia and Government My Account Prices & Access Business Solutions Academia and Government Statistics Popular Statistics Topics Markets Reports Market Insights Consumer Insights eCommerce Insights Research AI Daily Data Services About Statista Statista+ Statista Q ask Statista nxt Statista Content & Design Statista R Solutions Why Statista Industry Agencies Consulting Goods Finance Tech Function Marketing Research & Development Strategy Use case Tell the story Find the fact Win the pitch Understand the market Develop your strategy DE ES FR Economy & Politics› International Gross domestic product (GDP) growth in South Korea 1954-2029 Published by Aaron O'Neill, Nov 29, 2024 The statistic shows the growth of the real gross domestic product (GDP) in South Korea from 1954 to 2023, with projections up until 2029. GDP is the total value of all goods and services produced in a country in a year. It is considered to be a very important indicator of the economic strength of a country and a change in it is a sign of economic growth. In 2023, the real GDP in South Korea grew by about 1.4 percent compared to the previous year.", 'score': 0.97745, 'raw_content': None}, {'url': 'https://www.ceicdata.com/en/indicator/korea/gdp-per-capita', 'title': 'South Korea GDP per Capita | Economic Indicators - CEIC', 'content': 'Accept Decline ENG English Chinese Japanese Indonesian Korean German Portuguese Countries Indicators Products Our insights About ENG English 中文 日本語 한국어 Indonesian Deutsche Português Request a demo Home > Countries/Regions > South Korea > South Korea GDP per Capita South Korea GDP per Capita 1953 - 2024 | Yearly | USD | The Bank of Korea Key information about South Korea GDP Per Capita South Korea Gross Domestic Product (GDP) per Capita reached 36,113.000 USD in Dec 2024, compared with 35,569.900 USD in Dec 2023. South Korea GDP Per Capita data is updated yearly, available from Dec 1953 to Dec 2024, with an average number of 5,442.300 USD. The data reached an all-time high of 37,503.100 USD in Dec 2021 and a record low of 64.230 in Dec 1955. South Korea Gross Domestic Product (GDP) per Capita reached 36,113.000 USD in Dec 2024, compared with 35,569.900 USD in Dec 2023.', 'score': 0.97663, 'raw_content': None}], 'response_time': 6.42}[0m[32;1m[1;3mThe GDP per capita of South Korea from 2010 to 2024 is as follows:

- 2023: $33,121
- 2022: $32,395
- 2021: $35,126
- 2020: $31,721
- Other years' data is available from sources like Macrotrends and CEIC.

To get the exact figures for other years, I recommend checking the Macrotrends or CEIC websites.[0m

[1m> Finished chain.[0m

Agent 실행 결과:
The GDP per capita of South Korea from 2010 to 2024 is as follows:

- 2023: $33,121
- 2022: $32,395
- 2021: $35,126
- 2020: $31,721
- Other years' data is available from sources like Macrotrends and CEIC.

To get the exact figures for other years, I recommend checking the Macrotrends or CEIC websites.
```

- The output when `verbose=False` is as follows.

```bash
Agent 실행 결과:
1인당 GDP는 국내 총생산을 인구 수로 나눈 값입니다. 
2010년부터 2024년까지 대한민국의 1인당 GDP를 알아보겠습니다.

먼저, 2010년부터 2023년까지의 데이터는 국가통계포털에서 확인할 수 있습니다. 
2010년: 22,134 달러
2011년: 24,302 달러
2012년: 24,801 달러
2013년: 25,977 달러
2014년: 28,166 달러
2015년: 28,738 달러
2016년: 29,114 달러
2017년: 31,362 달러
2018년: 33,434 달러
2019년: 31,942 달러
2020년: 31,994 달러
2021년: 35,168 달러
2022년: 32,661 달러
2023년: 33,147 달러

2024년의 데이터는 아직 확정되지 않았으며, 정확한 값을 알기 위해서는 공식 통계 발표를 기다려야 합니다. 다만, 국제통화기금(IMF)이나 세계은행(Word Bank) 등에서 예측치를 발표하고 있습니다. 
2024년 대한민국의 1인당 GDP 예측치는 다음과 같습니다.
- 국제통화기금(IMF): 약 36,144 달러
- 세계은행(Word Bank): 약 35,813 달러

이러한 예측치는 경제 상황에 따라 변동될 수 있으므로 참고용으로만 사용하시기 바랍니다. 정확한 2024년 대한민국의 1인당 GDP는 공식 통계 발표 이후 확인 가능합니다.
```


### *5-2. Stream*

Streams the agent’s thought process step-by-step until the final answer.  
This method yields each step of the agent's reasoning process — including thought, action, observation, and intermediate decisions — in sequence.   

|Output|Description|
|---|---|
|**Action**|`actions`: An `AgentAction` object or its subclass<br>`messages`: Chat messages corresponding to the action call|
|**Observation**|`steps`: A record of the agent's actions and observations up to this point<br>`messages`: Chat messages including the result of the function/tool call (i.e., the observation)|
|**Final Answer**|`output`: An `AgentFinish` object<br>`messages`: Chat messages including the final output|

```python
result = agent_executor.stream({"input": "AI 투자와 관련된 뉴스를 검색해 주세요."})  

for step in result:
    print(step)
    print("===" * 20)
```

- The output when `verbose=True` is as follows.

```bash
[1m> Entering new None chain...[0m
{'actions': [ToolAgentAction(tool='googlenews_search_tool', tool_input={'query': 'AI 투자 상태'}, log="\nInvoking: `googlenews_search_tool` with `{'query': 'AI 투자 상태'}`\n\n\n", message_log=[AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'xz5p5zam9', 'function': {'arguments': '{"query":"AI 투자 상태"}', 'name': 'googlenews_search_tool'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'meta-llama/llama-4-maverick', 'system_fingerprint': 'fp_c527aa4474'}, id='run--c42bab3a-7d8b-4176-bd2c-339cae236f03', tool_calls=[{'name': 'googlenews_search_tool', 'args': {'query': 'AI 투자 상태'}, 'id': 'xz5p5zam9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 983, 'output_tokens': 42, 'total_tokens': 1025, 'input_token_details': {}, 'output_token_details': {}}, tool_call_chunks=[{'name': 'googlenews_search_tool', 'args': '{"query":"AI 투자 상태"}', 'id': 'xz5p5zam9', 'index': 0, 'type': 'tool_call_chunk'}])], tool_call_id='xz5p5zam9')], 'messages': [AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'xz5p5zam9', 'function': {'arguments': '{"query":"AI 투자 상태"}', 'name': 'googlenews_search_tool'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'meta-llama/llama-4-maverick', 'system_fingerprint': 'fp_c527aa4474'}, id='run--c42bab3a-7d8b-4176-bd2c-339cae236f03', tool_calls=[{'name': 'googlenews_search_tool', 'args': {'query': 'AI 투자 상태'}, 'id': 'xz5p5zam9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 983, 'output_tokens': 42, 'total_tokens': 1025, 'input_token_details': {}, 'output_token_details': {}}, tool_call_chunks=[{'name': 'googlenews_search_tool', 'args': '{"query":"AI 투자 상태"}', 'id': 'xz5p5zam9', 'index': 0, 'type': 'tool_call_chunk'}])]}
============================================================
[32;1m[1;3m
Invoking: `googlenews_search_tool` with `{'query': 'AI 투자 상태'}`


[0m[38;5;200m[1;3m[{'url': 'https://news.google.com/rss/articles/CBMiWkFVX3lxTE5KTHJybjN0RGdUSHBYT0g1SURuSnF5cDBKVG5TZWtzbnNtSFdPanRwVG5TbTFsSTRaYnBPc0ZYM0ZYVm9RdjRIS2xOcFotbTg3RkN3ZUxyQjYxd9IBVEFVX3lxTE5NNzlVZEpiRDFlaG10VzhnZDEwSzA1eDE3aUhYdmEzcmJ6dGUta0JWZHZ4WDVxdEE1ZmdHQzlEMTlEWS1yd3pZTkRWZmh3QzdfSHo1dg?oc=5', 'content': '실리콘밸리에 VC 만드는 네이버…"AI 총력전" - 한국경제'}, {'url': 'https://news.google.com/rss/articles/CBMiakFVX3lxTE9Rd2F5dGtpeHBWYXdRdkF2NUtFU2NiOVkyMXpwbU5UNDc0Q01YTFJUY3VTU05WeUdpUEVpS1ZuZ29yaW85ZThFYXJNTTFjZDI5OUpXaDlkaHYtdUQycTd3TWF4SDhUcE5oNUE?oc=5', 'content': '"중국, AI 데이터센터 러시 붕괴...상당수는 유휴 상태" - AI타임스'}, {'url': 'https://news.google.com/rss/articles/CBMiVkFVX3lxTE5ka3hfOXlaUGJlNlBzaEJhVVRYYVN2YTVVMG9UOTk1WW1xZHM4Ykk2N3RZbTJ1VTFiaGhUUUVrcGZGU195R2puSTQwNlN6dGRXTUxNV2Nn?oc=5', 'content': "무라티 오픈AI 전 CTO, '3조' 투자 유치…스타트업 가치 '14조' 평가 - 지디넷코리아"}][0m{'steps': [AgentStep(action=ToolAgentAction(tool='googlenews_search_tool', tool_input={'query': 'AI 투자 상태'}, log="\nInvoking: `googlenews_search_tool` with `{'query': 'AI 투자 상태'}`\n\n\n", message_log=[AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'xz5p5zam9', 'function': {'arguments': '{"query":"AI 투자 상태"}', 'name': 'googlenews_search_tool'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'meta-llama/llama-4-maverick', 'system_fingerprint': 'fp_c527aa4474'}, id='run--c42bab3a-7d8b-4176-bd2c-339cae236f03', tool_calls=[{'name': 'googlenews_search_tool', 'args': {'query': 'AI 투자 상태'}, 'id': 'xz5p5zam9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 983, 'output_tokens': 42, 'total_tokens': 1025, 'input_token_details': {}, 'output_token_details': {}}, tool_call_chunks=[{'name': 'googlenews_search_tool', 'args': '{"query":"AI 투자 상태"}', 'id': 'xz5p5zam9', 'index': 0, 'type': 'tool_call_chunk'}])], tool_call_id='xz5p5zam9'), observation=[{'url': 'https://news.google.com/rss/articles/CBMiWkFVX3lxTE5KTHJybjN0RGdUSHBYT0g1SURuSnF5cDBKVG5TZWtzbnNtSFdPanRwVG5TbTFsSTRaYnBPc0ZYM0ZYVm9RdjRIS2xOcFotbTg3RkN3ZUxyQjYxd9IBVEFVX3lxTE5NNzlVZEpiRDFlaG10VzhnZDEwSzA1eDE3aUhYdmEzcmJ6dGUta0JWZHZ4WDVxdEE1ZmdHQzlEMTlEWS1yd3pZTkRWZmh3QzdfSHo1dg?oc=5', 'content': '실리콘밸리에 VC 만드는 네이버…"AI 총력전" - 한국경제'}, {'url': 'https://news.google.com/rss/articles/CBMiakFVX3lxTE9Rd2F5dGtpeHBWYXdRdkF2NUtFU2NiOVkyMXpwbU5UNDc0Q01YTFJUY3VTU05WeUdpUEVpS1ZuZ29yaW85ZThFYXJNTTFjZDI5OUpXaDlkaHYtdUQycTd3TWF4SDhUcE5oNUE?oc=5', 'content': '"중국, AI 데이터센터 러시 붕괴...상당수는 유휴 상태" - AI타임스'}, {'url': 'https://news.google.com/rss/articles/CBMiVkFVX3lxTE5ka3hfOXlaUGJlNlBzaEJhVVRYYVN2YTVVMG9UOTk1WW1xZHM4Ykk2N3RZbTJ1VTFiaGhUUUVrcGZGU195R2puSTQwNlN6dGRXTUxNV2Nn?oc=5', 'content': "무라티 오픈AI 전 CTO, '3조' 투자 유치…스타트업 가치 '14조' 평가 - 지디넷코리아"}])], 'messages': [FunctionMessage(content='[{"url": "https://news.google.com/rss/articles/CBMiWkFVX3lxTE5KTHJybjN0RGdUSHBYT0g1SURuSnF5cDBKVG5TZWtzbnNtSFdPanRwVG5TbTFsSTRaYnBPc0ZYM0ZYVm9RdjRIS2xOcFotbTg3RkN3ZUxyQjYxd9IBVEFVX3lxTE5NNzlVZEpiRDFlaG10VzhnZDEwSzA1eDE3aUhYdmEzcmJ6dGUta0JWZHZ4WDVxdEE1ZmdHQzlEMTlEWS1yd3pZTkRWZmh3QzdfSHo1dg?oc=5", "content": "실리콘밸리에 VC 만드는 네이버…\\"AI 총력전\\" - 한국경제"}, {"url": "https://news.google.com/rss/articles/CBMiakFVX3lxTE9Rd2F5dGtpeHBWYXdRdkF2NUtFU2NiOVkyMXpwbU5UNDc0Q01YTFJUY3VTU05WeUdpUEVpS1ZuZ29yaW85ZThFYXJNTTFjZDI5OUpXaDlkaHYtdUQycTd3TWF4SDhUcE5oNUE?oc=5", "content": "\\"중국, AI 데이터센터 러시 붕괴...상당수는 유휴 상태\\" - AI타임스"}, {"url": "https://news.google.com/rss/articles/CBMiVkFVX3lxTE5ka3hfOXlaUGJlNlBzaEJhVVRYYVN2YTVVMG9UOTk1WW1xZHM4Ykk2N3RZbTJ1VTFiaGhUUUVrcGZGU195R2puSTQwNlN6dGRXTUxNV2Nn?oc=5", "content": "무라티 오픈AI 전 CTO, \'3조\' 투자 유치…스타트업 가치 \'14조\' 평가 - 지디넷코리아"}]', additional_kwargs={}, response_metadata={}, name='googlenews_search_tool')]}
============================================================
[32;1m[1;3m네이버가 실리콘밸리에 벤처 캐피탈을 설립하여 AI에 대한 총력전을 펼치고 있습니다. 또한 중국은 AI 데이터센터에 대한 투자가 붕괴되어 상당수의 데이터센터가 유휴 상태에 있습니다. 무라티 오픈AI 전 CTO는 3조원의 투자를 유치하여 스타트업 가치를 14조로 평가 받았습니다.[0m

[1m> Finished chain.[0m
{'output': '네이버가 실리콘밸리에 벤처 캐피탈을 설립하여 AI에 대한 총력전을 펼치고 있습니다. 또한 중국은 AI 데이터센터에 대한 투자가 붕괴되어 상당수의 데이터센터가 유휴 상태에 있습니다. 무라티 오픈AI 전 CTO는 3조원의 투자를 유치하여 스타트업 가치를 14조로 평가 받았습니다.', 'messages': [AIMessage(content='네이버가 실리콘밸리에 벤처 캐피탈을 설립하여 AI에 대한 총력전을 펼치고 있습니다. 또한 중국은 AI 데이터센터에 대한 투자가 붕괴되어 상당수의 데이터센터가 유휴 상태에 있습니다. 무라티 오픈AI 전 CTO는 3조원의 투자를 유치하여 스타트업 가치를 14조로 평가 받았습니다.', additional_kwargs={}, response_metadata={})]}
============================================================
```

- The output when `verbose=False` is as follows.

```bash
{'actions': [ToolAgentAction(tool='googlenews_search_tool', tool_input={'query': 'AI 투자 뉴스'}, log="\nInvoking: `googlenews_search_tool` with `{'query': 'AI 투자 뉴스'}`\n\n\n", message_log=[AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'zk96rs0ka', 'function': {'arguments': '{"query":"AI 투자 뉴스"}', 'name': 'googlenews_search_tool'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'meta-llama/llama-4-maverick', 'system_fingerprint': 'fp_c527aa4474'}, id='run--a5816d12-3c7a-4b11-bfc5-5bf069c40e4a', tool_calls=[{'name': 'googlenews_search_tool', 'args': {'query': 'AI 투자 뉴스'}, 'id': 'zk96rs0ka', 'type': 'tool_call'}], usage_metadata={'input_tokens': 983, 'output_tokens': 28, 'total_tokens': 1011, 'input_token_details': {}, 'output_token_details': {}}, tool_call_chunks=[{'name': 'googlenews_search_tool', 'args': '{"query":"AI 투자 뉴스"}', 'id': 'zk96rs0ka', 'index': 0, 'type': 'tool_call_chunk'}])], tool_call_id='zk96rs0ka')], 'messages': [AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'zk96rs0ka', 'function': {'arguments': '{"query":"AI 투자 뉴스"}', 'name': 'googlenews_search_tool'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'meta-llama/llama-4-maverick', 'system_fingerprint': 'fp_c527aa4474'}, id='run--a5816d12-3c7a-4b11-bfc5-5bf069c40e4a', tool_calls=[{'name': 'googlenews_search_tool', 'args': {'query': 'AI 투자 뉴스'}, 'id': 'zk96rs0ka', 'type': 'tool_call'}], usage_metadata={'input_tokens': 983, 'output_tokens': 28, 'total_tokens': 1011, 'input_token_details': {}, 'output_token_details': {}}, tool_call_chunks=[{'name': 'googlenews_search_tool', 'args': '{"query":"AI 투자 뉴스"}', 'id': 'zk96rs0ka', 'index': 0, 'type': 'tool_call_chunk'}])]}
============================================================
{'steps': [AgentStep(action=ToolAgentAction(tool='googlenews_search_tool', tool_input={'query': 'AI 투자 뉴스'}, log="\nInvoking: `googlenews_search_tool` with `{'query': 'AI 투자 뉴스'}`\n\n\n", message_log=[AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'zk96rs0ka', 'function': {'arguments': '{"query":"AI 투자 뉴스"}', 'name': 'googlenews_search_tool'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'meta-llama/llama-4-maverick', 'system_fingerprint': 'fp_c527aa4474'}, id='run--a5816d12-3c7a-4b11-bfc5-5bf069c40e4a', tool_calls=[{'name': 'googlenews_search_tool', 'args': {'query': 'AI 투자 뉴스'}, 'id': 'zk96rs0ka', 'type': 'tool_call'}], usage_metadata={'input_tokens': 983, 'output_tokens': 28, 'total_tokens': 1011, 'input_token_details': {}, 'output_token_details': {}}, tool_call_chunks=[{'name': 'googlenews_search_tool', 'args': '{"query":"AI 투자 뉴스"}', 'id': 'zk96rs0ka', 'index': 0, 'type': 'tool_call_chunk'}])], tool_call_id='zk96rs0ka'), observation=[{'url': 'https://news.google.com/rss/articles/CBMiTkFVX3lxTE1hamdYZktCX1hseDJ2eENvdWhoQ2ZtSi1ETkJ5LWlSS1llQnRoWjR4NHpORE5tU3B3VG9fTElvNWM5VXlVQ0xYeFMyZkQwdw?oc=5', 'content': '카카오, SK스퀘어 지분 전량 매각…AI 투자 재원 4296억 확보 - 전자신문'}, {'url': 'https://news.google.com/rss/articles/CBMiZEFVX3lxTE9BdTBNcUthUm03Rjc1bUZTSkQxbThlUC1tNEpyakxkaE84ZUpiUF9ocE9ucXdGbERORW9iMGt0bmh3OGpuUzFHYVNFWjRYYWdVclQ2ODAyXzFmd3FaTW5jZ3NrOVc?oc=5', 'content': 'SK스퀘어 지분 4300억원 매각하는 카카오, AI 투자 늘릴까 - 디지털데일리'}, {'url': 'https://news.google.com/rss/articles/CBMiYEFVX3lxTE1nb3paX3BySEI1WFhDdERSQkpCU3FlRV9uaGNFSGJYT2xkZTlaNGNxS2hKakd2dHJaX0NBbU44WE5LYmVsRnJVaEJsVDFuX0lyMlZFdkFLX0YzbFNkOXdXeg?oc=5', 'content': '카카오, 4300억 규모 SK스퀘어 지분 매각…“AI 투자재원 확보” - 한겨레'}])], 'messages': [FunctionMessage(content='[{"url": "https://news.google.com/rss/articles/CBMiTkFVX3lxTE1hamdYZktCX1hseDJ2eENvdWhoQ2ZtSi1ETkJ5LWlSS1llQnRoWjR4NHpORE5tU3B3VG9fTElvNWM5VXlVQ0xYeFMyZkQwdw?oc=5", "content": "카카오, SK스퀘어 지분 전량 매각…AI 투자 재원 4296억 확보 - 전자신문"}, {"url": "https://news.google.com/rss/articles/CBMiZEFVX3lxTE9BdTBNcUthUm03Rjc1bUZTSkQxbThlUC1tNEpyakxkaE84ZUpiUF9ocE9ucXdGbERORW9iMGt0bmh3OGpuUzFHYVNFWjRYYWdVclQ2ODAyXzFmd3FaTW5jZ3NrOVc?oc=5", "content": "SK스퀘어 지분 4300억원 매각하는 카카오, AI 투자 늘릴까 - 디지털데일리"}, {"url": "https://news.google.com/rss/articles/CBMiYEFVX3lxTE1nb3paX3BySEI1WFhDdERSQkpCU3FlRV9uaGNFSGJYT2xkZTlaNGNxS2hKakd2dHJaX0NBbU44WE5LYmVsRnJVaEJsVDFuX0lyMlZFdkFLX0YzbFNkOXdXeg?oc=5", "content": "카카오, 4300억 규모 SK스퀘어 지분 매각…“AI 투자재원 확보” - 한겨레"}]', additional_kwargs={}, response_metadata={}, name='googlenews_search_tool')]}
============================================================
{'output': '카카오가 SK스퀘어 지분을 전량 매각하여 약 4300억 원의 자금을 확보했으며, 이 자금은 AI 투자 재원으로 사용될 예정입니다. 관련 기사에 따르면 카카오는 이번 매각을 통해 AI 투자에 필요한 재원을 마련할 계획입니다.', 'messages': [AIMessage(content='카카오가 SK스퀘어 지분을 전량 매각하여 약 4300억 원의 자금을 확보했으며, 이 자금은 AI 투자 재원으로 사용될 예정입니다. 관련 기사에 따르면 카카오는 이번 매각을 통해 AI 투자에 필요한 재원을 마련할 계획입니다.', additional_kwargs={}, response_metadata={})]}
============================================================
```

## *Agent That Remembers Previous Conversations*

To enable memory of previous conversations, wrap the `AgentExecutor` with `RunnableWithMessageHistory`, the same component used in LangChain for managing conversational memory.

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_ids):
    if session_ids not in store:
        store[session_ids] = ChatMessageHistory()
    return store[session_ids] 

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
```

```python
response = agent_with_chat_history.invoke(
    {"input": "pdf 문서에서 삼성전자가 개발한 생성형 AI는 뭐니? 출처도 표기해줘."},
    config={"configurable": {"session_id": "abc123"}},
)
response
```

```bash
{'input': 'pdf 문서에서 삼성전자가 개발한 생성형 AI는 뭐니? 출처도 표기해줘.',
 'chat_history': [],
 'output': "삼성전자가 개발한 생성형 AI는 '삼성 가우스'입니다. 삼성 가우스는 언어, 코드, 이미지의 3개 모델로 구성되어 있으며, 온디바이스에서 작동 가능하여 외부로 사용자 정보가 유출될 위험이 없다는 장점을 가지고 있습니다. 삼성전자는 삼성 가우스를 다양한 제품에 단계적으로 탑재할 계획입니다.\n\n출처: 삼성 AI 포럼 2023 (https://www.samsung.com/sec/news/press-releases/2023/11/08/)"}
```

```python
response = agent_with_chat_history.invoke(
    {"input": "이전의 답변을 영어로 번역해 주세요."},
    config={"configurable": {"session_id": "abc123"}},
)
response
```

```bash
{'input': '이전의 답변을 영어로 번역해 주세요.',
 'chat_history': [HumanMessage(content='pdf 문서에서 삼성전자가 개발한 생성형 AI는 뭐니? 출처도 표기해줘.', additional_kwargs={}, response_metadata={}),
  AIMessage(content="삼성전자가 개발한 생성형 AI는 '삼성 가우스'입니다. 삼성 가우스는 언어, 코드, 이미지의 3개 모델로 구성되어 있으며, 온디바이스에서 작동 가능하여 외부로 사용자 정보가 유출될 위험이 없다는 장점을 가지고 있습니다. 삼성전자는 삼성 가우스를 다양한 제품에 단계적으로 탑재할 계획입니다.\n\n출처: 삼성 AI 포럼 2023 (https://www.samsung.com/sec/news/press-releases/2023/11/08/)", additional_kwargs={}, response_metadata={})],
 'output': "The generative AI developed by Samsung Electronics is called 'Samsung Gauss'. Samsung Gauss consists of three models: language, code, and image, and has the advantage of being able to operate on-device, reducing the risk of user information being leaked externally. Samsung plans to gradually integrate Samsung Gauss into various products.\n\nSource: Samsung AI Forum 2023 (https://www.samsung.com/sec/news/press-releases/2023/11/08/)"}
```
