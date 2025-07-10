---
title: LangChain Custom Tools
description: Let's build custom tools in LangChain.
author: DS2Man
date: 2025-05-14 11:00:00 +0000
categories: [LLM&RAG, L&R-Tools]
tags:
  - LangChain
  - Agent
  - LangGraph
  - Tools
math: true
pin: true
---

LangChain offers a wide variety of built-in tools for search, math, web APIs, and more—but sometimes, you need something specific to your use case. That’s where **custom tools** come in.

In this post, we’ll explore what custom tools are, how to create them in LangChain, and when you should consider building your own.

## *How to Create a Custom Tool*

There are two ways to create a tool: using `@tool` or `Tool` from the `langchain_core.tools.tool` module. For flexible customization, the `@tool` decorator is more useful, so we'll use that. (The `Tool`-based approach will not be covered in this post.)

**[Note!]** If possible, wrap built-in tools as custom tools as well—for the sake of code consistency!


### *python_repl_tool*

```python
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool

# 도구 생성
python_repl_tool_instance = PythonREPLTool()

@tool
def python_repl_tool(
    code,
):
    """
    Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    """
    result = ""
    try:
        result = python_repl_tool_instance.invoke(code)
    except BaseException as e:
        print(f"Failed to execute. Error: {repr(e)}")
    finally:
        return result
```

```python
print("==========="*4)
print(f"tool name: {python_repl_tool.name}")
print(f"tool description: {python_repl_tool.description}")
print("==========="*4)

code = "print(2 + 3 * 5)" # 17
code = "print(sum([1, 2, 3, 4]))" # 10
code = "import os; print(os.getcwd())" # d:\02.MyCode\GP-MyReference\13.MyLLM
result = python_repl_tool.invoke(code) # repl_tool.run(code), ok!
print("Result1:", result)  # → 50

python_repl_tool.invoke("""
def square(x):
    return x * x
""")
result = python_repl_tool.invoke("print(square(5))") # 25
print("Result2:", result)  # → 50
```

```bash
============================================
tool name: python_repl_tool
tool description: Use this to execute python code. If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.
============================================
Result1: d:\02.MyCode\GP-MyReference\13.MyLLM

Result2: 25
```

### *tavily_search_tool*

```python
from langchain_tavily.tavily_search import TavilySearch
from langchain_core.tools import tool
import os

os.environ["TAVILY_API_KEY"] = "tvly-dev-***********************"

tavily_search_instance = TavilySearch(
    max_results=3,
    include_answer=False,
    include_raw_content=True,
    # include_images=True,
    search_depth="basic", # or "advanced"
    # include_domains=["github.io", "wikidocs.net"],
    # exclude_domains = []
)

@tool
def tavily_search_tool(query: str):
    """
    Use this tool to search the web using Tavily.
    """
    try:
        return tavily_search_instance.invoke(query)
    except BaseException as e:
        print(f"Failed to search Tavily. Error: {repr(e)}")
        return []
```

```python
print("==========="*4)
print(f"tool name: {tavily_search_tool.name}")
print(f"tool description: {tavily_search_tool.description}")
print("==========="*4)

result = tavily_search_tool.invoke("2010년 ~ 2024년 대한민국의 1인당 GDP는?")
result
```

```bash
============================================
tool name: tavily_search_tool
tool description: Use this tool to search the web using Tavily.
============================================

{'query': '2010년 ~ 2024년까지의 대한민국의 1인당 GDP는?',
 'follow_up_questions': None,
 'answer': None,
 'images': [],
 'results': [{'url': 'https://ko.tradingeconomics.com/south-korea/gdp-per-capita', ...
   'title': '대한민국 1인당 GDP | 1960-2023 데이터 | 2024-2025 예상 - 경제 지표',
   'content': '대한민국의 2023년 1인당 국내총생산(GDP)은 34121.02 미국 달러로 기록되었습니다. | 농업 GDP ...
   'score': 0.7997713,
   'raw_content': '대한민국 - 1인당 국내총생산 | 1960-2023 데이터 | 2024-2025 예상\n=============== \n ...
  {'url': 'https://www.ceicdata.com/ko/indicator/korea/gdp-per-capita',
   'title': '대한민국 | 1인당 국내총생산 | 1953년 – 2025년 | 경제 지표 - CEIC',
   'content': 'Accept Decline ENG 영어 중국말 일본어 인도네시아 인 한국어 독일 사람 포르투갈 인 국가 지표 ...
   'score': 0.79293,
   'raw_content': 'Published Time: Fri Jun 01 2018 05:11:07 GMT+0000 (Coordinated Universal Time) ...
  {'url': 'https://namu.wiki/w/%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD/GDP',
   'title': '대한민국/GDP - 나무위키',
   'content': '[1] 기준년 개편 결과, 2차 개편결과 참조.[2] 2005년엔 10위, 2020년엔 9위를 기록했다.[3] IMF, ...
   'score': 0.40938663,
   'raw_content': '대한민국/GDP - 나무위키\n===============                        \n\n[](https://namu.wiki/ "나무위키") ...
 'response_time': 1.33}
```

### *googlenews_search_tool*

```python
from langchain_teddynote.tools import GoogleNews
from langchain.tools import tool
from typing import List, Dict

googlenews_instance = GoogleNews()

@tool
def googlenews_search_tool(query: str) -> List[Dict[str, str]]:
    """Look up news by keyword"""
    return googlenews_instance.search_by_keyword(query, k=3)
```

```python
googlenews_search_tool.invoke("AI 뉴스 알려줘") # ok
# googlenews_search_tool.invoke({'query':"AI 뉴스 알려줘"}) # ok
```

```bash
[{'url': 'https://news.google.com/rss/articles/CBMiW0FVX3lxTFBmdmdiN3B1V2F5Mmc3alBDVElDM3NMcVlFT0x1NU5QcFoydkdwbmtEcF9rR1JNUHh1VWlzYjgtVG5abXgyYzBWRWlwb0c5dFlneE5zVTI2SUVXOEXSAWBBVV95cUxPRmViYzJSRWZ3UFpMLWMxX19sc3Zld2lzY3gtZEJXLWZ5ZjdYN3gyUV9HRG5FVkR5NmhlVmxuLTFaZkVTWU1TMHRBWmRmbzROOVp3TWozZHVmelhWakstUjI?oc=5',
  'content': '"딥시크, 랜섬웨어·화염병 정보 알려줘…범죄 악용 우려" - 연합뉴스'},
 {'url': 'https://news.google.com/rss/articles/CBMiZ0FVX3lxTE1MVDAyWkRRa0c4Rlh4UnBuVEUxamJ6RWJCcWpxSHhKeUFGQ1hZZEZQRHEyMFhTRnJhNzBiTU1xcjROVkJJdVlYV0RUSUgtTjJtLUNGZG8wSkRlZl9zb2VfTi1DSDBpc1k?oc=5',
  'content': '네이버 지도 내비게이션에 인공지능 탑재...AI가 운전 속도, 주행 패턴 등 특징적 운전 습관 기반 개인 맞춤 도착 시간 알려줘 - 인공지능신문'},
 {'url': 'https://news.google.com/rss/articles/CBMiZEFVX3lxTE9pTWxDdGVXUnV0RzVSM1JiUXVna0tMcWZ6NXZjM0pyODZSWlhqMWpBVDJUUFVqSk01MlVyd3Vpcy1zZUNzWXlFS1dmVGZCYXNuODE4X1RVUDEyam1TX19fRzRJREs?oc=5',
  'content': '동물병원에 부는 AI 바람…"논문 근거 답변에 심장 수치도 알려줘" - 뉴스1'}]
```

### *get_current_datetime_tool*

```python
from datetime import datetime
from langchain_core.tools import tool

@tool
def get_current_datetime_tool() -> str:
    """Returns the current date and time."""
    now = datetime.now()
    return now.strftime("Today is %B %d, %Y at %H:%M.")
```

```python
# get_current_datetime_tool.invoke("What is today's date?") # ok
get_current_datetime_tool.invoke({'query':"What is today's date?"}) # ok
```

```bash
'Today is July 10, 2025 at 22:11.'
```