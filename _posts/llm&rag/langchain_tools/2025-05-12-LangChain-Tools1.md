---
title: LangChain Tools/Toolkits
description: Let's explore various built-in tools.
author: DS2Man
date: 2025-05-12 11:00:00 +0000
categories: [LLM&RAG, L&R-Tools]
tags:
  - LangChain
  - Agent
  - LangGraph
  - Tools
math: true
pin: true
---

LangChain is a powerful framework that simplifies the process of building applications powered by large language models (LLMs). One of its key strengths lies in its **extensive collection of built-in tools** that help developers seamlessly integrate external capabilities into their LLM workflows.

In this post, we'll explore some of the most commonly used [built-in tools](https://python.langchain.com/docs/integrations/tools/) in LangChain and how they can accelerate the development of intelligent, agent-driven applications.

There are so many built-in tools that it's impossible to learn them all. Instead, I will continuously add the tools I come across while working to this post.

## *PythonREPLTool vs PythonAstREPLTool vs PythonREPL*

When using LangChain, there are times when it's essential to have the LLM execute Python code. To support this, LangChain offers various Python execution tools. In this post, we'll compare three main tools provided by the `langchain.experimental` module.

~~~python
from langchain_experimental.tools import PythonREPLTool, PythonAstREPLTool
from langchain_experimental.utilities.python import PythonREPL

repl_tool = PythonREPLTool()
code = "print(2 + 3 * 5)" # 17
code = "print(sum([1, 2, 3, 4]))" # 10
code = "import os; print(os.getcwd())" # d:\02.MyCode\GP-MyReference\13.MyLLM
result = repl_tool.invoke(code) # repl_tool.run(code), ok!
repl_tool.invoke("""
def square(x):
    return x * x
""")
result = repl_tool.invoke("print(square(5))") # 25
print("REPL Result:", result)  # → 50

print("==========="*4)

ast_tool = PythonAstREPLTool()
code = "print(2 + 3 * 5)" # 17
code = "print(sum([1, 2, 3, 4]))" # 10
code = "import os; print(os.getcwd())" # d:\02.MyCode\GP-MyReference\13.MyLLM
result = ast_tool.invoke(code) # ast_tool.run(code), ok!
ast_tool.invoke("""
def square(x):
    return x * x
""")
result = ast_tool.invoke("print(square(5))") # 25
print("AST Result:", result)  # → 17

print("==========="*4)

repl  = PythonREPL()
code = "print(2 + 3 * 5)" # 17
code = "print(sum([1, 2, 3, 4]))" # 10
code = "import os; print(os.getcwd())" # d:\02.MyCode\GP-MyReference\13.MyLLM
result = repl.run(code) # repl.run(code), no!
repl.run("""
def square(x):
    return x * x
""")
result = repl.run("print(square(5))") # 25
print("Raw REPL Result:", result)  # → 10
~~~

|Item|`PythonAstREPLTool`|`PythonREPLTool`|`PythonREPL`|
|---|---|---|---|
|Execution Method|`eval()` + AST analysis|`exec()`|`exec()`|
|Safety|High (static analysis)|Low (direct execution)|Low|
|Expression Return|Automatically returned|Requires `print()`|Requires `print()`|
|State Persistence|None|Maintained|Maintained|
|Intended User|General users|LangChain users|Advanced users, tool devs|
|Location|`tools`|`tools`|`utilities`|

It’s worth considering which of these tools is best suited for your needs. I plan to allow the LLM to freely generate and execute code through an Agent. According to the [LangChain documentation](https://python.langchain.com/api_reference/experimental/tools/langchain_experimental.tools.python.tool.PythonREPLTool.html#), `PythonREPLTool` seems to be the primary REPL tool in use. If possible, I intend to use `PythonREPLTool` as well.

<!--
위 표로 각각의 도구중에 어떤것이 좋을지는 고민해 볼 필요가 있다. 나는 앞으로 Agent를 통해서 LLM이 자유롭게 코드를 생성하고 실행하게 할 예정이다. LangChain 문서에서 REPL도구로 PythonREPLTool을 주로 사용하고 있는거 같다. 나는 가능하면 PythonREPLTool로 사용하려 한다.
-->

## *TavilySearch*

When building LLM-powered applications, real-time access to the web can significantly boost the quality and accuracy of responses. One of the emerging tools that makes this easy in LangChain is **TavilySearch**—a fast and reliable web search API.

- Sign up for [Tavily](https://www.tavily.com/) and issue an API key.
     ![Tavily API](/assets/img/langgraph/2025-05-12-LangGraph1_1.png)
	_Tavily API_


```python
from langchain_tavily.tavily_search import TavilySearch
import os

os.environ["TAVILY_API_KEY"] = "tvly-dev-****************"

tavilysearch_tool = TavilySearch(
    max_results=3,
    include_answer=False,
    include_raw_content=True,
    # include_images=True,
    search_depth="basic", # or "advanced"
    # include_domains=["github.io", "wikidocs.net"],
    # exclude_domains = []
)
  
tavilysearch_tool.invoke({"query": "2010년~2024년 대한민국의 1인당 GDP는?."}) # ok
# tavilysearch_tool.invoke("2010년 ~ 2024년 대한민국의 1인당 GDP는?") # ok
```

```bash
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
