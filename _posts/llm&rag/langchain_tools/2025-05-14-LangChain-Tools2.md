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

LangChain offers a wide variety of built-in tools for search, math, web APIs, and more. but sometimes, you need something specific to your use case. That’s where **custom tools** come in.

In this post, we’ll explore what custom tools are, how to create them in LangChain, and when you should consider building your own.

## *How to Create a Custom Tool*

There are two ways to create a tool: using `@tool` or `Tool` from the `langchain_core.tools.tool` module. For flexible customization, the `@tool` decorator is more useful, so we'll use that. (The `Tool`-based approach will not be covered in this post.)

**[Note!]** If possible, wrap built-in tools as custom tools as well for the sake of code consistency!

### *python_repl_tool*

```python
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool

python_repl_tool_instance = PythonREPLTool()

@tool
def python_repl_tool(
    query: str,
):
    """
    Executes Python code in a REPL. 

    To show the result to the user, use `print(...)`. 
    Only printed output will be returned and visible.
    """
    result = ""
    try:
        result = python_repl_tool_instance.invoke(query)
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
# result = python_repl_tool.invoke(code) # ok!
result = python_repl_tool.invoke({"query" : code}) # ok!
print("Result1:", result)  # → 50

python_repl_tool.invoke("""
def square(x):
    return x * x
""")
# result = python_repl_tool.invoke("print(square(5))") # 2ok!
result = python_repl_tool.invoke({"query":"print(square(5))"}) # ok!
print("Result2:", result)  # → 50
```

```bash
============================================
tool name: python_repl_tool
tool description: Executes Python code in a REPL. 

To show the result to the user, use `print(...)`. 
Only printed output will be returned and visible.
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
    Search the web using Tavily and return a list of relevant documents.

    Provide a search query as input.
    Returns a list of documents containing raw content from search results.
    Use this tool to retrieve up-to-date web information.
    """
    try:
        return tavily_search_instance.invoke(query)
    except Exception as e:
        error_msg = f"[Tavily Search Error] {type(e).__name__}: {str(e)}"
        print(error_msg)
        return [{"error": error_msg}]
```

```python
print("==========="*4)
print(f"tool name: {tavily_search_tool.name}")
print(f"tool description: {tavily_search_tool.description}")
print("==========="*4)

result = tavily_search_tool.invoke("2010년 ~ 2024년 대한민국의 1인당 GDP는?")
result = tavily_search_tool.invoke({"query": "2010년 ~ 2024년 대한민국의 1인당 GDP는?"})
result
```

```bash
============================================
tool name: tavily_search_tool
tool description: Search the web using Tavily and return a list of relevant documents. 

Provide a search query as input. 
Returns a list of documents containing raw content from search results. 
Use this tool to retrieve up-to-date web information.
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
    """
    Searches Google News for recent articles related to the given keyword.

    Returns a list of dictionaries, each containing article metadata such as title, link, and content.
    """
    try:
        return googlenews_instance.search_by_keyword(query, k=3)
    except Exception as e:
        error_msg = f"[GoogleNews Error] {type(e).__name__}: {str(e)}"
        print(error_msg)
        return [{"error": error_msg}]
```

```python
print("==========="*4)
print(f"tool name: {googlenews_search_tool.name}")
print(f"tool description: {googlenews_search_tool.description}")
print("==========="*4)

googlenews_search_tool.invoke("AI 뉴스 알려줘") # ok
# googlenews_search_tool.invoke({'query':"AI 뉴스 알려줘"}) # ok
```

```bash
============================================ 
tool name: googlenews_search_tool 
tool description: Searches Google News for recent articles related to the given keyword.

Returns a list of dictionaries, each containing article metadata such as title, link, and content.
============================================

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
```

```python
print("==========="*4)
print(f"tool name: {get_current_datetime_tool.name}")
print(f"tool description: {get_current_datetime_tool.description}")
print("==========="*4)

# get_current_datetime_tool.invoke("What is today's date?") # ok
get_current_datetime_tool.invoke({'query':"What is today's date?"}) # ok
```

```bash
============================================ 
tool name: get_current_datetime_tool 
tool description: Returns the current date and time in a human-readable format. 

Format: 'Today is Month Day, Year at HH:MM.' 
Example: 'Today is July 11, 2025 at 16:45.' ============================================

'Today is July 10, 2025 at 22:11.'
```

### *pdf_retriever_tool*

```python
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool

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
```

```python
print("==========="*4)
print(f"tool name: {pdf_retriever_tool.name}")
print(f"tool description: {pdf_retriever_tool.description}")
print("==========="*4)

# pdf_retriever_tool.invoke("삼성전자가 개발한 생성형 AI 관련 내용을 문서에서 찾아줘") # ok
pdf_retriever_tool.invoke({'query':"삼성전자가 개발한 생성형 AI 관련된 정보를 문서에서 찾아주세요."}) # ok
```

```bash
============================================
도구 이름: pdf_search_tool
도구 설명: Search information from the PDF document using semantic similarity.

Provide a question or keyword, and this tool will return relevant chunks from the document.
============================================
SPRi AI Brief |  
2023-12월호
10
삼성전자, 자체 개발 생성 AI ‘삼성 가우스’ 공개
n 삼성전자가 온디바이스에서 작동 가능하며 언어, 코드, 이미지의 3개 모델로 구성된 자체 개발 생성 
AI 모델 ‘삼성 가우스’를 공개
n 삼성전자는 삼성 가우스를 다양한 제품에 단계적으로 탑재할 계획으로, 온디바이스 작동이 가능한 
삼성 가우스는 외부로 사용자 정보가 유출될 위험이 없다는 장점을 보유
KEY Contents
£ 언어, 코드, 이미지의 3개 모델로 구성된 삼성 가우스, 온디바이스 작동 지원
n 삼성전자가 2023년 11월 8일 열린 ‘삼성 AI 포럼 2023’ 행사에서 자체 개발한 생성 AI 모델 
‘삼성 가우스’를 최초 공개
∙정규분포 이론을 정립한 천재 수학자 가우스(Gauss)의 이름을 본뜬 삼성 가우스는 다양한 상황에 
최적화된 크기의 모델 선택이 가능
∙삼성 가우스는 라이선스나 개인정보를 침해하지 않는 안전한 데이터를 통해 학습되었으며, 
온디바이스에서 작동하도록 설계되어 외부로 사용자의 정보가 유출되지 않는 장점을 보유
∙삼성전자는 삼성 가우스를 활용한 온디바이스 AI 기술도 소개했으며, 생성 AI 모델을 다양한 제품에 
단계적으로 탑재할 계획
n 삼성 가우스는 △텍스트를 생성하는 언어모델 △코드를 생성하는 코드 모델 △이미지를 생성하는 
이미지 모델의 3개 모델로 구성
∙언어 모델은 클라우드와 온디바이스 대상 다양한 모델로 구성되며, 메일 작성, 문서 요약, 번역 업무의 
처리를 지원
∙코드 모델 기반의 AI 코딩 어시스턴트 ‘코드아이(code.i)’는 대화형 인터페이스로 서비스를 제공하며 
사내 소프트웨어 개발에 최적화
∙이미지 모델은 창의적인 이미지를 생성하고 기존 이미지를 원하는 대로 바꿀 수 있도록 지원하며 
저해상도 이미지의 고해상도 전환도 지원
n IT 전문지 테크리퍼블릭(TechRepublic)은 온디바이스 AI가 주요 기술 트렌드로 부상했다며,

∙이 플랫폼은 데이터 관리, 모델 배포와 평가, 신속한 엔지니어링을 위한 종합 도구 모음을 제공하여 
다양한 기업들이 맞춤형 AI 모델을 한층 쉽게 개발할 수 있도록 지원
∙생성 AI 개발에 필요한 컴퓨팅과 데이터 처리 요구사항을 지원하기 위해 AI 플랫폼(PAI), 
데이터베이스 솔루션, 컨테이너 서비스와 같은 클라우드 신제품도 발표
n 알리바바 클라우드는 AI 개발을 촉진하기 위해 올해 말까지 720억 개 매개변수를 가진 통이치엔원 
모델을 오픈소스화한다는 계획도 공개
☞ 출처 : Alibaba Cloud, Alibaba Cloud Launches Tongyi Qianwen 2.0 and Industry-specific Models to Support 
Customers Reap Benefits of Generative AI, 2023.10.31.

Ⅰ. 인공지능 산업 동향 브리프

2023년 12월호
Ⅰ. 인공지능 산업 동향 브리프
 1. 정책/법제 
   ▹ 미국, 안전하고 신뢰할 수 있는 AI 개발과 사용에 관한 행정명령 발표  ························· 1
   ▹ G7, 히로시마 AI 프로세스를 통해 AI 기업 대상 국제 행동강령에 합의··························· 2
   ▹ 영국 AI 안전성 정상회의에 참가한 28개국, AI 위험에 공동 대응 선언··························· 3
   ▹ 미국 법원, 예술가들이 생성 AI 기업에 제기한 저작권 소송 기각····································· 4
   ▹ 미국 연방거래위원회, 저작권청에 소비자 보호와 경쟁 측면의 AI 의견서 제출················· 5
   ▹ EU AI 법 3자 협상, 기반모델 규제 관련 견해차로 난항··················································· 6
 
 2. 기업/산업 
   ▹ 미국 프런티어 모델 포럼, 1,000만 달러 규모의 AI 안전 기금 조성································ 7
   ▹ 코히어, 데이터 투명성 확보를 위한 데이터 출처 탐색기 공개  ······································· 8
   ▹ 알리바바 클라우드, 최신 LLM ‘통이치엔원 2.0’ 공개 ······················································ 9
   ▹ 삼성전자, 자체 개발 생성 AI ‘삼성 가우스’ 공개 ··························································· 10
   ▹ 구글, 앤스로픽에 20억 달러 투자로 생성 AI 협력 강화 ················································ 11
```