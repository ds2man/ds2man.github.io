---
title: Wait
description: Let's execute Python code in LangChain.
author: DS2Man
date: 2025-05-16 11:00:00 +0000
categories: [LLM&RAG, L&R-Tools]
tags:
  - LangChain
  - Tools
math: true
pin: true
---

When using LangChain, there are times when it's essential to have the LLM execute Python code. To support this, LangChain offers various Python execution tools. In this post, we'll compare three main tools provided by the `langchain.experimental` module.

<!--
LangChain을 사용하다 보면 LLM에게 파이썬 코드를 실행시키는 기능이 꼭 필요할 때가 있습니다. LangChain은 이를 위해 다양한 Python 실행 도구를 제공합니다. 이 글에서는 LangChain Experimental 모듈에서 제공하는 세 가지 주요 도구를 비교해보겠습니다.
-->

## *1. Steps to Build LangChain*

1. Load the model and tokenizer using **transformers**
2. Configure the **pipeline**   
<ins>`These steps are identical to the basic usage of LLMs (explicit approach).`</ins>
3. Create a **HuggingFacePipeline** object for LangChain
4. Generate a **prompt**
5. Create a **LangChain**

<!--
1. Transformers : 모델과 토크나이저 로드
2. Pipeline : 파이프 라인 구성
 여기까지가 기본 LLM 사용법(명시적 접근)과 동일
3. HuggingFacePipeline : LangChain을 위한 HuggingFacePipeline 객체 생성
4. PromptTemplate : Prompt 생성
5. langchain : LangChain 생성
-->

~~~python
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

myModel = "Llama-3.2-3B-Instruct"

if myModel == "Llama-3.2-3B-Instruct":
    model_id = "./Pretrained_byGit/Llama-3.2-3B-Instruct"
else:
    model_id = "./Pretrained_byGit/polyglot-ko-1.3b"

# Step 1: Load the model and tokenizer using transformers
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = 0  # You can choose 0, 50256, or another token ID
    print(f"Pad token ID is set to: {tokenizer.pad_token_id}")
else:
    print(f"Pad token ID already set: {tokenizer.pad_token_id}")

# Step 2: Configure the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    pad_token_id=tokenizer.pad_token_id,
)

# Step 3: Create a HuggingFacePipeline object for LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Step 4:  Generate a prompt
if myModel == "Llama-3.2-3B-Instruct":    
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a friendly AI assistant. Your name is DS2Man. Please answer questions briefly.
    <|eot_id|><|start_header_id|>user<|end_header_id|>{question}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
else:
    template = "### Question: {question}### Answer:"

prompt = PromptTemplate.from_template(
    template
)

# Step 5: Create a LangChain
chain = prompt | llm
~~~

## *2. invoke*

1. **invoke** processes input data in a single instance and returns the response at once.
2. When a user provides input, the model generates the entire result and then returns it in one go.
3. The response is provided only after the model completes generating all the text.

<!--
1.`invoke`는 한 번에 입력 데이터를 처리하여 전한 응답을 반환하는 방식입니다.
2. 사용자가 입력을 주면, 모델은 전체 결과를 생성한 후 한꺼번에 반환합니다.
3. 모델이 생성하는 모든 텍스트가 완성된 후에 응답을 제공합니다.
-->

~~~python
reponse = chain.invoke({"question": "What is the capital of the United States?"})
print("Invoke Result:")
print(reponse)
~~~
```
Invoke Result: <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a friendly AI assistant. Your name is DS2Man. Please answer questions briefly. <|eot_id|><|start_header_id|>user<|end_header_id|>What is the capital of the United States? <|eot_id|><|start_header_id|>assistant<|end_header_id|> The capital of the United States is Washington, D.C.
```

## *3. stream*

1. **stream** is a method that partially returns results in real-time as the model generates text.
2. You can observe the process of the model generating the response text in real-time.

<!--
1.`stream`는 모델이 텍스트를 생성하는 동안 부분적으로 결과를 실시간으로 반환하는 방식입니다.
2. 모델이 응답 텍스트를 생성하는 과정을 실시간으로 확인할 수 있습니다.
-->

~~~python
from langchain_core.messages import AIMessageChunk

response = chain.stream({"question": "What is the capital of the United States?"})
print("Streamed Result:")
answer = ""
iflag = 0
for chunk in response:
    if isinstance(chunk, AIMessageChunk):
        if iflag == 1: print("The type of chunk is AIMessageChunk...")
        iflag += 1
        answer += chunk.content
        print(chunk.content, end="", flush=True)
    elif isinstance(chunk, str):
        if iflag == 1: print("The type of chunk is str...")
        iflag += 1
        answer += chunk
        print(chunk, end="", flush=True)
    else:
        if iflag == 1: print(f"The type of chunk is {type(chunk)}...")
        iflag += 1
~~~

```
Streamed Result:
The type of chunk is str...
The capital of the United States is Washington, D.C.
```
