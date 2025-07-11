---
title: RAG Prologue
description: Let’s Use RAG Like This
author: DS2Man
date: 2025-02-04 11:00:00 +0000
categories: [LLM&RAG, L&R-Prologue]
tags:
  - LLM
  - RAG
math: true
pin: true
---

Since the public release of ChatGPT on November 30, 2022, large language models (LLMs) have driven revolutionary advancements in artificial intelligence, gaining attention across various industries.
LLMs are models trained on vast amounts of text data, enabling them to understand and generate human-like language. These models are widely applied in tasks such as natural language processing, translation, and text summarization.
In 2023, we witnessed the development of various LLMs, including OpenAI’s GPT-4, Google’s BERT, and Meta’s LLaMA. Each of these models has continued to evolve, leveraging unique features and strengths. Notably, Meta’s LLaMA series offers versions with 7 billion, 13 billion, and 70 billion parameters, showcasing exceptional performance in diverse natural language processing tasks.
The advancements in LLMs have brought about transformative changes in areas such as customer service, content creation, and education.
Through this tutorial, I aim to share the knowledge and insights I have gained. I will cover topics such as using publicly available models on Hugging Face, performing LLM inference with Ollama, applying various techniques with LangChain, and exploring AI agents and LangGraph. Please stay tuned and join me on this journey!

<!-- 
2022년 11월 30일 ChatGPT의 공개 이후, 대규모 언어 모델(Large Language Model, LLM)은 인공지능 분야에서 혁신적인 발전을 이끌며 다양한 산업에서 주목받고 있습니다. 
LLM은 방대한 텍스트 데이터를 학습하여 인간과 유사한 언어 이해 및 생성 능력을 갖춘 모델로, 자연어 처리, 번역, 텍스트 요약 등 여러 작업에서 활용되고 있습니다. 
2023년에는 OpenAI의 GPT-4, Google's BERT, Meta의 LLaMA 등 다양한 LLM이 개발되어 각자의 특성과 강점을 바탕으로 발전해왔습니다. 
특히, Meta의 LLaMA 시리즈는 7억, 130억, 700억 파라미터 버전으로 제공되며, 다양한 자연어 처리 작업에서 뛰어난 성능을 보여주고 있습니다. 
이러한 LLM의 발전은 고객 서비스, 콘텐츠 생성, 교육 등 다양한 분야에서 혁신적인 변화를 가져오고 있습니다.
이번 Tutorial통해서 제가 익히고 필요한 내용들을 공유하고자 합니다.
Huggingface에 공개된 모델 사용법, Ollama를 활용한 LLM 추론, LangChain을 활용한 다양한 기법, AI Agent, LangGraph 등을 공유할 예정이니, 관심있게 지켜봐 주세요. 
-->

## *Recommended For:*

This tutorial is perfect for those who relate to any of the following scenarios ~~(and, to be honest, all of these apply to me!)~~:

- Your company has restricted the use of ChatGPT due to security concerns.
- You want to run an LLM on your own computer but don’t know how to get started.
- You feel that ChatGPT gives irrelevant or nonsensical answers to slightly more technical questions, and you’re wondering if this can be improved.
- You want to build a dataset tailored to your domain and create your own version of ChatGPT.
- You’d like to share a ChatGPT model infused with your domain knowledge with others.

<!-- 
이런 분들께 추천합니다!
아래 내용에 해당되는 분들께서 본Tutorial을 보시면 좋을 것 같아요. (아래는 사실 전부 제 이야기입니다…)

회사에서 보안상 문제로 Chat GPT 사용에 제약을 걸었다.
LLM을 내 컴퓨터로 돌려보고 싶은데 어떻게 하는지 모르겠다.
조금만 전문적인 내용을 물어봐도 헛소리를 하는 것 같은데, 개선할 수 없을까?
나의 도메인에 해당되는 데이터셋을 구축해서 나만의 Chat GPT를 갖고 싶다.
나의 도메인 지식이 담긴 Chat GPT를 다른 사람에게 공유하고 싶다. 
-->

## *Limitations of LLMs: Hallucination*

LLMs rely heavily on the data they are trained on, which means they can generate inaccurate information due to biased or insufficient training data or model overfitting.<br>
Additionally, they often struggle to provide accurate answers for information that has not been included in their training, such as recent or unlearned updates.<br>
This phenomenon is referred to as **"hallucination"**.<br>
While the output may seem creative, it is important to remember that LLMs are ultimately systems driven by instructions and probabilities.

<!-- 
LLM의 한계: 환각(Hallucination) 
LLM은 학습된 데이터에 의존하기 때문에, 편향되거나 불충분한 학습 데이터, 모델의 과적합 등으로 인해서 부정확한 정보를 생성하곤 합니다. 
또한 학습하지 않은 최신 정보에 답변을 정확하게 제공하지 못하는 문제가 있습니다.이것을 환각(Hallucination)이라고 합니다. 
창의적으로 보이지만 결국 LLM도 명령과 확률에 의해 돌아가는 시스템이기 때문입니다.
-->


## *Overcoming the Limitations of LLMs: RAG*

To address the limitations of LLMs, various approaches have been proposed.   
One such approach is fine-tuning, which involves further training a pre-trained model on data from a specific domain (e.g., medical, legal, financial) to optimize the model.   
This allows the model to acquire domain-specific expertise and provide accurate, specialized responses. However, fine-tuning requires significant time and resources, and it may reduce the model's general applicability.   
In contrast, **RAG (Retrieval-Augmented Generation)** offers a way to maintain the model's generality and adaptability while generating accurate and reliable answers.   
RAG connects LLMs to external knowledge sources, allowing the model to overcome its limitations while leveraging its strengths. The concept of RAG is to provide the LLM with a "cheat sheet" when it encounters unfamiliar questions (e.g., questions involving the latest knowledge or untrained information). The model then uses this cheat sheet to generate informed and accurate responses.   
RAG stands out as a promising approach for addressing the challenges of LLMs while retaining their advantages.

<!-- 
LLM의 한계를 뛰어넘는 방법: RAG
이러한 LLM의 한계를 극복하기 위해 여러 방법이 제안되고 있습니다. 
그중 하나는 Fine tuning으로, 사전 학습 모델(pre-trained model)에 특정 도메인(예: 의료, 법률, 금융)의 데이터를 추가 학습시켜 모델을 최적화하는 방식입니다. 
이를 통해 모델은 특정 분야에 대한 전문 지식을 습득하고 정확하고 전문적인 답변을 제공할 수 있습니다. 
그러나 Fine tuning은 시간과 비용이 많이 소요되고, 모델의 범용성이 저하될 수 있습니다.
 
반면 RAG는 외부 지식 소스와 연계하여 모델의 범용성과 적응력을 유지하면서도 정확하고 신뢰할 수 있는 답변을 생성할 수 있습니다. 
즉, RAG(Retrieval-Augmented Generation)는 LLM의 한계를 극복하면서도 그 장점을 살릴 수 있는 접근 방식이라고 할 수 있습니다.
RAG의 컨셉은 LLM이 모르는 질문(최신 지식, 학습에 사용되지 않은 지식)을 받았을 때, 잘 대답하기 위한 Cheating Sheet를 제공해줘서 
Cheating Sheet에 있는 내용을 기반으로 LLM이 대답하게 만드는 기술입니다. 
-->

## Sequence of LLMs Using RAG

1. **User Query Input**  
   The user inputs a question or query.

2. **Data Retrieval and Segmentation for the Cheating Sheet**  
   Relevant data is gathered and segmented from sources such as web searches or documents (e.g., PDF, HTML).

3. **Data Embedding and Storage in a Vector Database**  
   Data is embedded into vector representations and stored in a vector database.

4. **Text Embedding of the Query**  
   The query is transformed into an embedding vector.

5. **Similarity-Based Retrieval via Retriever**  
   The retriever searches the vector database for data most similar to the query(e.g., cosine similarity).

6. **Prompt Context Construction**  
   Contexts for the prompt are created using the retrieved data.

7. **Response Generation by the LLM**  
   The LLM generates an answer based on the constructed prompt contexts.

<!--
1. 사용자의 질문 입력(Query)
2. Cheating Sheet위한 데이터 확보 및 분할(Web Search, Documents(e.g. PDF, Html)
3. 데이터 임베딩 이후 Vector DB로  저장
4. 질문에 대한 텍스트 임베딩(Embedding)
5. Retriever 통해 Vector DB와 Query 유사도 기반 검색
6. Prompt Contexts 구성
7. LLM의 답변 생성(Response)
-->
![RAG Intro](/assets/img/llm&rag/2025-02-04-RAG-Prologue_1.png)
_RAG Intro(Source: [NAVER CLOUD PLATFORM](https://www.ncloud-forums.com/topic/277/))_