---
title: Svelte Prologue
description: Let’s Use Svelte Like This
author: DS2Man
date: 2025-03-23 11:00:00 +0000
categories: [Frontend, S-Prologue]
tags: [Frontend]
math: true
pin: true
---
The reason I started this blog is because of LLMs, a form of generative AI. I believe that the manufacturing industry will evolve beyond simple automation toward autonomous manufacturing, and LLMs will be the most crucial technology driving this change.

To integrate LLMs into semiconductor manufacturing, models like ChatGPT, Claude, and Grok are essential. However, before deploying such models, it is critical to have a user interface (UI) that enables them to be used as actual services. No matter how powerful a model may be, if it isn’t easily accessible and usable on the shop floor, real-world adoption becomes difficult.

In late 2024, Jensen Huang introduced the concept of "Physics AI." But I believe that before we get there, we first need to establish an interface where humans and LLMs can communicate naturally.

However, in a corporate environment, due to security concerns and technical limitations, there is no flexible UI available to utilize LLMs effectively. Moving forward, we need to embrace new technologies such as MCP (Model Context Protocol) to drive innovation in manufacturing, but the current situation leaves much to be desired.

While I was exploring these ideas, I came across an open-source project called [OpenWebUI](https://openwebui.com/). After trying it out, I found that it already included most of the features I had envisioned. It inspired me to consider building a service of my own based on it.

The only problem was that I had little to no experience in frontend development. So, starting on March 24, I began diving into the related technologies. I learned that OpenWebUI is built with Svelte, Vite, TypeScript, and Tailwind CSS. Among these four, I decided to first explore the frontend framework Svelte/Sveltekit (including Vite), which forms the core of the UI layer.

<!-- 
내가 블로그를 시작하게 된 계기는 LLM이라는 생성형 AI 때문이다. 앞으로의 제조 산업 생태계는 단순한 자동화를 넘어 자율제조로 나아갈 것이고, 이때 가장 핵심적인 기술이 바로 LLM이라고 생각한다. 
반도체 제조에 LLM을 도입하기 위해서는 ChatGPT, Claude, Grok과 같은 모델들이 필요하지만, 그 이전에 이런 모델들을 실제로 서비스하기 위한 UI가 필수적이다. 아무리 뛰어난 모델이 있어도, 현장에서 쉽게 접근하고 사용할 수 없다면 실질적인 도입은 어렵다. 
2024년 말, 젠슨 황은 ‘Physics AI’라는 개념을 제시했는데, 나는 그 이전 단계에서 사람과 LLM이 자연스럽게 소통할 수 있는 인터페이스가 선행되어야 한다고 생각한다. 

그러나 회사 환경에서는 보안상의 문제, 기술력 한계 등으로 Flexible하게 LLM을 사용할 수 있는 UI가 없는 상황이다. 앞으로는 MCP(Model Context Protocol) 등의 새로운 기술을 받아들여 제조의 혁신을 이루어야 하는데, 아쉬움이 많은 상황이였다.

그렇게 고민하던 중 오픈소스로 공개된 [OpenWebUI](https://openwebui.com/)를 알게 되었고, 실제로 사용해보니 내가 구상하던 기능들이 대부분 들어 있었다. 이걸 바탕으로 직접 서비스를 만들어보면 좋겠다는 생각이 들었다.

유일한 문제는 나에게 프론트엔드 개발 경험이 거의 없다는 점이었다. 3월 24일부터 관련 내용을 본격적으로 파악하기 시작했고, OpenWebUI가 Svelte, Vite, TypeScript, 그리고 Tailwind CSS로 구성되어 있다는 것을 알게 되었다. 4가지 항목 중에서 프론트엔드 프레임워크인 Svelte/Sveltekit(Vite 포함)에 관련된 내용부터 파악해 보고자 한다. 
-->

## *Recommended For:*

This tutorial is perfect for those who relate to any of the following scenarios ~~(and, to be honest, all of these apply to me!)~~:

- A UI is essential for effectively utilizing LLMs, and OpenWebUI is the optimal solution for that.  
- In an on-premise environment, it is crucial to rapidly develop various functionalities of LLMs.
- This time, I’d like to take the opportunity to learn frontend development.

<!-- 
이런 분들께 추천합니다!
아래 내용에 해당되는 분들께서 본Tutorial을 보시면 좋을 것 같아요. (아래는 사실 전부 제 이야기입니다…)

LLM을 효과적으로 사용하기 위한 UI가 필요한데, OpenWebUI가 최적이다.
온프레미스 환경에서 LLM의 다양한 기능들을 빨리 개발해야 한다.
Frontend를 이번에 배워보고 싶다.
-->

## *OpenWebUI*

**Open WebUI is an extensible, feature-rich, and user-friendly self-hosted AI platform designed to operate entirely offline.** It supports various LLM runners like **Ollama** and **OpenAI-compatible APIs**, with **built-in inference engine** for RAG, making it a **powerful AI deployment solution**.

- [OpenWebUI official website](https://docs.openwebui.com/)
- [OpenWebUI github](https://github.com/open-webui/open-webui)

![OpenWebUI](https://docs.openwebui.com/assets/images/demo-d3952c8561c4808c1d447fc061c71174.gif)
_OpenWebUI_

