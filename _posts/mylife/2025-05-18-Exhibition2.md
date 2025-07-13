---
title: A Review of the AWS Summit Seoul 2025 
description: 
author: DS2Man
date: 2025-05-18 11:00:00 +0000
categories: [Mylife, Exhibition]
tags:
  - Mylife
  - Exhibition
math: true
pin: true
---

From May 14 to 15, 2025, I attended the **AWS Summit Seoul**, held at COEX. Although the event is held annually, this was my first time attending. As I mentioned in a previous post, I try to participate in as many conferences and workshops as possible to gain diverse experiences rather than settling into the comfort of the internal company environment.

## *Why I Attended*

Currently, I’m working on **generative AI services** at my company. I’ve started serving quantized LLMs in an **isolated (air-gapped)** server environment, and I’m preparing to launch a POC-type LLM service utilizing our manufacturing database through LangChain. Before doing so, I was curious to see how AWS approaches generative AI and wanted to compare it with the service I’ve been building—thus, I decided to attend this conference.

## *Conference Experience*

![AWS2025](/assets/img/mylife/2025/2025-05-15-AWS2025_1.png)
The event took place on **the 1st and 3rd floors of COEX**, and admission was **free with prior registration**, which I completed ahead of time.  (~~Even though I registered a month in advance, the QR code didn’t arrive until just two days before the event, which made me slightly anxious.~~)  Instead of just browsing the exhibition booths—which I felt wouldn’t provide deep insights—I focused my schedule around **workshops**.  However, since the workshops ran from 11 a.m. to 6 p.m., I missed most of the main stage sessions after the keynote.  It was a bit disappointing to head straight home after the workshop, so next time, I’m considering attending for both days to explore more leisurely.

![AWS2025](/assets/img/mylife/2025/2025-05-15-AWS2025_2.png)
As expected, the AWS ecosystem was vast. A wide range of topics was covered across various sessions.  Still, I was particularly excited to attend the **Generative AI Development Workshop using Amazon Bedrock**, which made the keynote all the more enjoyable.

## *Workshop Field Notes*

![AWS2025](/assets/img/mylife/2025/2025-05-15-AWS2025_3.png)
The workshop was held in **Conference Room 402 on the 4th floor of COEX**.  
The main topics included how to write AI-powered reports using **generative BI (QuickSight Q)** and how to build and deploy an **LLM assistant using AI Agents**. At my company, we use Spotfire as our BI tool, while AWS offers **QuickSight Q** as a comparable solution.  In practice, it felt somewhat lacking compared to Spotfire—possibly because I’m more familiar with the latter.  
Still, AWS’s approach to **automating data analysis and report generation using generative AI** was noteworthy and promising.    
The part that intrigued me the most was definitely the **LLM service based on AI Agents**.  
Midway through the workshop, I had about 40 minutes of free time, so I visited the main stage on the first floor.  I was lucky enough to find **one empty seat** and was able to join the session. Out of nearly 150 seats, every single one was taken—talk about perfect timing.

## *LLM Service Based on AI Agents*

The official title of the session I attended was **"Bedrock Multi-Agent Workshop in AWS Summit 2025 Seoul."**  I first learned about **Amazon Bedrock** during the **AW2025 event** in March, where **Hyundai AutoEver** introduced their **SD Brain** project.  Through a post-presentation conversation with one of the developers, I learned that the system had been built in the **Bedrock environment** in collaboration with AWS.  That may have been what ultimately led me to attend this workshop.

![AI_Agent](/assets/img/mylife/2025/2025-05-15-AWS2025_4.png)
The workshop did a great job of clearly explaining the concept of AI Agents—especially **why we need to scale from Single-Agent to Multi-Agent setups**, and how agents can **collaborate** with one another.  Real examples made these concepts easy to understand.

![AI_Agent](/assets/img/mylife/2025/2025-05-15-AWS2025_5.png)
This workshop also marked my first hands-on experience with **Amazon SageMaker**, which was quite **eye-opening**.  Following the manual took some time, but seeing the process actually result in a deployed service was impressive.  (~~Of course, as the saying goes, “the devil is in the details”—when building a real-world service, there would be much more to consider from the design stage onward.~~)  Once again, I realized how far ahead the world outside of my company really is.By the time I finished going through the tutorial and completing the hands-on exercises, **four hours had flown by**,  and I was so immersed that I didn’t even manage to take a single picture.

## *Final Thoughts*

Attending **AWS Summit Seoul 2025** gave me a clearer understanding of the **direction generative AI is heading** and the **technological trends** that are shaping it.  
In particular, the **Multi-Agent implementation using Amazon Bedrock** provided valuable insights that will help shape the direction of my future work in manufacturing AI.  
The **AI Agent development** project I plan to begin in the second half of this year will no doubt benefit from the experiences I gained through this workshop.