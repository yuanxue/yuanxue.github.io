---
layout: post
title: "Reflections on Richard Sutton's Interview: Intro"
date:   2025-10-05 00:17:56 -0700
tags: [AGI, Reinforcement Learning, LLM]
categories: [LLM, AI, Deep Learning]
---

{:toc}

I recently listened to [Richard Sutton’s interview on the *Dwarkesh Podcast*](https://www.youtube.com/watch?v=BF1aXbY0hS8). Often called the father of reinforcement learning, Sutton spoke candidly about the limitations of large language models (LLMs), the importance of continual learning, and his lifelong conviction that true intelligence arises from experience—not imitation.

For me, the conversation was both intellectually dense and personally inspiring. I approach it from two perspectives:

- As a **curious researcher**, I’m drawn to questions at the core of AGI: What does it mean to understand and to act with purpose?

- As a **practitioner**, I would ask a more practical question:  
> *If our goal is not AGI, but AI systems that function as capable knowledge workers—delivering real-world, economically valuable tasks—is the current path of LLM scaling still a feasible one? And what boundaries or blindspots should we be mindful of?*

These reflections led me to a set of questions:

- Is the problem with next-token prediction itself?

Next-token prediction defines pretraining and much of supervised fine-tuning. But modern post-training relies heavily on reinforcement learning—RLHF (human feedback) for human alignment and verifier-based RL (RLVF) for eliciting reasoning capabilities.
*Can post-training RL reshape a model’s behavior beyond pretraining limits—or does it remain bounded by what was initially imitated?*

- Is the limitation about when learning happens—training-time vs. deployment-time?

Sutton argues for continual learning: adapting through direct experience after deployment.
Most current models are trained once and then frozen, accessed via APIs. However, this boundary is beginning to soften. Techniques like Reinforcement Fine-Tuning (RFT) allow models to adapt based on customer-specific objectives within real environments. Cursor’s recent work(https://cursor.com/en-US/blog/tab-rl) on online RL demonstrate early but promising signs of on-the-fly adaptation. If models can learn continuously from real-world interaction, could this represent a meaningful step toward addressing Sutton’s critique?

- Is it about modality—text versus interaction-rich experience?

LLMs originated in language, yet sparked a multimodal wave: diffusion models, Gemini, Claude 3, and embodied predictive models like Genie 3, which predicts future video frames from passive observation. Can multimodal prediction serve as a proxy for world experience—or is true interactivity necessary for agency?

- If direct experience is essential, how do we scale it?

The world is vast; no single agent can live enough lifetimes to experience it fully. Humans transcend this through language—abstracting and transmitting experiences across generations. If experience is the key, how do we scale it to match the richness of human understanding without millions of years of simulation?  And if *language abstraction* is not the answer, how can we enable scalable experience—and ensure that learned world models are passed on and inherited?

These questions echo ideas from many voices—Sutton’s belief in experience, Ilya Sutskever’s conviction in next-token prediction, and Andrej Karpathy’s optimism about scaling.

This blog series is my attempt to connect these threads: to explore where these views align, where they diverge, and how they might evolve. My hope is that by structuring the questions clearly, we can reason a little better—collectively.
