---
layout: post
title: "Reflections on Richard Sutton's Interview: Intro"
date:   2025-10-05 00:17:56 -0700
tags: [AGI, Reinforcement Learning, LLM]
categories: [LLM, AI, Deep Learning]
---

{:toc}

I recently listened to [Richard Sutton’s interview on the *Dwarkesh Podcast*](https://www.youtube.com/watch?v=BF1aXbY0hS8). Often called the father of reinforcement learning, Sutton spoke candidly about the limitations of large language models (LLMs), the importance of continual learning, and his lifelong conviction that true intelligence arises from experience—not imitation.

For me, the conversation was both intellectually dense and personally inspiring. 

Sutton’s interview raised questions that reach beyond algorithms and touch on the philosophical core of modern AI research and the practical of LLM in support Agent develpment via reasoning.

As I listened, a few themes stood out.

---

### 1. Is next-token prediction a fundamental limitation—or an efficient way to compress knowledge?

The term *LLM* has become a kind of shorthand.  
Pretraining is indeed next-token prediction—but post-training introduces reinforcement learning from human or verifier feedback (RLHF, RLVF), injecting goals, preferences, and interaction loops.  
Are these systems still “just predicting the next word,” or have they quietly evolved beyond that description?

---

### 2. Is the issue about **when** learning happens—during training or after deployment?

Sutton often emphasizes continual learning: the ability to adapt through experience after deployment.  
Most models today are fixed—trained once, deployed behind APIs, and updated in large batches.  
But that boundary is softening.  
Techniques like reinforcement fine-tuning based on customer objectives, Cursor’s online RL tuning, and adaptive personalization models hint at systems that can keep learning safely in the wild.  
If models could adapt continuously after deployment, would that address Sutton’s critique?

---

### 3. Is the problem one of modality—text versus multi-modality?

LLMs began in text, yet they sparked the broader generative wave: diffusion models for vision, multimodal architectures like Gemini and Claude 3, and DeepMind’s *Genie 3*, a video-based world model that predicts and controls future frames.  
When video data provides sequences of passive perception and reaction, can we learn a world model from it—without explicit interaction or embodied experience?

---

### 4. Scaling experience: Can abstraction replace direct interaction?

The world is vast. Even humans, unlike squirrels, cannot experience everything firsthand.  
Language became our bridge—allowing us to share abstract concepts and accumulated knowledge across generations.  
If *experience* is the right paradigm, how can we scale it to the richness of human understanding without millions of years of simulation?  
And if *language abstraction* is not the answer, how can we enable scalable experience—and ensure that learned world models are passed on and inherited?

---

### 5. Can a system learn through **cultural feedback**, as humans do?

(Here, I turn to Joseph Henrich and Yuval Harari for insight.)  
Humans learn not only through direct reinforcement but also through the social feedback loops of culture—stories, norms, and shared values.  
Could an LLM, trained on human language and behavior, serve as a foundation for this kind of cultural learning?

---

### 6. The Value of Imitation

Each of these questions sits at the edge of what LLMs *are* and what true intelligence might *require*.  
They prompted me to reflect on past insights from many great minds:  
Ilya Sutskever’s optimism about scaling, Geoff Hinton’s biologically inspired view of learning, Yann LeCun’s vision of world models, and Demis Hassabis’s synthesis of agents, planning, and memory.  

Together, these perspectives trace a continuum of thought about what it means to learn and to be intelligent.  

This blog series is my attempt to connect those threads—  
to see how these ideas align, diverge, and evolve—  
and to ask whether, by organizing our questions clearly,  
we might reason a little better together.
