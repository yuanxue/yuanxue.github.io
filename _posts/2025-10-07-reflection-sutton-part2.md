---
layout: post
title: "Reflections on Richard Sutton's Interview: Part II — Goal and Acting"
date: 2025-10-07 00:00:00 -0700
tags: [Reinforcement Learning, LLM, AGI, Sutton]
categories: [LLM, AI, Deep Learning]
---

{:toc}

In [Part I](#), I explored the idea of a *world model* — how systems like large language models (LLMs) understand the world, and where that understanding falls short.  
In this second part, I turn to the other half of intelligence: **acting toward a goal**.  
Understanding describes the world; goals decide what to do in it.

---

### Intelligence Requires a Goal

As Richard Sutton reminds us, *“Intelligence is about achieving goals in the world.”*  
A system can model the world endlessly, but without a purpose, its knowledge remains inert.  
Where Part I focused on learning the world’s dynamics, this part centers on **learning to act** — the bridge from prediction to decision.

In reinforcement learning (RL), this bridge is formalized through the **Markov Decision Process (MDP)**, defined by a tuple \( (S, A, P, R, \gamma) \):
- \( S \): states  
- \( A \): actions  
- \( P \): transition dynamics (the world model)  
- \( R \): reward function  
- \( \gamma \): discount factor for future rewards  

The world model tells us *what happens*; the reward tells us *what matters*.

---

### From Dynamics to Objective

Once we can model transitions \( f(s, a) \rightarrow (s', r) \), we can define the agent’s objective:

\[
J(\pi) = \mathbb{E}_{\pi}\!\left[\sum_t \gamma^t r_t\right]
\]

This objective expresses intelligence as *optimization through experience*.  
The agent learns a **policy** \( \pi(a|s) \) — a mapping from states to actions — that maximizes expected cumulative reward.  

Two broad approaches emerge:
- **Model-based RL**, which plans actions using an explicit world model.
- **Model-free RL**, which learns directly from trial and error, without ever constructing the model explicitly.

---

### Two Broad Approaches to RL

#### **Value-Based Approach**

Value-based methods estimate the long-term return of each state or action via a **value function** \( V(s) \) or **action-value function** \( Q(s, a) \).  
This estimation connects short-term decisions to long-term outcomes — a way to see beyond the immediate reward.  
Sutton’s *startup analogy* illustrates this: an early-stage company may take actions that lose money now, but yield value later.  
Without a sense of long-term value, intelligent action collapses into short-sighted imitation.

#### **Policy-Based Approach**

Policy-based methods take a different route. Instead of estimating value first, they directly **optimize the policy parameters** to improve expected performance.  
This approach, known as **policy gradient**, adjusts the policy in proportion to the goodness of outcomes:

\[
\nabla_\theta J(\pi_\theta) = \mathbb{E}\!\left[\nabla_\theta \log \pi_\theta(a|s) \, Q^\pi(s,a)\right]
\]

In Sutton’s interview, he put it simply:  
> “You act, you see what happens, and you change your behavior accordingly — not because someone told you what to do, but because the world responded.”  

This line captures the spirit of policy-based RL: intelligence as the art of adjusting to feedback, not imitating prescriptions.

---

### The LLM Analogy: Policy Without and With Goals

#### **Pretraining: Policy from Imitation**

A pretrained LLM can be viewed as a **policy** \( \pi_\text{LLM}(a_t | s_t) \) that imitates human linguistic behavior.  
It generates the next token by predicting what *a human would likely say next*.  
This is statistical learning — powerful but passive.  
Such a model lacks goal-driven correction; it learns human *descriptions* of success, not the *experience* of success itself.

#### **Post-Training: Policy with a Goal**

Post-training introduces goals through **reinforcement learning**.  
In **RLHF (Reinforcement Learning from Human Feedback)**, the goal is to align outputs with human preferences encoded in a reward model.  
In **RLVR** and related techniques, the objective becomes *task success* — e.g., solving a math problem or generating correct code.

These phases transform LLMs from pure imitators into systems that optimize for explicit outcomes.  
Yet the transformation is partial: the model learns within narrow, human-specified boundaries.  
It is still far from an open-ended learner that explores and redefines its goals through experience.

---

### Open Reflections

#### **Is Post-Training Correction Enough?**

RLHF and RLVR bring purpose, but within limited and static objectives.  
Recent work in RL emphasizes the importance of **entropy** — the capacity to keep exploring.  
High-entropy policies resist premature convergence, maintaining curiosity in uncertain environments.  
Perhaps the next leap for LLMs lies in *continual, on-policy interaction* with the world — learning not just from labeled feedback but from the consequences of their own actions.

#### **Reasoning as Exploration**

Reasoning models and **Chain-of-Thought (CoT)** prompting introduce implicit exploration.  
Each reasoning step expands the model’s internal trajectory beyond the training corpus, forming a kind of *mental simulation*.  
Inference-time scaling — generating multiple thought chains before selecting the best one — resembles *policy rollout and evaluation*.  
Could this process act as an internal feedback loop, a primitive form of on-policy learning?

---

### Closing Thought

Understanding is half of intelligence; acting completes it.  
A world model provides *prediction*, but a goal gives *direction*.  
As Sutton has long argued, the essence of intelligence lies in the continual cycle of **acting, observing, and adjusting**.  
Only when knowledge and purpose form a closed loop can we say a system truly learns from the world — not just about it.
