---
layout: post
title: "Reflections on Richard Sutton's Interview: Part I — World Model and Understanding"
date: 2025-10-06 00:00:00 -0700
tags: [Reinforcement Learning, LLM, World Model, Sutton]
categories: [LLM, AI, Deep Learning]
mermaid: true
---

{:toc}

### Sutton’s View: Lack of World Model in LLMs

Sutton begins with a simple observation: **LLMs never experience the world.** He argues that these systems **lack a world model** — a sense of cause and effect, surprise, and correction.  
Without it, they cannot truly understand.

> “They learn what people would say, not what would happen,” Sutton notes.

An LLM is a **statistical mirror** of human linguistic behavior. It reflects how people describe reality, not how reality unfolds.  

But this raises a question: if LLMs truly lack a model that connects actions to consequences — the essence of understanding — then how do they still appear to know? When we ask LLM with a promot, “If I touch a boiling pot, what will happen?” the model will respond, “you’ll likely burn your skin immediately.” How should we reconcile this ability to state consequences with Sutton’s claim that LLMs don’t possess a world model? Or put more simply — is the ability to describe a consequence through language good enough, or even equivalent, to understanding the consequence itself?

### What Is a World Model?

To unpack Sutton’s view, we must first understand what a *world model* is, why it is central to intelligence, and how it can be built. 

Let’s begin with its simplest and most fundamental form: the dynamics model in reinforcement learning — a predictive model that describes how states evolve in response to actions.

A dynamics model is a function, often written as $\hat{f}$, that predicts the next state $s_{t+1}$ and reward $r_t$ given the current state $s_t$ and action $a_t$:

$$s_{t+1},\ r_t \approx \hat{f}(s_t, a_t)$$

With such a model, we can predict possible outcomes before acting — an ability fundamental to planning and reasoning.
In reality, we rarely have direct access to the true state of the world. Instead, we rely on observations from various senses — vision, hearing, touch — to infer an internal representation of the hidden state. State transition is also stochastic, reflecting ambiguity in how actions affect future states. A conceptual world model can therefore be represented as:

$$
\begin{aligned}
s_t &= \text{Enc}(o_t) && \text{(state inference)} \\
s_{t+1} &\sim P(s_{t+1} \mid s_t, a_t) && \text{(stochastic transition)} \\
r_t &= g(s_t, a_t) && \text{(reward model)} \\
o_t &\sim P(o_t \mid s_t) && \text{(observation generation)}
\end{aligned}
$$

### How a World Model Should Be Built

Building a world model involves three intertwined processes:

1. Exploration — Gathering Experience

    One must act in the world to learn its dynamics. It explores different states and actions, collecting trajectories of experience:

    $$
    (s_t, a_t, r_t, s_{t+1})
    $$

    Through this process, we uncover causal relationships — how actions produce outcomes.

2. Perception — Interpreting Observations

    In real environments, the true state \( s_t \) is often hidden. One perceives observations \( o_t \) (images, sounds, text) and infer latent states:

    $$
    s_t \approx g(o_t)
    $$

    Here, \( g \) is a perceptual encoder that extracts the relevant information from raw sensory data.

3. Representation - Modeling the Dynamics

    $$
    \hat{f}(s_t, a_t) \rightarrow (s_{t+1}, r_t)
    $$

### How a World Model Is Built in LLMs

Now, let’s turn to LLMs. Unlike embodied agents, they do not explore the physical world; instead, they learn from human experience recorded as language.


<div class="mermaid" markdown="0">
graph TD
A[Human Interaction with the World] -->|Trajectory| B[Experience Expressed in Language]
B -->|Text Corpus| C[LLM Pretraining]
C -->|Next-Token Prediction| D[LLM as Learned Model]
</div>


LLMs are trained on the trajectories of humans interacting with the world.
Language acts as the medium through which those experiences are compressed and transmitted.

However, this introduces several gaps:

Limited Perception: LLMs observe only text — not sensory, spatial, or causal data.

Limited Exploration: The exploration policy comes from humans; the model cannot try new actions or observe new consequences.

Biased Representation: The distribution of text reflects cultural, social, and cognitive biases of its authors.

Despite these gaps, some argue that next-token prediction can implicitly build a world model.

### The Case For LLMs as World Models

Ilya Sutskever offers a counter-view: that large-scale next-token prediction is itself a form of world-model learning.

Compression Argument:
To predict the next token accurately, a neural network must compress reality.
It must infer latent structures behind text — physical facts, social dynamics, emotions, intentions.
In this view, accurate prediction requires implicitly modeling the causes behind human language.

World Manifest in Language:
Since all human experience — physical, emotional, cultural — is encoded in language, mastering language could mean mastering the structures of the world itself.

Next-Token Prediction Is Enough:
Sutskever challenges the notion that imitation limits intelligence.
A model that perfectly predicts human text, he suggests, might even surpass its teachers — because understanding the generator of the data can enable extrapolation beyond it.


### The Limitations and Possible Paths Forward

Sutton’s skepticism remains grounded in a different intuition — experience matters.
Without feedback, there is no surprise; without surprise, no learning beyond imitation.

Still, progress is being made to close this gap:

Perception: Multimodal models extend beyond text, integrating vision, audio, and embodiment.

Representation: Research (e.g., Anthropic’s findings on internal representations) shows LLMs learning latent world structures spontaneously.

Exploration: The hardest part — models still rely on human-collected trajectories.
True world modeling may require interactive agents that can experiment, observe, and revise their own understanding.

Yet perhaps the lesson is more nuanced.
Humans themselves operate under incomplete world models — limited perception, biased exploration — yet still act intelligently.
So the question shifts from “Can LLMs learn the whole world?” to “Can they reason and act effectively within their limited one?”

That brings us to the next part: Goal and Policy — where understanding meets action.