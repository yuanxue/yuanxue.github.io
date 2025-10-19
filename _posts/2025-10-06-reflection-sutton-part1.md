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

> "They learn what people would say, not what would happen," Sutton notes.

Put in another way, an LLM is a **statistical mirror** of human linguistic behavior. It reflects how people describe reality, not how reality unfolds.  

But this raises a question: if LLMs truly lack a model that connects actions to consequences — the essence of understanding — then how do they still appear to know? When we ask LLM with a promot, “If I touch a boiling pot, what will happen?” the model will respond, “you’ll likely burn your skin immediately.” How should we reconcile this ability to state consequences with Sutton’s claim that LLMs don’t possess a world model? Or put more simply — is the ability to describe a consequence through language good enough, or even equivalent, to understanding the consequence itself?

### What Is a World Model?

To unpack Sutton’s view, we must first understand what a *world model* is, why it is central to intelligence, and how it can be built. 

Let’s begin with its simplest and most fundamental form: the dynamics model in reinforcement learning — a predictive model that describes how states evolve in response to actions.

A dynamics model is a function, often written as $\hat{f}$, that predicts the next state $s_{t+1}$ and reward $r_t$ given the current state $s_t$ and action $a_t$:

$$s_{t+1},\ r_t \approx \hat{f}(s_t, a_t)$$

With such a model, an **agent** can predict possible outcomes before acting — an ability fundamental to planning and reasoning.
In reality, we rarely have direct access to the true state of the world. Instead, we rely on observations from various senses — vision, hearing, touch — to infer an internal representation of the hidden state. State transition is also stochastic, reflecting ambiguity in how actions affect future states. A conceptual **world model** can therefore be represented as:

$$
\begin{aligned}
s_t &= g(o_t) && \text{(state inference)} \\
s_{t+1} &\sim P(s_{t+1} \mid s_t, a_t) && \text{(stochastic transition)} \\
r_t &= r(s_t, a_t) && \text{(reward model)} \\
o_t &\sim P(o_t \mid s_t) && \text{(observation generation)}
\end{aligned}
$$

### How a World Model Should Be Built

Building a world model involves three intertwined processes:

1. **Exploration** — Gathering Experience

    To learn the dynamics of an environment, an agent must act within it. It explores different states and actions, collecting trajectories of experience:

    $$
    (s_t, a_t, r_t, s_{t+1})
    $$

    These trajectories encode causal relationships — how actions lead to consequences — forming the empirical foundation of the model.

2. **Perception** — Interpreting Observations

    In most real-world settings, the true state $s_t$ is often hidden. Instead, the agent observes sensory inputs $o_t$ (images, sounds, text) and infer latent states:

    $$
    s_t \approx g(o_t)
    $$

    Here, $g$ is a perceptual encoder that extracts task-relevant information from raw sensory data, transforming perception into internal understanding.

3. **Learning** - Modeling the Dynamics

    Finally, learning integrates exploration and perception into a coherent predictive framework. Given collected trajectories and inferred latent states, the agent learns a stochastic transition function that approximates how the world evolves: 

    $$
    \hat{f}(s_t, a_t) \rightarrow (s_{t+1}, r_t)
    $$

    This process typically involves representation learning (to encode  latent states), sequence modeling (to capture temporal dependencies), and probabilistic estimation (to handle ambiguity). 
    
    With deep learning becoming the de facto approach for representation due to its expressive power, large neural networks now serve as the core mechanism for modeling dynamics. They are trained on sequences of state transitions as ground truth, where each step in the trajectory supervises the model to minimize prediction error between predicted and observed outcomes. Through this iterative process, the learned model captures both environmental regularities and stochastic variability — forming the foundation of world model learning.

### How a World Model Is Built in LLMs

Now, let’s turn to large language models (LLMs).
LLMs do not interact with the physical world directly. Instead, they learn from humans — observers and actors who have already explored the world and recorded their experiences as language. As shown in the diagram below, LLMs are trained on traces of human experience: vast text corpus that reflects how people perceive, reason about, and describe the world around them. Language serves as a powerful medium — distilling complex sensory, emotional, and causal interactions into symbols and sequences. Next-token prediction, the objective of pretraining, provides the concrete mechanism through which this learning occurs.

<div class="mermaid" markdown="0">
graph TD
A[Human Interaction with the World] -->|Trajectory| B[Experience Expressed in Language]
B -->|Text Corpus| C[LLM Pretraining]
C -->|Next-Token Prediction| D[LLM as Learned Model]
</div>

### The Limitations and Possible Paths Forward

This process clearly introduces several fundamental limitations:

* Limited Exploration:
The exploration originates from human experience, producing a narrow trajectory that reflects only a fraction of the world. The model cannot act or gather new evidence beyond what humans have already recorded.

* Limited Perception and Representation:
LLMs perceive only text — a symbolic medium through which humans express sensory, visual, spatial, and auditory experiences. Inevitably, some information is lost or distorted in translation. This representational gap leads to incomplete or inaccurate estimates of the underlying state of the world.

Acknowledging these limitations, it is worth examining what learning truly means in this context — and whether next-token prediction provides a sufficient form of self-supervision and deep neural networks, serve as powerful function approximators, capable of capturing complex dependencies and representations at scale. 

On this point, Ilya Sutskever — in his conversation with Dwarkesh Patel (same Youtube Podcast) *[Why Next-Token Prediction Could Surpass Human Intelligence](https://www.youtube.com/watch?v=Yf1o0TQzry8)* — offered a particularly insightful perspective - Sutskever views next-token prediction as a profoundly deep task that extends far beyond surface-level statistics:

> “Predicting the next token well means that you understand the underlying reality that led to the creation of that token.” While the process is statistical, Sutskever notes that the compression required to perform it effectively forces the model to internalize what it is about the world that produces those statistics.

If this interpretation holds, then approaches such as increasing the diversity of pretraining data through large-scale simulation environments or expanding perception through multimodal models become practical and valuable directions for advancing LLMs.

Still, exploration remains the hardest frontier — models continue to rely on fixed set of human-collected trajectories. True world modeling requires interactive systems that can act, observe, and revise their own understanding through experience. 

**Continuous learning** represents a promising direction — adapting models within real-world environments through domain-specific fine-tuning. Such post-training adaptation could mark a step toward bridging static knowledge and experiential learning.
This also raises a compelling question: if we place an LLM in a new environment after pretraining, how well could it explore within that domain and generate new representations grounded in its evolving experience?

Viewed from another angle, we might ask whether a complete world model is even necessary.
Humans themselves operate under incomplete world models — constrained by limited perception and biased exploration — yet still act intelligently. Thus, the question shifts from “Can LLMs learn the whole world?” to “Can they reason, act, and gather enough information to accomplish the task at hand?”

That brings us to the next part: Goal and Policy — where understanding meets action.