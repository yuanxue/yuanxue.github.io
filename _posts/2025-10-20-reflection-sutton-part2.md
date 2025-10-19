---
layout: post
title: "Reflections on Richard Sutton's Interview: Part II — Goal and Acting"
date: 2025-10-20 00:00:00 -0700
tags: [Reinforcement Learning, LLM, AGI, Sutton]
categories: [LLM, AI, Deep Learning]
---

{:toc}

*Understanding describes the world; goals decide what to do in it.*

In [Part I](https://yuanxue.github.io/2025/10/06/reflection-sutton-part1.html), I discussed *world model* — what is it, why it is central to intelligence, how it can be built, how LLMs build such understanding, and where that understanding falls short.  In this second part, I turn to the other half of intelligence: **acting toward a goal**.  

---

### Sutton's View: Intelligence Requires a Goal

As Richard Sutton reminds us of John McCarthy’s classic definition:

> "Intelligence is the computational part of the ability to achieve goals."
— John McCarthy

Let’s begin again with the simplest form for expressing this objective. In reinforcement learning (RL), acting toward a goal is formalized through the **Markov Decision Process (MDP)**.

Recall that the world model (or dynamics model) describes how the environment changes and how rewards are generated based on the agent’s actions. This model provides the underlying components of the MDP:

$$
\begin{aligned}
s_t &= g(o_t) && \text{(state inference)} \\
s_{t+1} &\sim P(s_{t+1} \mid s_t, a_t) && \text{(stochastic transition)} \\
r_t &= r(s_t, a_t) && \text{(reward model)} \\
o_t &\sim P(o_t \mid s_t) && \text{(observation generation)}
\end{aligned}
$$

While the dynamics model tells us what will happen (the mechanics of the world),
the objective tells us what matters (the agent’s goal). We need a mathematical expression that formalizes this goal — one that captures intelligence as the ability to pursue desirable outcomes over time. This objective is formally defined as the expected cumulative discounted reward, or return:

**Objective** 

This objective is defined as the **expected cumulative discounted reward**, or *return*:

$$
J = \mathbb{E}\!\left[\sum_t \gamma^t r_t\right]
$$

where:

- $\mathbb{E}$ denotes the expected value of the reward sequence under the model’s behavior.  
- $\gamma \in [0,1]$ is the **discount factor**, ensuring that rewards received sooner are valued more highly than those received later.  
- $T$ is the **time horizon**, which may be finite or infinite.  

**Policy** 

To act toward a goal, the agent needs a rule to decide *what to do* in order to optimize its reward. This rule is called the **policy**. Concretely, a policy $\pi(a\mid s)$ is a **mapping from states to actions** that dictates the agent’s behavior at every time step.  It defines *how* the agent behaves, given what it currently knows about the world.  

The goal in reinforcement learning is to find the **optimal policy** — the one that maximizes the expected cumulative discounted reward (or *return*) over time.

In most cases, the policy is **stochastic**, meaning that the agent selects actions according to a probability distribution conditioned on the current state:

$$
a_t \sim \pi(a_t \mid s_t)
$$

This allows for exploration — the ability to sample different actions and discover better strategies over time.

Formally, the optimization objective can be expressed as:

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
$$

where $\tau = (s_0, a_0, s_1, a_1, \ldots)$ denotes a trajectory sampled according to the policy $\pi$.  The goal of learning is to find the policy $\pi^*$ that maximizes this expected return:

$$
\pi^* = \arg\max_{\pi} J(\pi)
$$


### Two Approaches to RL

At a high level, reinforcement learning methods fall into two broad categories, depending on whether the agent has access to (or learns) a dynamics model. The diagram below (from [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)) summarizes these families of RL algorithms:

![RL Algorithms Overview](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)


- **Model-based RL**
  Model-based methods use an explicit *world model* to simulate and plan future actions before execution. The key advantage is foresight — the ability to “think ahead,” evaluate the consequences of possible actions, and select the best one before acting.  
  By using its model to plan, the agent can distill the results of this reasoning process into a learned policy.  

  The main challenge is the availability of the dynamics (world) model. An imperfect model can mislead the agent into optimizing for its own errors — performing well within the learned model but failing in the real world. 

- **Model-free RL**
  Model-free methods, by contrast, does not need an explicit *world model*, rather, it learns *directly from experience*. Instead of simulating possible futures, the agent interacts with the environment, receives rewards, and gradually adjusts its behavior based on observed outcomes.  

  This family includes two major subcategories — **value-based** and **policy-based** approaches.

#### Value-Based Approach

Value-based methods estimate the long-term return of each state or action using a **value function** $V(s)$ or an **action-value function** (or **Q-function**) $Q(s, a)$. These functions predict how good it is to be in a certain state or to take a certain action, in terms of future rewards. The value function is one of the *four fundamental components* of Sutton’s *“base common model”* of the RL agent.

This estimation connects short-term actions to long-term goals — a mechanism for learning from *sparse rewards* that occur far in the future. Sutton’s *startup analogy* captures this idea well: Imagine an entrepreneur working toward a long-term goal: the reward may arrive once in ten years — the “exit” moment when the startup succeeds. To stay motivated and effective, humans construct intermediate signals of progress: milestones, customer feedback, or product growth. Each small improvement updates the belief that the long-term goal is achievable, reinforcing the behaviors that contribute to it.

In RL, the value function serves this same purpose. When an agent achieves a local success it updates its estimate of eventual victory. That increase in expected value immediately reinforces the move that led to it.  This mechanism, known as **temporal-difference (TD) learning**, allows the agent to convert distant goals into immediate learning signals.

In essence, the **value function** transforms long-term objectives into tangible, auxiliary predictive rewards that drive learning in the moment. 


#### Policy-Based Approach

The **policy** is another *fundamental component* of the RL agent based on Sutton’s *“base common model”* — the part that determines *how* it acts. Policy-based methods (also called **policy optimization**) take a different route from value-based ones. Rather than estimating value functions first, they directly **optimize the policy parameters** to improve expected performance.

A common technique, known as the **policy gradient method**, adjusts the policy parameters in proportion to the goodness of outcomes. Intuitively, actions that lead to higher returns are made more likely, while less successful actions are suppressed.  

Formally, the policy gradient can be written as:

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}\!\left[\nabla_\theta \log \pi_\theta(a|s) \, Q^\pi(s,a)\right]
$$

This equation expresses a simple but profound idea: the gradient of performance depends on how likely an action is under the current policy and how valuable that action turns out to be. By following this gradient, the agent continually improves its behavior through experience.

Sutton summarized this intuitively in his interview:  

> “You act, you see what happens, and you change your behavior accordingly — not because someone told you what to do, but because the world responded.”

This captures the essence of policy-based RL — **intelligence as the art of adjustment to feedback**.

---

### LLMs as Policies

Large Language Models (LLMs) can be viewed as **policy models** that map a given context (prompt) to a distribution over next-token actions. Sampling from the model corresponds to executing a stochastic policy $\pi_\theta(a_t \mid s_t)$, where:

- $s_t$: the current input sequence,  
- $a_t$: the token selected at time step $t$,  
- $\theta$: the model parameters.  

And language generation can be expressed as a **Markov Decision Process (MDP)**, where  

- Transition is deterministic: $s_{t+1} = s_t \circ a_t$,  
- Reward $r$ is a scalar value provided after sequence generation.  

To understand how this policy is learned — and how it differs from Sutton’s view — we must ask:  **Does it have a goal? And if so, what is that goal?**  

Let’s look at how large language models are trained to seek an answer. 

### From Behavior Cloning to Reinforcement Learning

LLMs undergo a multi-stage training process as **Pre-Training** and  **Post-Training**, which usually includes **Supervised Fine-Tuning (SFT)** and **Reinforcement Learning** with various types of rewards *Human Feedback (RLHF)*, *Verifiable Reward (RLVF)*, etc. 

#### Pretraining and SFT: Policy from Imitation

In pretraining, the policy is optimized for next-token prediction from pretraining data — learning to imitate human linguistic behavior through **behavior cloning**. This imitation-based paradigm effectively mirrors human language distributions rather than optimizing for outcomes. The model’s capability is limited by the quality and diversity of its training data. Without *goal-driven correction*, any flaws or biases in human data are inherited. 

In SFT, the policy is trained directly on **human demonstrations**.  Given an input context $s_t$, the model learns to predict the next token $a_t$ that aligns with a human-provided reference output. Formally, the objective minimizes the negative log-likelihood of the target sequence:

$$
\mathcal{L}_{\text{SFT}} = - \sum_t \log \pi_\theta(a_t^{*} \mid s_t)
$$

Here, $a_t^{*}$ represents the human-annotated “correct” token at each step.  
Through this process, the model learns to imitate expert behavior — optimizing per-token likelihood given the context, rather than optimizing for task-specific rewards.  SFT refines the base model to follow human intent more reliably, but it remains fundamentally **imitation-based**: the model learns *how humans respond*, not *why* they respond that way. As a result, SFT has the following limitations:

- Performance Ceiling: It struggles to exceed the quality of its demonstrations.  
- Limited Generalization: Unable to adapt to unseen context. When the model deviates from the demonstrated trajectory and encounters out-of-distribution states, it can not recover. Small initial errors can quickly compound, leading to significant performance degradation.
- High Data Costs: Curating large-scale, high-quality human data is expensive and slow.


    \item \textbf{Difficulty Surpassing Demonstrations:} The model struggles to achieve performance superior to the quality of the training data itself. It can only reproduce what it has seen, not innovate or discover more efficient strategies.
    \item \textbf{Limited Generalization:} Adapting to novel or significantly different scenarios not explicitly covered in the training data becomes challenging. The model may struggle with variations or complexities it hasn't directly observed.
    \item \textbf{Risk of Encoding Undesirable Patterns:} Any spurious correlations or undesirable behaviors present in the dataset can be inadvertently learned and propagated by the model, as it lacks an objective function to penalize such patterns.
    \item \textbf{High Cost of Data Acquisition:} Obtaining high-quality, diverse, and comprehensive expert demonstration data is often \textbf{extremely expensive, time-consuming, and labor-intensive}. This cost can be a major barrier, especially for complex tasks or those requiring highly specialized expertise, making it impractical to collect enough data to cover all possible scenarios.

#### Reinforcement Learning in Post-Training: Policy with a Goal

Post-training introduces **explicit goals** through reinforcement learning.  
In **RLHF (Reinforcement Learning from Human Feedback)**, the model is fine-tuned using a learned reward model reflecting human preference.  
In **RLAIF** or **RLVR**, rewards correspond to task success — such as solving a math problem or generating correct code.  

These techniques transform LLMs from passive imitators into systems that optimize for defined outcomes.  
\subsection{Reward Signals After the Action}

In the MDP formulation of LLMs, \textbf{reward is typically delayed} until a full response is generated. This is unlike classical RL environments that provide per-step rewards. Instead:
\begin{itemize}
    \item Rewards are computed \emph{after} the final token is generated,
    \item They may be based on human preferences, automated evaluators, or task success,
    \item This sparsity introduces the challenge of \textbf{credit assignment} across the full sequence of actions.
\end{itemize}

### Ground Truth and Correction

Reinforcement learning provides an intrinsic mechanism for **self-correction**.  
When an agent predicts a reward or transition, it receives immediate, objective feedback from the environment.  
This continuous loop enables adaptation through experience.  


### Open Reflections

#### Is Post-Training Correction Enough?

RLHF and RLVR bring purpose, but within fixed objectives.  
Recent RL research emphasizes the importance of **entropy** — maintaining curiosity and exploration to prevent premature convergence.  
Perhaps the next leap for LLMs lies in **continual, on-policy interaction** with the world: learning not just from curated feedback, but from the consequences of their own actions.

By contrast, LLMs only receive *ground truth* during training not during deployment.  
They lack experiential feedback on whether their generated outputs actually succeeded in the real world. Simulation envioronment

#### Reasoning as Exploration

Reasoning models and **Chain-of-Thought (CoT)** prompting implicitly introduce exploration.  
Each reasoning step expands the model’s internal trajectory, forming a kind of *mental simulation*.  
Inference-time scaling — generating multiple thought chains and selecting the best — resembles *policy rollout and evaluation*.  
Could this process serve as an internal feedback loop — a primitive form of on-policy learning?

---

*In this view, LLMs occupy a fascinating middle ground: not yet world-interacting agents, but systems learning to act within the space of language.  
They model behavior, receive sparse rewards, and adjust through preference feedback — inching closer to the reinforcement learning framework that Sutton once described as the “essence of intelligence.”*


==============



\subsection{From Behavior Cloning to Reinforcement Learning}

Large Language Models (LLMs) typically undergo an initial training phase comprising pretraining, followed by instruction tuning and supervised fine-tuning (SFT) \cite{osa2018bc}. During these stages, the models are primarily optimized for next-token prediction, learning to imitate sequences found in human-written datasets. This approach inherently functions as a form of \textbf{behavior cloning} or \textbf{offline imitation learning} \cite{osa2018bc, christiano2017deep}. In this paradigm, the agent mimics observed demonstrations without direct optimization for long-term success or an explicit task reward signal. The absence of such a scalar reward is a key distinction from reinforcement learning, where an agent actively seeks to maximize cumulative rewards through interaction with an environment.

Imitation-based training, while effective for learning directly from demonstrations, faces three core limitations that significantly impact model quality, real-world applicability, and deployment costs:

\begin{itemize}
\item Lack of an Explicit Reward Objective
Unlike reinforcement learning, imitation learning doesn't optimize for a direct scalar reward signal or long-term task success. Instead, it aims to replicate observed behaviors. This means the model doesn't inherently understand the \textbf{ultimate goal} or what truly constitutes optimal behavior. It can't discern ``good'' from ``bad'' actions beyond what's shown, leading to two major issues:
\begin{itemize}[noitemsep,topsep=0pt]
    \item \textbf{Suboptimal Performance:} If the expert demonstrations contain any flaws, biases, or suboptimalities (which is common with human data), the model will learn and perpetuate these. It has no mechanism to identify or correct its own mistakes, preventing it from surpassing the expert's performance.
    \item \textbf{Compounding Errors:} When the model deviates from the demonstrated trajectory and encounters out-of-distribution states (situations not seen during training), it lacks the guidance of a reward signal to recover. Small initial errors can quickly compound, leading to significant performance degradation or even catastrophic failures.
\end{itemize}

\item Data-Bound Constraints and Acquisition Cost
The model's knowledge and capabilities are inherently \textbf{confined to the demonstrated behaviors} within its training data. This limitation manifests in several ways, exacerbated by the practical challenge of data acquisition:
\begin{itemize}[noitemsep,topsep=0pt]
    \item \textbf{Difficulty Surpassing Demonstrations:} The model struggles to achieve performance superior to the quality of the training data itself. It can only reproduce what it has seen, not innovate or discover more efficient strategies.
    \item \textbf{Limited Generalization:} Adapting to novel or significantly different scenarios not explicitly covered in the training data becomes challenging. The model may struggle with variations or complexities it hasn't directly observed.
    \item \textbf{Risk of Encoding Undesirable Patterns:} Any spurious correlations or undesirable behaviors present in the dataset can be inadvertently learned and propagated by the model, as it lacks an objective function to penalize such patterns.
    \item \textbf{High Cost of Data Acquisition:} Obtaining high-quality, diverse, and comprehensive expert demonstration data is often \textbf{extremely expensive, time-consuming, and labor-intensive}. This cost can be a major barrier, especially for complex tasks or those requiring highly specialized expertise, making it impractical to collect enough data to cover all possible scenarios.
\end{itemize}

#### **Post-Training: Policy with a Goal**

SFT is imitation, 

Post-training introduces goals through **reinforcement learning**.  
In **RLHF (Reinforcement Learning from Human Feedback)**, the goal is to align outputs with human preferences encoded in a reward model.  
In **RLVR** and related techniques, the objective becomes *task success* — e.g., solving a math problem or generating correct code.

These phases transform LLMs from pure imitators into systems that optimize for explicit outcomes.  
Yet the transformation is partial: the model learns within narrow, human-specified boundaries.  
It is still far from an open-ended learner that explores and redefines its goals through experience.

---
Ground Truth and Correction: RL provides an intrinsic mechanism for self-correction. When an agent predicts a reward or a transition, it gets immediate, objective feedback from the environment. This direct experience is the ground truth that enables perpetual learning and adaptation. LLMs only have "ground truth" during training (did I predict the right next word?), not during deployment (did my generated instructions actually lead to success in the world?).

### Open Reflections

#### **Is Post-Training Correction Enough?**

explore in the neighborhood, base model is good, 
if base model does not cover the neighorhood of the state, then 

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
