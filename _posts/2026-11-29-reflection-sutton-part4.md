Reward sparse
1) value function
2) self-supervision to learn presentation. 

Human knowledge is the limit to optimization space. 


Bootstrap and limitation.

Bitter lesson 

Part I Addition:

### TODO: Is Genie 3 a World Model

Demis Hassabis and Google DeepMind explicitly describe Genie 3 as a "world model" and a crucial step toward Artificial General Intelligence (AGI). 
**What Makes Genie 3 a World Model** Unlike traditional Large Language Models (LLMs) that only generate text, Genie 3 generates interactive, dynamic 3D environments from a single text prompt. This is vital because it addresses the missing elements in current LLMs. 
* Real-time Interaction: The environments are playable and can be navigated in real-time (e.g., at 24 frames per second). This allows an agent (or a user) to act within the world, not just watch a generated video.
* Physical Consistency/Memory: Genie 3 is designed to maintain "long-horizon consistency" and a form of short-term visual memory. If you walk past an object and come back, it's still there. This is critical for an AI to learn object permanence and the causality of the physical world—an agent's actions have consequences that persist.

**How it is trained**

Genie 3 is trained on massive amounts of Video Data
* Learning from Observation: The model's "experience" comes from watching countless hours of video (likely including gameplay footage, public video clips, and real-world video) from the internet.
* Intuitive Physics: By observing how things move and interact in these videos—how a ball falls, how water splashes, how a person's perspective changes when they walk—the model develops an "intuitive understanding of physics" and causality. It learns the rules of the world not from explicit programming (like a game engine), but from pattern recognition in data, much like how a human child develops common sense.
Genie is trained using a technique that infers the action that must have occurred between frames, even though the video data is unlabeled (meaning no one explicitly tagged it with "move right," "jump," etc.).
**Latent Action Model**: 

Inferred Actions: Genie includes a component called a Latent Action Model. This model is trained to look at two consecutive frames in a video and figure out the most compact, discrete "action" that explains the change between them.

The "Why" of Motion: For example, if frame t shows a character standing still and frame t+1 shows the character a bit to the right, the model infers a discrete token that represents "move right." It is forced to learn a small set of these abstract, latent actions that are consistent across all the videos it sees.

Unsupervised Learning: This is done entirely without human-provided action labels, a form of unsupervised learning. The training objective is to successfully predict the next frame, but it can only do this accurately if it first correctly infers the underlying action that caused the change.

While Genie 3 itself is trained on observational data (videos), its primary value as a world model is to create a dynamic, interactive environment where other AI systems, called agents (like DeepMind's SIMA), can learn through experience. The primary strategic purpose is to provide an "unlimited curriculum" of rich, flexible simulation environments to train AI agents like robots and autonomous systems. This allows agents to learn through trial and error (reinforcement learning) in a safe, cost-effective manner before deployment in the real world

Prediction is the Key to Interaction: Once the Latent Action Model is trained, the main Dynamics Model is trained to predict the next frame. Crucially, it predicts the next frame based on the previous frame AND an action token.

Training for Cause and Effect: During training, it learns: Next Frame=f(Previous Frames,Inferred Action). This teaches the model the causal relationship—what happens when a specific action is taken in a specific world state.

Replacing the Inferrer: When a human or an AI agent uses Genie 3 in real-time, the Latent Action Model (the part that infers the action) is thrown out.

External Control: Instead, the user's input (like pressing the 'D' key on the keyboard) is mapped to one of the learned latent action tokens (e.g., the "move right" token).

Interactive Generation: The model then uses the Dynamics Model to generate the next frame based on the previous frame and the action token it just received from the user: New Frame=f(Previous Frames,User’s Latent Action).

In short, DeepMind ingeniously turned passive observation (video) into an active experience by training a sub-model to reverse-engineer the "controller input" from the visual changes in the video, and then replacing that inferrer with a real-time user controller.

Genie bridges this gap by incorporating a latent action model to infer an action space from observation and then using that to enable real-time interaction. While Genie is built on a large-scale generative model foundation, its core technical leap is an embrace of the RL paradigm: it's not just generating video; it's generating a controllable world where actions lead to predictable outcomes.

For Sutton, models like Genie represent a step toward his vision because they are fundamentally about predicting what will happen when an action is taken, a hallmark of an experience-based world model, even if the training data started as passive video.

TODO: 

https://gemini.google.com/app/ace75346537b8674
Yann LeCun is highly skeptical that Large Language Models (LLMs) like ChatGPT can ever achieve true human-level intelligence (often referred to as AGI). Instead, he advocates for an architectural approach centered on World Models.

LeCun's View on LLMs
LeCun views current LLMs as fundamentally limited due to their reliance on language and token prediction. Key criticisms include:

Lack of World Understanding: LLMs are primarily trained on text, which, in LeCun's view, represents only a small, constrained, and serialized version of human knowledge. They lack a true, grounded understanding of the physical world, its objects, and the laws of common sense physics.

No True Reasoning or Planning: LLMs operate like a "System 1" (fast, reactive, and intuitive) in the Daniel Kahneman framework. They produce one token after another through a fixed amount of computation, making them reactive and incapable of the deliberative, long-term reasoning and planning required for complex tasks. They primarily exploit statistical patterns from their training data rather than performing genuine causal reasoning.

Data Inefficiency: While LLMs are trained on massive text datasets (trillions of tokens), LeCun argues that humans (e.g., a four-year-old child) acquire an equivalent amount of data through high-bandwidth visual and sensory perception in a fraction of the time, demonstrating the superior efficiency of non-linguistic learning.

"Obsolete" Architecture: He has stated that LLMs are a technological dead-end for achieving human-level intelligence and advises young developers to focus on the next generation of AI systems.

LeCun's Vision for World Models
LeCun believes the path to advanced machine intelligence lies in systems that can build and utilize World Models.

What is a World Model? A world model is an internal, abstract representation of the structure, dynamics, and causal relationships of the environment. It allows an intelligent agent to predict consequences, plan action sequences, and perform reasoning. Humans and animals constantly run simulations based on their mental world models to navigate reality.

The Goal: Prediction and Planning: AI systems need to learn how the world works by observing it, primarily through sensory data like videos, rather than just text. By predicting the "next state of the world" given a potential action, the AI can plan a sequence of actions to reach a goal.

Proposed Architecture: LeCun's research at Meta focuses on the Joint Embedding Predictive Architecture (JEPA). This approach aims to create abstract representations of the physical world based on multi-modal input, allowing the system to predict how its internal representations will evolve, which is far more efficient than predicting raw, high-dimensional inputs like every pixel in a video frame.

Key Capabilities: World Models are designed to give AI the capabilities LLMs lack: understanding the physical world, having persistent memory, and the ability to reason and plan hierarchically.

The video below features Yann LeCun discussing the limitations of Large Language Models and the need for new approaches.

Yann LeCun: We Won't Reach AGI By Scaling Up LLMS
This YouTube clip is relevant because Yann LeCun directly explains why he believes scaling up Large Language Models is not the way to achieve Artificial General Intelligence.