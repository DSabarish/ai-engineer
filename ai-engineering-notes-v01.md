# 🤖 AI Engineering & LLM APIs — Complete Study Notes
### Topic: Getting Started with Text Generation LLM APIs (OpenAI API)
### Platform: Scaler Academy | Date: 19 Feb – 28 Feb 2026

---

## 📋 TABLE OF CONTENTS

1. [What is AI Engineering?](#1-what-is-ai-engineering)
2. [Types of LLM Models](#2-types-of-llm-models)
3. [The AI Engineering Ecosystem](#3-the-ai-engineering-ecosystem)
4. [Key Generation Parameters](#4-key-generation-parameters)
5. [Temperature — Deep Dive](#5-temperature--deep-dive)
6. [Top-P (Nucleus Sampling)](#6-top-p-nucleus-sampling)
7. [Top-K Sampling](#7-top-k-sampling)
8. [Greedy Search vs Beam Search](#8-greedy-search-vs-beam-search)
9. [How to Choose the Right Model](#9-how-to-choose-the-right-model)
10. [Prompt Engineering Techniques](#10-prompt-engineering-techniques)
11. [API Message Structure](#11-api-message-structure)
12. [RAG — Retrieval-Augmented Generation](#12-rag--retrieval-augmented-generation)
13. [Vector Databases & FAISS](#13-vector-databases--faiss)
14. [AI Agents](#14-ai-agents)
15. [Eval Pipelines for AI Apps](#15-eval-pipelines-for-ai-apps)
16. [LLM Frameworks & Tools Ecosystem](#16-llm-frameworks--tools-ecosystem)
17. [LLM Infrastructure & Deployment](#17-llm-infrastructure--deployment)
18. [Code Examples](#18-code-examples)
19. [Acronym Glossary](#19-acronym-glossary)
20. [Comparison Tables](#20-comparison-tables)
21. [Interview Important Points](#21-interview-important-points)
22. [Quick Revision Notes](#22-quick-revision-notes)
23. [Top 20 Takeaways](#23-top-20-takeaways)
24. [Interview Q&A with Answers](#24-interview-qa-with-answers)
25. [Beginner-Friendly Explanation](#25-beginner-friendly-explanation)
26. [Advanced Concepts](#26-advanced-concepts)
27. [One-Page Cheat Sheet](#27-one-page-cheat-sheet)

---

## 1. What is AI Engineering?

### 1.1 Definition
AI Engineering is the practice of **building intelligent applications** by integrating, orchestrating, and deploying **pre-trained foundation models** (like GPT, Claude, Llama). It is fundamentally different from traditional ML Engineering.

### 1.2 AI Engineering vs Traditional ML Engineering

| Aspect | Traditional ML Engineering | AI Engineering |
|---|---|---|
| **Core Activity** | Training models from scratch | Integrating pre-trained models |
| **Key Skills** | Statistics, math, distributed training | Prompt design, API usage, orchestration |
| **Data Need** | Huge labeled datasets | Minimal (for prompting/RAG) |
| **Tools** | PyTorch, TensorFlow, Spark | LangChain, FAISS, OpenAI API |
| **Primary Output** | Trained model weights | Working AI-powered application |
| **Cost** | Very high (GPU clusters) | Low to moderate (API calls) |
| **Focus Areas** | Feature engineering, model architecture | Prompt design, RAG, agent orchestration, eval |

> ⚡ **Key Insight:** AI Engineers don't usually train models — they **use** powerful pre-built models and build systems on top of them.

### 1.3 Key Emphasis Areas in AI Engineering
- **Prompt design and optimization** — crafting inputs that get the best outputs
- **RAG (Retrieval-Augmented Generation)** — combining search with generation
- **Agent orchestration** — coordinating LLMs with tools and control loops
- **Evaluation and observability** — measuring how well the system performs
- ❌ **NOT in scope:** Training models from scratch (that is ML Engineering territory)

---

## 2. Types of LLM Models

### 2.1 The Three Model Types

| Model Type | Training Method | Behavior | Use Cases |
|---|---|---|---|
| **Base** | Next-token prediction on raw internet text | Continues text, does NOT follow instructions | Text completion, fine-tuning starting point |
| **Instruct** | Fine-tuned on instruction→response pairs | Follows single-turn instructions | Q&A, summarization, single tasks |
| **Chat** | Fine-tuned on multi-turn conversations | Handles dialogue, maintains context | Chatbots, assistants, conversations |

### 2.2 The Evolution Path

```
Raw Internet Text
        │
        ▼
Pre-training (Next Word Prediction)
        │
        ▼
Base Model (e.g. GPT-2)
    — Only predicts next word
    — Does NOT follow instructions
    — Cannot answer questions properly
        │
        ▼
Fine-tuning on Instruction→Response Pairs (SFT)
        │
        ▼
Instruct Model
    — Follows single-turn instructions
    — Can answer questions, summarize, etc.
        │
        ▼
Fine-tuning on Multi-Turn Conversation Data
    + RLHF (Reinforcement Learning from Human Feedback)
        │
        ▼
Chat Model (e.g. ChatGPT, Claude)
    — Maintains conversation context
    — Aligned to be helpful, harmless, honest
```

### 2.3 Why Base Models Fail at Q&A — The Key Proof

When GPT-2 (a base model) is asked: *"What is the capital of France?"*  
It outputs: *"France is not a country like most countries in Europe, where you can have a government that controls property..."*

**Why?** Because GPT-2 doesn't know what a "question" is. It just saw words and predicted the most likely continuation from its training data (Wikipedia articles about France).

**More failures of GPT-2:**
- Asked to write a poem → outputs a random URL
- Asked to summarize → just repeats the input sentence
- Asked "What is 2+2?" → outputs nonsense definition of a cube

**The fix:** Load an Instruct/Chat model (e.g. `TinyLlama-1.1B-Chat-v1.0`) and use the **chat template format**:
```
<|user|>
{prompt}
</s>
<|assistant|>
```
Output becomes: *"Yes, the capital of France is Paris."* ✅

### 2.4 The Importance of Chat Templates

Each instruction-tuned model is trained on a specific conversation structure. If you don't use the right tags, the model gets confused.

- **Llama-style models:** Use `<|user|>`, `<|assistant|>`, `</s>` tags
- **OpenAI API:** Handled automatically via the `role` field (`system`, `user`, `assistant`)
- **Hugging Face Pipeline:** Automatically applies the right template when you pass a list of dictionaries with `role` and `content`

---

## 3. The AI Engineering Ecosystem

### 3.1 Major API Platforms

| Platform | Type | Notes |
|---|---|---|
| **OpenAI API** | Closed-source, paid | Industry standard, widely used, large ecosystem |
| **Anthropic (Claude)** | Closed-source, paid | Strong reasoning, safety-focused |
| **Google (Gemini)** | Closed-source, paid | Multimodal capabilities |
| **Hugging Face** | Open-source hub | Thousands of models, self-hostable |
| **DeepSeek** | Open-weights | Strong reasoning model, free/cheap |

> ⚡ **Key Insight:** OpenAI's API format has become the *de facto* standard. Many providers (Mistral, Together AI, DeepSeek) follow the same API structure. Learning OpenAI API syntax transfers broadly.

### 3.2 Popular Models Reference

| Model | Provider | Tier | Best For |
|---|---|---|---|
| **GPT-5.2** ⭐ | OpenAI | Latest/Best | Most capable general use |
| **GPT-4.1** | OpenAI | Previous gen | Strong reasoning + generation |
| **GPT-4o-mini** | OpenAI | Budget | Fast, cheap general tasks |
| **GPT-5-mini** | OpenAI | Compact | Smaller, faster |
| **GPT-5-nano** | OpenAI | Lightest | Resource-constrained |
| **o1, o3-mini** | OpenAI | Reasoning | Math, code, complex logic |
| **Claude Opus** | Anthropic | Premium | Deep reasoning, long context |
| **Claude Sonnet** | Anthropic | Balanced | Good quality, faster |
| **DeepSeek R1** | DeepSeek | Open-weights reasoning | Math, logic, cheaper |
| **Gemini 1.5** | Google | Multimodal | Long context, vision |

---

## 4. Key Generation Parameters

When calling an LLM API, these parameters control how the model generates text:

### 4.1 Temperature
Controls the **randomness/creativity** of output.
- **Low (0.0–0.3):** Deterministic, focused, always picks most likely token → Use for math, code, factual Q&A
- **High (0.7–2.0):** Random, creative, picks less likely tokens → Use for creative writing, brainstorming

### 4.2 Top-P (Nucleus Sampling)
Considers only tokens whose **cumulative probability** reaches threshold P.
- Dynamically selects a shortlist: keeps adding most likely words until their total probability = P
- Removes the bottom (1-P)% of tokens from consideration
- More adaptive than Top-K

### 4.3 Top-K
Considers only the **top K most likely** tokens at each step.
- Example: Top-K = 50 → always considers exactly 50 candidates
- Simpler but less adaptive than Top-P

### 4.4 Greedy Search
Always picks the **single most likely next token**.
- Fast, deterministic
- Risk: gets stuck in repetitive or "local optimum" sequences

### 4.5 Beam Search
Keeps track of **top K candidate sequences** (beams) at every step.
- K = beam width (e.g. 4 means track 4 paths simultaneously)
- Slower and memory-intensive but produces more coherent, logical text
- Used heavily in: Machine Translation, Summarization

### 4.6 Quick Reference

| Parameter | Controls | Low Value Effect | High Value Effect |
|---|---|---|---|
| Temperature | Randomness | Deterministic | Creative/Random |
| Top-P | Vocabulary breadth | Small focused set | Large diverse set |
| Top-K | # of candidates | Few options | Many options |
| Beam Width | # of paths tracked | Like greedy | More thorough search |

---

## 5. Temperature — Deep Dive

### 5.1 How It Works Mathematically

The LLM outputs raw scores called **logits** for every token in its vocabulary (~50,000 tokens). Temperature modifies these scores before the Softmax function converts them to probabilities.

**The Formula:**
$$\text{softmax}_T(z_i) = \frac{e^{z_i/T}}{\sum_{j=1}^{n} e^{z_j/T}}$$

Where:
- `z_i` = raw score (logit) for token i
- `T` = Temperature
- The output is the probability for token i

### 5.2 Step-by-Step Softmax Calculation

**Given:** Prompt: "I love to ride ___"  
**Raw scores (logits):** A=2.0, B=1.0, C=0.1

**Step 1 — Apply Temperature (divide by T):**
- At T=1: scores unchanged [2.0, 1.0, 0.1]
- At T=0.5: scores become [4.0, 2.0, 0.2] (magnified → sharper)
- At T=2.0: scores become [1.0, 0.5, 0.05] (shrunk → flatter)

**Step 2 — Numerical Stability Trick (subtract max):**
```
Original: [2.0, 1.0, 0.1]
Subtract max (2.0): [0, -1.0, -1.9]
```
⚠️ **Why?** Without this, `e^1000` would cause numerical overflow and crash the program. Subtracting the max keeps numbers manageable while being mathematically equivalent.

**Step 3 — Apply Exponential (e^x):**
```
e^0    = 1.000
e^-1   = 0.368
e^-1.9 = 0.150
```

**Step 4 — Normalize (divide by sum):**
```
Sum = 1 + 0.368 + 0.150 = 1.518
A = 1/1.518     ≈ 0.659 (65.9%)
B = 0.368/1.518 ≈ 0.242 (24.2%)
C = 0.150/1.518 ≈ 0.098 (9.8%)
```

### 5.3 Temperature Comparison Table (Same Logits)

| Temperature | Token A Probability | Token B Probability | Token C Probability | Entropy | Effect |
|---|---|---|---|---|---|
| **T = 0.5** | **86.7%** | 11.7% | 1.6% | 0.44 | Very sharp, highly confident |
| **T = 1.0** | **66.5%** | 24.5% | 9.0% | 0.83 | Default balanced behaviour |
| **T = 2.0** | **50.6%** | 30.7% | 18.6% | 1.02 | Flat, creative |
| **T = 5.0** | **40.2%** | 32.9% | 26.9% | 1.09 | Very flat, highly random |

> **Entropy** = measure of uncertainty/randomness. Higher entropy = more random output.

### 5.4 The "Sharpening" Effect Explained

**At T = 0.5 (Low Temperature):**
- Original scores: [2.0, 1.0, 0.1]
- After dividing by 0.5: [4.0, 2.0, 0.2]
- **Gap between A and B:** was 1.0, now 2.0 — bigger gap → A dominates even more

**At T = 2.0 (High Temperature):**
- After dividing by 2.0: [1.0, 0.5, 0.05]
- **Gap between A and B:** was 1.0, now 0.5 — smaller gap → B and C get more chances

### 5.5 Practical Usage Guide

| Use Case | Recommended Temperature | Reason |
|---|---|---|
| Math problems | 0.0 – 0.2 | Need correct, consistent answers |
| Code generation | 0.0 – 0.3 | Need valid, working syntax |
| Factual Q&A | 0.0 – 0.3 | Need accurate facts |
| Data extraction | 0.0 | Deterministic output needed |
| Chatbot replies | 0.5 – 0.7 | Balanced, natural responses |
| Creative writing | 0.7 – 1.2 | Want variety and creativity |
| Brainstorming | 1.0 – 1.5 | Want diverse ideas |
| Poetry/Fiction | 1.2 – 2.0 | Maximum creativity |

---

## 6. Top-P (Nucleus Sampling)

### 6.1 What It Is
Top-P (also called Nucleus Sampling) dynamically selects a shortlist of tokens based on their **cumulative probability**.

### 6.2 Step-by-Step Example

**Vocabulary (50,000 tokens), sorted by probability:**
```
Token   | Probability | Cumulative Sum
--------|-------------|---------------
"the"   | 0.40        | 0.40
"a"     | 0.25        | 0.65
"owl"   | 0.15        | 0.80  ← STOP (p=0.80 reached)
"my"    | 0.10        | 0.90
"an"    | 0.05        | 0.95
"ow."   | 0.03        | 0.98
...49,994 more tokens...
```

**With p = 0.80:**
1. Add tokens top-to-bottom until cumulative sum ≥ 0.80
2. The "nucleus" = {"the", "a", "owl"} — only 3 out of 50,000 tokens qualify
3. All other 49,997 tokens get probability 0

**Step 3 — Renormalization:**
After dropping the bottom tokens, the remaining probabilities sum to 0.80 (not 1.0). Must rescale:
```
"the": 0.40 / 0.80 = 0.50 (50%)
"a":   0.25 / 0.80 = 0.31 (31%)
"owl": 0.15 / 0.80 = 0.19 (19%)
Total = 1.00 ✅
```

### 6.3 Why Top-P is Adaptive (Advantage over Top-K)

| Scenario | Top-K (K=3) | Top-P (p=0.8) |
|---|---|---|
| Model is very confident (1 token = 90% prob) | Still considers 3 tokens | Only 1 token qualifies → Deterministic |
| Model is unsure (50 tokens each ~2% prob) | Only 3 tokens → too restrictive | 40 tokens qualify → More creative |

> **Top-P is self-adapting**: when the model is confident, it becomes strict; when the model is uncertain, it becomes liberal.

---

## 7. Top-K Sampling

### 7.1 What It Is
At each step, only the **K highest probability tokens** are considered. All others are set to 0.

### 7.2 Example
- K = 3, vocabulary = 50,000 tokens
- Only the top 3 most likely tokens are kept
- Probabilities are renormalized to sum to 1.0
- Token is sampled from this set of 3

### 7.3 Limitation
Top-K is a fixed number — it doesn't adapt to the model's confidence level. When the model is very sure, you might want K=1. When it's unsure, you want K=50. Top-P handles this automatically.

---

## 8. Greedy Search vs Beam Search

### 8.1 Greedy Search

**How it works:**
- At every step, pick the single token with the **highest probability**
- No memory of alternative paths
- Extremely fast, O(1) memory

**The Problem — Local vs Global Optimum:**

```
Prompt: "I love to ride"

Step 1:  Pick max prob → "bike" (0.5) [ignores "car" (0.4) and "horse" (0.3)]
Step 2:  From "bike" branch → pick max prob next word
Result:  "I love to ride bike night" — joint prob = 0.5 × 0.4 × 0.9 = 0.18
```

The problem: "car" (0.4) was ignored even though:
```
"I love to ride car fast" — joint prob = 0.4 × 0.9 = 0.36 > 0.18
```
Greedy search missed the better sentence! It got trapped in a **local optimum**.

### 8.2 Beam Search

**How it works:**
- Keep top K paths (beams) alive at every step
- At each step, expand all K paths, calculate cumulative scores
- Keep only the top K overall sequences
- At the end, return the highest-scoring sequence

**Example with Beam Width = 2:**

```
Step 1:  Keep top 2 → "bike" (0.6) AND "car" (0.3) [both alive]
Step 2:  Expand both:
           "bike" → "night" (0.4) → total: 0.6+0.4 = 1.0
           "car"  → "fast"  (0.9) → total: 0.3+0.9 = 1.2 ✅ WINNER
Result:  "I love to ride car fast" — higher joint probability
```

### 8.3 Joint Probability — Why It Matters

The probability of a **sequence** = product of probabilities of each word:
```
P("I love to ride bike night") = P(bike|context) × P(night|bike) × ...
                                = 0.4 × 0.5 × 0.9 = 0.18

P("I love to ride car fast")   = P(car|context) × P(fast|car) × ...
                                = 0.3 × 0.8 × 0.9 = 0.216
```
0.216 > 0.18 → "car fast" is the better sequence, but greedy never finds it.

### 8.4 Comparison Table

| Aspect | Greedy Search | Beam Search |
|---|---|---|
| **Paths tracked** | 1 | K (beam width) |
| **Memory usage** | O(1) | O(K × sequence_length) |
| **Speed** | Very fast | Slower by K× |
| **Output quality** | May miss better sequences | Usually more coherent |
| **Risk** | Gets trapped in local optima | Computationally expensive |
| **Best for** | Simple classification, quick responses | Translation, summarization |
| **HuggingFace param** | `do_sample=False` (default) | `num_beams=4` |

---

## 9. How to Choose the Right Model

### 9.1 The Core Philosophy

> *"Match the model's cognitive architecture to your problem's demands, not its position on a leaderboard."*

### 9.2 Two Major Model Families

#### Generation Models (Fluency + Creativity)
**Examples:** GPT-4o, Claude Sonnet, Gemini 1.5

| Aspect | Details |
|---|---|
| Optimized for | Natural language quality, fluency, creativity |
| **Best for** | Content creation, summarization, chatbots, marketing copy, Q&A, sentiment analysis, poems |
| Trade-off ✅ | Fast & cost-effective |
| Trade-off ⚠️ | Requires prompt scaffolding for complex logic |
| Key limitation | Needs "Chain of Thought" prompting to reason step-by-step |

#### Reasoning Models (Multi-Step Logic)
**Examples:** o1, o3-mini, DeepSeek R1

| Aspect | Details |
|---|---|
| Optimized for | Logical correctness, multi-step reasoning |
| **Best for** | Code generation, math problems, legal analysis, strategic planning, complex comparisons |
| Trade-off ✅ | Built-in reasoning chains (thinks before answering) |
| Trade-off ⚠️ | Slower & more expensive per API call |
| Key feature | Silently generates a "chain of thought" before outputting the answer |

### 9.3 The Evaluation Framework

When you need to pick a model objectively:

1. **Assemble a benchmark dataset:** 10,000+ real examples from your domain
2. **Label by difficulty:** 1 (easy) to 4 (hard), or S-E (simple to expert)
3. **Run all candidate models** on the same dataset
4. **Score each answer:** ✓ (correct) / ✗ (incorrect)
5. **Domain-specific testing:** e.g., for medical apps, test on medical Q&A specifically
6. **Pick based on YOUR data** — not on general leaderboard rankings

### 9.4 The Generation vs Reasoning Decision Tree

```
What does your task require?
         │
    ┌────┴────┐
    │         │
Fluency +   Multi-step
Creativity   Logic
    │         │
Generation  Reasoning
Models      Models
(GPT-4o,   (o1, o3-mini,
 Claude)    DeepSeek R1)
```

### 9.5 Practical Examples

| Task | Category | Recommended Model Type |
|---|---|---|
| Summarize a PDF | General | Generation (GPT-4o-mini) |
| Write a blog post | General | Generation |
| Customer service chatbot | General | Generation |
| Sentiment analysis | General | Generation |
| Solve math problem | Reasoning | o1/o3-mini |
| Write complex code | Reasoning | o1/o3-mini |
| Legal contract analysis | Reasoning | Reasoning model |
| Flight booking comparison | Reasoning | Reasoning model |
| Strategic planning | Reasoning | Reasoning model |

---

## 10. Prompt Engineering Techniques

### 10.1 Zero-Shot Prompting

**What it is:** Direct instruction without any examples. The model relies on its pre-trained knowledge.

**When to use:** Simple, well-defined tasks where the model already understands the format.

**Example:**
```python
zero_shot_prompt = """Classify the sentiment of this review as 
POSITIVE, NEGATIVE, or NEUTRAL.
Review: "The battery life is incredible but the camera quality disappointed me."
Sentiment:"""
# Output: MIXED (model interprets both aspects)
```

### 10.2 Few-Shot Prompting

**What it is:** Provide 2–5 input→output examples before the actual query. Also called **In-Context Learning (ICL)**.

**When to use:** Custom output formats, domain-specific tasks, edge cases, inconsistent zero-shot results.

**Key insight:** The model learns the pattern from examples without any weight updates — this is learning at inference time, not training time.

**Example:**
```python
few_shot_prompt = """Classify sentiment as POSITIVE, NEGATIVE, or NEUTRAL.
Review: "Absolutely love this product!" Sentiment: POSITIVE
Review: "Terrible experience. Broke after one day." Sentiment: NEGATIVE
Review: "It works as expected. Nothing special." Sentiment: NEUTRAL
Review: "Battery life is incredible but camera disappointed me." Sentiment:"""
# Output: NEUTRAL or MIXED (more consistent with examples)
```

**Why Few-Shot > Zero-Shot for custom tasks:**
- Zero-shot: Model guesses what "MIXED" means
- Few-shot: Model sees the exact 3-option format you want and mirrors it

### 10.3 Chain of Thought (CoT) Prompting

**What it is:** Adding "Think step by step" or showing reasoning examples forces the model to reason through problems before giving a final answer.

**When to use:** Math problems, multi-step logic, complex reasoning tasks.

**Why it works:** Instead of predicting just the final answer token, the model predicts many intermediate reasoning tokens, each serving as context for the next calculation.

**Example:**
```python
cot_prompt = """Solve this problem step by step.
Problem: A store sells apples for $2 each. If you buy 5 or more, 
you get 20% off. How much would 7 apples cost?
Think through this step by step:"""

# Output (with CoT):
# 1. Price per apple = $2
# 2. Total without discount: 7 × $2 = $14
# 3. Buying 7 ≥ 5, so 20% discount applies
# 4. Discount amount: $14 × 0.20 = $2.80
# 5. Final price: $14 - $2.80 = $11.20 ✅
```

### 10.4 Comparison of Prompting Techniques

| Technique | Examples Needed | Best For | Limitation |
|---|---|---|---|
| **Zero-Shot** | 0 | Simple, well-known tasks | May not follow your exact format |
| **Few-Shot** | 2-5 | Custom formats, domain-specific | Uses up context window tokens |
| **CoT** | 0 or few | Math, logic, reasoning | Slower (more tokens generated) |
| **CoT + Few-Shot** | 2-3 with reasoning shown | Hard reasoning tasks | Expensive (many tokens) |

### 10.5 The Prompt Engineering Workflow

**Recommended iteration path:**
```
GPT-4.1  ──►  Craft 2-page system prompt  ──►  Test on examples
                        │
              Once prompt is solid
                        ▼
                    GPT-5.2
               (better results with same prompt)
```

Start on a cheaper/older model to iterate on the prompt. Once the prompt is well-engineered, upgrade to the latest model for production.

---

## 11. API Message Structure

### 11.1 The Three Roles

Every LLM API call (OpenAI-compatible) uses a messages array with three role types:

| Role | Purpose | When to Use |
|---|---|---|
| **system** | Sets model's behaviour, persona, constraints | Always — define who the model is |
| **user** | Human input, questions, instructions | Every user turn |
| **assistant** | Model's response | Inject previous AI replies for conversation history |

### 11.2 Standard API Call Structure

```python
from openai import OpenAI
client = OpenAI()

def get_response(messages, model="gpt-4o-mini", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=500
    )
    return response.choices[0].message.content

# Single-turn example
messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is the capital of France?"},
]
response = get_response(messages)
```

### 11.3 Multi-Turn Conversation (Chat History)

```python
# Multi-turn: inject history as messages
messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},          # ← previous AI reply
    {"role": "user",      "content": "Multiply that by 5."},  # ← new question
]
# Model has context: knows "that" refers to 4 → outputs 20
```

### 11.4 Generation vs Reasoning Model API Difference

**For Generation Models (GPT-4o-mini):**  
Requires detailed, structured prompts (prompt scaffolding):
```python
structured_prompt = """You are a loan approval officer. Follow these steps:
STEP 1: Calculate debt-to-income ratio
STEP 2: Check if credit score >= 700
STEP 3: Verify employment duration >= 2 years
STEP 4: Make final decision
Applicant data: {data}
Decision:"""
```

**For Reasoning Models (o1-mini):**  
Can use minimal, simple prompts:
```python
minimal_prompt = """
Applicant data: {data}
Lending rules: {rules}
Decision: Approve or deny? Explain your reasoning.
"""
# The model's built-in chain of thought handles all the steps internally
```

**Trade-off:**
- Generation model: Cheap per call, but engineer must design the reasoning chain
- Reasoning model: Expensive per call, but engineer just defines success criteria

---

## 12. RAG — Retrieval-Augmented Generation

### 12.1 The Problem RAG Solves

LLMs have two fundamental limitations:
1. **Knowledge cutoff** — they don't know about recent events
2. **No access to private data** — your company's documents, databases, PDFs

RAG solves both without retraining the model.

### 12.2 How RAG Works — Complete Flow

```
User Query: "What was our Q3 revenue?"
         │
         ▼
[1. EMBED THE QUERY]
Query → Embedding Model → [0.12, -0.34, 0.56, ...] (vector)
         │
         ▼
[2. RETRIEVE (Similarity Search)]
Search vector database (FAISS, Pinecone, Chroma)
Find top-K documents most similar to query vector
         │
         ▼
[3. AUGMENT (Inject Context)]
Retrieved docs inserted into LLM prompt:
"Context: [doc1] [doc2] [doc3]
Question: What was our Q3 revenue?"
         │
         ▼
[4. GENERATE]
LLM reads the injected context and generates answer
grounded in your documents
         │
         ▼
Final Answer: "Based on the Q3 report, revenue was $4.2M..."
```

### 12.3 Why RAG vs Fine-tuning?

| Aspect | RAG | Fine-tuning |
|---|---|---|
| **Cost** | Low (API calls + vector DB) | High (GPU training) |
| **Update frequency** | Real-time (update docs in DB) | Expensive to re-train |
| **Knowledge scope** | Any documents you add | Fixed at training time |
| **Privacy** | Documents stay on your server | Must send to training pipeline |
| **Best for** | Dynamic, current, private data | Specific style/domain adaptation |

### 12.4 The Retrieval Step — Primary Purpose

> ⚠️ **Common misconception:** The retrieval step is NOT for:
> - Fine-tuning the model
> - Reducing inference latency
> - Compressing the model
>
> ✅ The retrieval step's **only purpose** is to **fetch relevant documents to add as context**.

---

## 13. Vector Databases & FAISS

### 13.1 Why Vector Databases?

Traditional databases search by exact match: `WHERE name = 'John'`

Vector databases search by **semantic similarity**: "Find content that *means* the same as this query"

**Enabled by:** Converting text/images into numerical vectors (embeddings) using models like OpenAI's `text-embedding-3-small`

### 13.2 How Embeddings Work

```
"affordable laptops" → embedding model → [0.12, -0.34, 0.56, ...] (1536 numbers)
"budget notebooks"   → embedding model → [0.11, -0.33, 0.55, ...] (very similar!)
```
Because these two phrases mean the same thing, their vectors are mathematically close to each other.

### 13.3 Distance Metrics

| Metric | Formula | Best For |
|---|---|---|
| **Cosine Similarity** | cos(θ) = (A·B) / (‖A‖‖B‖) | Text embeddings (normalized) |
| **Euclidean (L2)** | √Σ(a-b)² | Image embeddings |
| **Dot Product** | Σ(a·b) | When magnitude matters |

### 13.4 FAISS Implementation — Step by Step

**FAISS (Facebook AI Similarity Search)** — open-source library for fast vector similarity search.

**Step 1: Prepare your documents**
```python
documents = [
    "Python was created by Guido van Rossum in 1991.",
    "FAISS is a library for similarity search by Facebook AI.",
    "RAG combines retrieval with generation for grounded LLM responses.",
    "Vector databases store embeddings for nearest-neighbor search.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional vectors."
]
```

**Step 2: Create embeddings**
```python
import numpy as np

def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([e.embedding for e in response.data], dtype="float32")

doc_embeddings = get_embeddings(documents)
print(doc_embeddings.shape)  # (5, 1536) → 5 docs, each 1536-dimensional
```

**Step 3: Store in FAISS**
```python
import faiss

dimension = doc_embeddings.shape[1]  # 1536
index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance
index.add(doc_embeddings)
print(f"FAISS index: {index.ntotal} vectors")  # FAISS index: 5 vectors
```

**Step 4: Retrieve relevant documents**
```python
def retrieve(query, k=2):
    query_embedding = get_embeddings([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

retrieved = retrieve("Who created Python?", k=2)
# Returns: ["Python was created by Guido van Rossum...", ...]
```

**Step 5: Generate answer with RAG**
```python
def generate_answer(query, context):
    context_str = "\n".join(context)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based ONLY on the provided context."},
            {"role": "user",   "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content

answer = generate_answer("Who created Python?", retrieved)
# Output: "Python was created by Guido van Rossum."
```

### 13.5 Vector DB Tools Comparison

| Tool | Type | Best For |
|---|---|---|
| **FAISS** | Library (local) | Fast prototyping, research, no server needed |
| **Chroma** | Embedded DB | Dev/testing, simple setup |
| **Pinecone** | Managed cloud | Production, serverless, easy scaling |
| **Milvus** | Self-hosted | Large scale, open source |
| **Weaviate** | Self-hosted/cloud | Hybrid search (vector + keyword) |

---

## 14. AI Agents

### 14.1 What is an AI Agent?

An AI Agent is an **autonomous system** that uses an LLM as its "brain" to:
- Reason about tasks
- Make decisions
- Execute tools/actions
- Observe results
- Iterate until the goal is complete

**Key difference from a chatbot:** A chatbot just responds. An agent **acts** — it takes real-world actions like searching the web, running code, calling APIs.

### 14.2 Components of an AI Agent

| Component | Role | Example |
|---|---|---|
| **LLM (Reasoning Engine)** | The brain — understands, decides, plans | GPT-4o, Claude |
| **Tools (Actions)** | Functions the agent can call | web_search(), calculator(), run_code() |
| **Control Loop (Orchestration)** | Manages the reasoning→acting→observing cycle | LangGraph, custom code |
| ❌ **Training Dataset** | NOT a component of an agent | — |

### 14.3 The ReAct Pattern (Reason + Act)

**ReAct** is the most common agent architecture. It alternates between:

```
┌──────────────────────────────────────────────┐
│                 ReAct Loop                    │
│                                              │
│  Thought: "What do I need to do next?"       │
│      ↓                                       │
│  Action: Call a tool                         │
│      ↓                                       │
│  Observation: Read tool result               │
│      ↓                                       │
│  Thought: "What does this mean?"             │
│      ↓                                       │
│  (Repeat until goal achieved)                │
│      ↓                                       │
│  Final Answer: Output to user                │
└──────────────────────────────────────────────┘
```

**Concrete Example:**
```
User:     "What is Apple's current stock price?"

Thought:  "I need to find the current AAPL price."
Action:   web_search("AAPL stock price today")
Observation: "$189.45 as of 3:45 PM"

Thought:  "I have the current price. I can now answer."
Answer:   "Apple's current stock price is $189.45."
```

### 14.4 The Agent Control Loop in Detail

1. User sends query
2. LLM receives query + tool definitions
3. LLM decides: "I need to call tool X with parameters Y"
4. System executes the tool call (NOT the LLM — the orchestration layer does this)
5. Tool result (Observation) is appended to the conversation
6. LLM receives updated conversation (including tool result)
7. LLM decides: "I have enough info" → gives Final Answer
   OR "I need more info" → calls another tool (back to step 3)

> ⚠️ **Common exam trick:** After deciding to call a tool, the agent does NOT give the final answer immediately. It must wait for the Observation (tool result) first.

### 14.5 Why Agents Matter

Agents extend LLMs from **passive responders** to **active problem-solvers**:
- Break complex tasks into steps automatically
- Gather real-time information (web search)
- Perform computations (calculator)
- Execute code
- Self-correct based on intermediate results
- Chain multiple tools together

### 14.6 LangGraph Agent Example (Architecture)

```
User Query
    │
    ▼
LLM Reasoning (GPT-4o-mini)
    │
    ▼
Tool Call? ──YES──► Execute Tool (add/multiply/subtract)
    │                       │
   NO                       ▼
    │              Tool Result (Observation)
    │                       │
    └──────────────────────►│
                            ▼
                      Loop back to LLM
                      (max 5 iterations to prevent infinite loops)
                            │
                    Final Answer to User
```

---

## 15. Eval Pipelines for AI Apps

### 15.1 Why Evaluation Matters

The generation parameters (Temperature, Top-K, Top-P, Beam Search) **directly impact output quality**. An eval pipeline helps you scientifically determine which settings produce the best results for your specific application.

### 15.2 The Evaluation Decision Tree

```
Do you have reference examples (ground truth)?
         │
    ┌────┴────┐
   YES        NO
    │          │
    ▼          ▼
Reference-  Reference-Free
Based Eval  Evaluation
    │
Is there ONLY ONE correct answer?
    │
┌───┴───┐
YES     NO
 │       │
 ▼       ▼
Is it  Many possible correct answers
predictive?  → Use Generative Metrics:
    │          1. BLEU/ROUGE (word overlap)
  ┌─┴─┐        2. BERTScore (semantic)
 YES   NO       3. LLM as Judge
  │     │
  ▼     ▼
Standard  Deterministic
ML Metrics  Validation
(Accuracy,  (Exact match,
NDCG, RMSE)  JSON match,
             Unit tests)
```

### 15.3 Evaluation Metrics Explained

#### A. BLEU (Bilingual Evaluation Understudy)
- **Origin:** Designed for Machine Translation
- **What it measures:** Precision — how many words in AI output match the reference
- **Formula:** Checks n-gram overlap between generated text and reference
- **Limitation:** Only checks exact word overlap, not semantic meaning

#### B. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **Origin:** Designed for Text Summarization
- **What it measures:** Recall — did the AI capture all important words from the reference?
- **Key difference from BLEU:** BLEU = precision (AI didn't say wrong things), ROUGE = recall (AI said all the right things)
- **Limitation:** Same as BLEU — only word overlap, misses semantic equivalence

#### C. BERTScore
- **What it measures:** Semantic similarity using BERT embeddings
- **How:** Compares meaning vectors, not exact words
- **Example:** "The feline rested" vs "The cat slept" → BERTScore: HIGH (same meaning), BLEU: 0 (no word overlap)

#### D. LLM as a Judge (Modern Approach)
- **How it works:** Use a powerful LLM (GPT-4) to evaluate another LLM's output
- **Prompt pattern:** "Here is the question, here is the reference answer, here is the AI's answer. Score the AI's answer 1-5 for accuracy and helpfulness."
- **Why it's better:** Evaluates semantic meaning, not just word overlap
- **Limitation:** Expensive (another API call per evaluation), may have bias

### 15.4 Which Metric to Use — Quick Guide

| Task Type | Has Reference? | One Answer? | Use This Metric |
|---|---|---|---|
| Classification | Yes | Yes | Accuracy, Precision, Recall, F1 |
| Regression | Yes | Yes | MAE, RMSE, R² |
| Ranking/Recommendation | Yes | Yes | NDCG, HitRate |
| Code generation | Yes | Yes | Unit Tests (pass/fail) |
| JSON extraction | Yes | Yes | Exact match, JSON schema validation |
| Machine Translation | Yes | No (many valid translations) | BLEU |
| Summarization | Yes | No (many valid summaries) | ROUGE, BERTScore |
| Chatbot response | Sometimes | No | LLM-as-Judge, Human Eval |
| Creative writing | No | No | LLM-as-Judge, Human Eval |

### 15.5 The LLM Training Lifecycle (Revisited in Eval Context)

```
Stage 1: Raw LM (Pre-training)
  → Metric: Perplexity (how well does it predict next word?)

Stage 2: Instruct Fine-tuning (SFT)
  → Metric: Accuracy on instruction-following benchmarks

Stage 3: RLHF / Alignment
  → Metric: Human preference scores, helpfulness ratings, safety checks

Stage 4: Production Application
  → Metric: BLEU, ROUGE, BERTScore, LLM-as-Judge, Unit Tests (task-dependent)
```

---

## 16. LLM Frameworks & Tools Ecosystem

### 16.1 Complete Ecosystem Overview

| Category | Tools | Description |
|---|---|---|
| **Vector Databases** | Chroma, Pinecone, Milvus, FAISS, Weaviate | Store and retrieve embeddings for semantic search and RAG |
| **Orchestration** | LangChain, LlamaIndex, MCP | Chain LLM calls with tools, memory, retrieval |
| **Agents** | LangGraph, smolagents, CrewAI, AutoGen | Build stateful agents with reasoning loops |
| **Fine-Tuning** | TRL, Unsloth, Axolotl | Train and align LLMs with SFT, DPO, RLHF, GRPO |
| **Quantization** | llama.cpp (GGUF), GPTQ, AWQ, SmoothQuant | Compress models to reduce memory |
| **Evaluation** | Ragas, DeepEval, LM Evaluation Harness | Benchmark model performance |
| **Security** | garak, Langfuse | Detect prompt injection, monitor production |
| **Experiment Tracking** | Weights & Biases, Comet, MLflow | Log metrics, hyperparameters across runs |
| **Structured Output** | Outlines, LMQL | Constrain outputs to JSON/regex/grammars |
| **Programmatic LLMs** | DSPy, TextGrad | Optimize prompts automatically |

### 16.2 LangChain Basics — Key Code

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")

# Create a reusable prompt template with variables
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains concepts simply."),
    ("user", "Explain {topic} in one sentence.")
])

# Chain: prompt → LLM → parse output
chain = prompt | llm | StrOutputParser()

response = chain.invoke({"topic": "vector databases"})
# Output: "Vector databases store data as numerical embeddings to enable 
#          fast similarity search based on meaning rather than exact matches."
```

### 16.3 Choosing the Right Stack

| What you're building | Recommended Tools |
|---|---|
| RAG / Q&A System | LlamaIndex, Chroma, OpenAI Embeddings |
| Chatbot with Tools | LangChain, LangGraph, GPT-4o |
| Fine-tuned Model | TRL/Unsloth, W&B, Hugging Face Hub |
| Production API | vLLM, AWQ quantization, OpenAI-compatible server |
| Local/Private | Ollama, Llama 3.2, Chroma |
| Research | DSPy, LangFuse, Custom eval |

---

## 17. LLM Infrastructure & Deployment

### 17.1 Deployment Options

| Type | Tools | Description |
|---|---|---|
| **Local** | LM Studio, Ollama, kobold.cpp | Run models on personal hardware |
| **Demo/Prototype** | Gradio, Streamlit, HF Spaces | Quick interactive ML demos |
| **Production Server** | vLLM, TGI, Ray Serve | High-throughput inference |
| **Edge** | MLC LLM, mnn-llm | Mobile, browser, resource-constrained |
| **Cloud Orchestration** | SkyPilot | GPU workloads across cloud providers |
| **AI-Powered IDE** | Cursor | Uses LLM APIs for code completion |

### 17.2 vLLM — Production Inference Engine

**vLLM (Virtual LLM)** is the industry-standard high-throughput inference server for LLMs.

| Feature | Description | Benefit |
|---|---|---|
| **PagedAttention** | Manages KV cache like OS virtual memory | 2-4× higher throughput vs HuggingFace |
| **Continuous Batching** | Dynamically adds/removes requests mid-generation | No idle GPU time |
| **Quantization Support** | AWQ, GPTQ, FP8, INT8, GGUF | 2-4× memory reduction |
| **Tensor Parallelism** | Split model across multiple GPUs | Run 70B+ models |
| **Speculative Decoding** | Small draft model predicts tokens | Up to 2× speedup |
| **Prefix Caching** | Cache common prompt prefixes | Faster for repeated system prompts |

### 17.3 RunPod Deployment Commands

```bash
# Download model from Hugging Face
huggingface-cli download openai-community/gpt2 --local-dir ./gpt2

# Start vLLM server (OpenAI-compatible endpoint)
vllm serve ./gpt2 --host 0.0.0.0 --port 8000

# Test the endpoint
curl -v http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{
    "model": "./gpt2",
    "prompt": "Say hello from inside the pod.",
    "max_tokens": 20
  }'
```

---

## 18. Code Examples

### 18.1 Softmax with Temperature (NumPy)

```python
import numpy as np

def softmax(z, temperature=1.0):
    z_temp = z / temperature           # Step 1: Apply temperature
    exp_z = np.exp(z_temp - np.max(z_temp))  # Step 2: Stabilize + exponentiate
    return exp_z / exp_z.sum()         # Step 3: Normalize to probabilities

logits = np.array([2.0, 1.0, 0.1])

print("T=0.5:", softmax(logits, temperature=0.5))  # [0.867, 0.117, 0.016]
print("T=1.0:", softmax(logits, temperature=1.0))  # [0.659, 0.242, 0.098]
print("T=2.0:", softmax(logits, temperature=2.0))  # [0.506, 0.307, 0.186]
```

### 18.2 Hugging Face: Base vs Instruct vs Chat Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Base Model (GPT-2)
base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Chat Model
chat_pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_base(prompt, max_tokens=50):
    inputs = base_tokenizer.encode(prompt, return_tensors="pt")
    outputs = base_model.generate(inputs, max_new_tokens=max_tokens,
                                   do_sample=True, temperature=0.7,
                                   pad_token_id=base_tokenizer.eos_token_id)
    return base_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

def generate_instruct(prompt, max_tokens=50):
    formatted = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    output = chat_pipe(formatted, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].split("<|assistant|>")[-1].strip()

def generate_chat(messages, max_tokens=50):
    output = chat_pipe(messages, max_new_tokens=max_tokens)
    return output[0]['generated_text'][-1]['content'].strip()

# Test: "What is the capital of Japan?"
# Base Model:    "Japan is an archipelago in the Pacific..." (fails)
# Instruct:      "The capital of Japan is Tokyo." ✅
# Chat:          "Tokyo is the capital of Japan." ✅
```

---

## 19. Acronym Glossary

| Acronym | Full Form | Simple Explanation |
|---|---|---|
| **LLM** | Large Language Model | A very big AI trained on enormous amounts of text |
| **API** | Application Programming Interface | A way for programs to talk to each other |
| **RAG** | Retrieval-Augmented Generation | Giving LLM access to external documents at query time |
| **RLHF** | Reinforcement Learning from Human Feedback | Training AI using human preferences (thumbs up/down) |
| **SFT** | Supervised Fine-Tuning | Further training a model on specific instruction-answer pairs |
| **CoT** | Chain of Thought | Making the AI reason step-by-step before answering |
| **ICL** | In-Context Learning | Teaching the model via examples in the prompt (no training) |
| **BLEU** | Bilingual Evaluation Understudy | Metric measuring word-level precision (for translation) |
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation | Metric measuring recall (for summarization) |
| **FAISS** | Facebook AI Similarity Search | Library for fast vector similarity search |
| **KV Cache** | Key-Value Cache | Memory optimization in transformer attention computation |
| **NDCG** | Normalized Discounted Cumulative Gain | Ranking quality metric |
| **MAE** | Mean Absolute Error | Average of absolute prediction errors |
| **RMSE** | Root Mean Square Error | Standard regression error metric |
| **GPU** | Graphics Processing Unit | Hardware chip used for AI training/inference |
| **HF** | Hugging Face | Popular open-source AI model hub |
| **vLLM** | Virtual LLM (inference server) | Fast production serving framework |
| **TGI** | Text Generation Inference | HuggingFace's production inference server |
| **SFT** | Supervised Fine-Tuning | Training on labeled examples |
| **DPO** | Direct Preference Optimization | Alignment technique, alternative to RLHF |
| **MCP** | Model Context Protocol | Standard for connecting AI models to tools/data |
| **LCEL** | LangChain Expression Language | Syntax for chaining LangChain components with `\|` |

---

## 20. Comparison Tables

### 20.1 Generation Models vs Reasoning Models

| Feature | Generation Models | Reasoning Models |
|---|---|---|
| **Examples** | GPT-4o, Claude Sonnet, Gemini 1.5 | o1, o3-mini, DeepSeek R1 |
| **Optimized for** | Fluency, creativity, NLQ | Logical correctness, multi-step |
| **Response speed** | Fast | Slow (internal thinking) |
| **Cost per call** | Low-Medium | High |
| **Prompt complexity** | Needs detailed prompt scaffolding | Works with simple prompts |
| **Use cases** | Chatbots, writing, summarization | Math, code, legal, strategy |
| **Chain of thought** | Must be prompted explicitly | Built-in, automatic |

### 20.2 Decoding Strategies

| Strategy | How It Works | Pros | Cons |
|---|---|---|---|
| **Greedy** | Always pick top-1 token | Fast, deterministic | Local optima traps |
| **Beam Search** | Keep top-K sequences | Better quality sequences | Slow, memory intensive |
| **Top-K Sampling** | Sample from top-K tokens | Controllable diversity | Fixed K not adaptive |
| **Top-P (Nucleus)** | Sample from top cumulative-P tokens | Adaptive, context-aware | Can be unpredictable |
| **Temperature** | Scale logits before softmax | Simple creativity knob | Doesn't limit vocabulary |

### 20.3 Evaluation Metrics

| Metric | Measures | Task | Limitation |
|---|---|---|---|
| **Accuracy** | % correct | Classification | Only yes/no |
| **BLEU** | Word-level precision | Translation | Misses paraphrases |
| **ROUGE** | Word-level recall | Summarization | Misses paraphrases |
| **BERTScore** | Semantic similarity | Generation | Needs BERT model |
| **LLM-as-Judge** | Holistic quality | Any | Expensive, possible bias |
| **Unit Tests** | Functional correctness | Code | Must write tests |
| **Exact Match** | String equality | Extraction | Brittle (case-sensitive) |

---

## 21. Interview Important Points

### Conceptual Questions

1. **What is the key difference between a Base LLM and an Instruct LLM?**
   - Base: Predicts next word, cannot follow instructions
   - Instruct: Fine-tuned to follow specific instructions in single turns

2. **What happens when Temperature → 0?**
   - The probability distribution becomes extremely sharp (peaked)
   - The model almost always picks the single most likely token
   - Output becomes deterministic and reproducible

3. **Why do we subtract the max before applying softmax?**
   - Numerical stability: prevents e^(very large number) = infinity (overflow)
   - Mathematically equivalent: subtracting a constant from all logits doesn't change the final probabilities

4. **Why might Greedy Search give worse output than Beam Search?**
   - Greedy gets trapped in local optima — it commits to the highest probability first word even if a slightly lower probability first word leads to a much better overall sentence
   - Joint probability of the full sequence can be higher via a non-greedy start

5. **What is the primary purpose of the Retrieval step in RAG?**
   - To fetch relevant documents to use as context for the LLM — NOT for fine-tuning, compressing, or speeding up the model

6. **Which component is NOT part of an AI Agent?**
   - Training Dataset (agents use pre-trained models, they don't train during execution)

7. **What does ReAct stand for?**
   - Reason + Act (the agent reasons about what to do, then acts by calling a tool)

8. **Why does Few-Shot prompting work without any model training?**
   - In-Context Learning (ICL): The model reads the examples as part of its context window and extracts the pattern, treating examples as implicit instructions

9. **When should you use Top-P over Top-K?**
   - Top-P is adaptive: when the model is confident, it restricts more; when uncertain, it allows more options
   - Top-K is fixed and doesn't adapt to context confidence

10. **What is Entropy in the context of temperature?**
    - Entropy measures the "randomness" of the probability distribution
    - Low temperature = low entropy = more predictable output
    - High temperature = high entropy = more random output

---

## 22. Quick Revision Notes

### The Big 3 Model Types
```
Base → only next-word prediction (GPT-2 style)
Instruct → follows single-turn instructions
Chat → multi-turn conversation with context
```

### Temperature Rules
```
T → 0: Deterministic (always top token)    ← Math, code, facts
T = 1: Default balanced                     ← General use
T > 1: Creative/Random                      ← Writing, brainstorming
```

### Softmax Formula
```
P(token_i) = e^(logit_i / T) / Σ e^(logit_j / T)
Stability trick: subtract max before e^x
```

### RAG Flow
```
Query → Embed → Similarity Search → Retrieve docs → Inject into prompt → Generate
```

### ReAct Agent Loop
```
Thought → Action (tool call) → Observation (result) → Thought → ... → Final Answer
```

### Eval Metric Selection
```
One correct answer + predictive task → Accuracy/RMSE/NDCG
One correct answer + extraction → Exact match/Unit tests
Many correct answers → BLEU/ROUGE/BERTScore/LLM-as-Judge
No reference at all → LLM-as-Judge/Human eval
```

### Model Selection
```
Fluency/creativity tasks → Generation models (GPT-4o, Claude Sonnet)
Logic/reasoning tasks → Reasoning models (o1, o3-mini, DeepSeek R1)
Always benchmark on YOUR specific data!
```

---

## 23. Top 20 Takeaways

1. **AI Engineering ≠ ML Engineering** — AI Engineers integrate pre-trained models, not train from scratch.
2. **Base models cannot follow instructions** — they only predict the next word. Always use Instruct or Chat models for applications.
3. **Temperature is a creativity dial** — low = deterministic (math/code), high = creative (writing).
4. **Softmax numerical stability** — always subtract the max before computing e^x to prevent overflow.
5. **Top-P is superior to Top-K** — it's adaptive to the model's confidence level.
6. **Greedy search misses global optima** — beam search tracks multiple paths to find better sequences.
7. **Joint probability matters** — the probability of a sequence = product of individual token probabilities.
8. **The OpenAI API format is the industry standard** — learning it transfers to most providers.
9. **Chat templates matter** — each model family (Llama, Mistral, etc.) has specific tags. Use wrong tags → broken output.
10. **RAG solves the knowledge cutoff problem** — plug in your docs without retraining.
11. **FAISS is for similarity search** — NOT for fine-tuning, prompting, or compressing models.
12. **Retrieval step = fetch context only** — the augmentation step is what injects it into the prompt.
13. **AI Agents = LLM + Tools + Control Loop** — Training Dataset is NOT a component.
14. **ReAct = Reason + Act** — agents must wait for tool observations before giving final answers.
15. **After tool call → Observation → then final answer** — never directly after tool call.
16. **BLEU = precision, ROUGE = recall** — BLEU checks AI didn't say wrong things; ROUGE checks it didn't miss things.
17. **LLM-as-Judge beats BLEU/ROUGE** — can evaluate semantic meaning, not just word overlap.
18. **"Match the model's cognitive architecture to your task"** — not its leaderboard position.
19. **Generation models need prompt scaffolding for logic** — reasoning models don't.
20. **vLLM's PagedAttention** — provides 2-4× throughput improvement for production serving.

---

## 24. Interview Q&A with Answers

**Q1. What is AI Engineering and how is it different from traditional ML Engineering?**

> **A:** AI Engineering focuses on building applications using pre-trained foundation models through integration, orchestration, prompt design, RAG, and agent systems. Traditional ML Engineering focuses on training models from scratch using statistical algorithms and large datasets. AI Engineers don't typically train models — they compose, configure, and orchestrate existing ones.

**Q2. Explain the difference between Base, Instruct, and Chat models.**

> **A:** Base models (like GPT-2) are trained on raw text with next-token prediction — they just autocomplete text and cannot follow instructions. Instruct models are fine-tuned on instruction→response pairs, enabling single-turn instruction following. Chat models are further fine-tuned on multi-turn conversations, allowing them to maintain context across a dialogue. The progression: Base → (SFT) → Instruct → (RLHF + conversation data) → Chat.

**Q3. What is Temperature in LLM generation and how does it work mathematically?**

> **A:** Temperature is a parameter that controls the randomness of LLM output by scaling the raw logit scores before applying Softmax. The formula is: P(token) = e^(logit/T) / Σe^(logit_j/T). At T<1, scores are magnified, making the distribution sharper (more confident). At T>1, scores are compressed, making the distribution flatter (more random). At T=0, the model always picks the top token (fully deterministic).

**Q4. Why do we subtract the maximum value before applying e^x in Softmax?**

> **A:** For numerical stability. Without this trick, if logits are large (e.g., 1000), computing e^1000 causes numerical overflow — the number exceeds the computer's float representation (infinity). By subtracting the max first, we ensure the largest value becomes 0, and all others become negative, keeping all exponentiated values ≤ 1. Mathematically, this subtraction cancels out in the final division, so the probabilities are identical.

**Q5. What is the difference between Greedy Search and Beam Search? When would you use each?**

> **A:** Greedy search picks the single highest-probability token at every step. It's fast but can get trapped in local optima — committing to a high-probability start even if a lower-probability start leads to a better overall sentence. Beam search maintains K candidate sequences (beam width K), expanding all of them and keeping the top K at each step. It finds better global sequences at the cost of K× more compute and memory. Use Greedy for simple, fast tasks (classification, quick Q&A). Use Beam Search for quality-critical generation (translation, summarization).

**Q6. Explain RAG and when you would use it over fine-tuning.**

> **A:** RAG (Retrieval-Augmented Generation) extends an LLM's knowledge at query time by fetching relevant documents from an external store and injecting them into the prompt. The flow is: embed query → vector similarity search → retrieve top-K docs → inject into prompt → generate. Use RAG over fine-tuning when: data changes frequently (fine-tuning is expensive to repeat), you need private/proprietary data, you want to cite sources, or the knowledge base is large. Fine-tune instead when you need to change the model's style, format, or domain-specific language patterns.

**Q7. What is the ReAct pattern in AI Agents?**

> **A:** ReAct stands for Reason + Act. It's the most common agent architecture where the LLM alternates between: (1) Reasoning — thinking about what needs to be done, (2) Acting — calling an external tool, and (3) Observing — reading the tool result. The loop repeats until the goal is achieved, then outputs a final answer. Key point: the agent does NOT give the final answer immediately after deciding to call a tool — it must receive and process the Observation first.

**Q8. When would you use LLM-as-a-Judge vs BLEU/ROUGE for evaluation?**

> **A:** Use BLEU for machine translation and ROUGE for summarization where word-level overlap is a meaningful proxy for quality. However, both fail when there are many valid paraphrases — "The cat slept" and "The feline rested" score 0 on BLEU despite being identical in meaning. Use LLM-as-a-Judge when: the task has many valid outputs (chatbots, creative writing, complex Q&A), when semantic equivalence matters more than exact wording, or when you need holistic evaluation (helpfulness, safety, tone). LLM-as-Judge is more expensive but far more accurate for generative tasks.

**Q9. What is Top-P Nucleus Sampling and why is it better than Top-K?**

> **A:** Top-P selects the smallest set of tokens whose cumulative probability reaches P (e.g., P=0.9), then renormalizes and samples from that set. Unlike Top-K which always considers exactly K tokens regardless of context, Top-P is adaptive: when the model is very confident (one token has 90% probability), the nucleus may contain just 1 token; when the model is uncertain (many tokens each ~2%), the nucleus may contain 50 tokens. This adaptive behaviour makes Top-P more suitable for varied generation tasks.

**Q10. What is the role of FAISS and how does it work?**

> **A:** FAISS (Facebook AI Similarity Search) is a library for efficient similarity search over dense vectors. It's used in the retrieval step of RAG pipelines. It works by: (1) indexing document embeddings (numerical vectors) in an efficient data structure, (2) when given a query embedding, finding the top-K vectors that are closest to it using distance metrics (L2, Cosine, Dot Product), and (3) returning the indices of the closest documents. FAISS is NOT for fine-tuning, prompting, model compression, or quantization — only for fast nearest-neighbor search.

---

## 25. Beginner-Friendly Explanation

### The Full Picture in Simple Words

Imagine you have a very intelligent friend (the LLM). This friend has read virtually the entire internet, so they know a LOT. But they have some limitations:

**1. Different "Modes" of this friend:**
- **Base mode:** Just autocompletes what you say — like your phone's autocomplete feature. You say "I love to ride" and they say "bike or bicycle." They don't know you're asking a question.
- **Instruct mode:** You can give them instructions. "Summarize this article." They'll do it.
- **Chat mode:** They remember the whole conversation. "What did I just ask?" — they know.

**2. The Creativity Knob (Temperature):**
Think of Temperature like a "creativity dial." Turn it to 0 and your friend always says the most obvious, safe answer. Turn it to 2 and they get wild and creative (maybe too creative). For math homework → dial at 0. For writing a poem → dial at 1.5.

**3. How they pick their next word:**
Every time they want to say the next word, they secretly rank all 50,000 words by how likely each is. Then:
- **Greedy:** They always pick #1 on the list
- **Top-K:** They randomly pick from the top 50
- **Top-P:** They randomly pick from however many words add up to 80% probability
- **Beam Search:** They draft 4 different sentences simultaneously and pick the best one at the end

**4. RAG — Giving your friend a cheat sheet:**
Your friend's knowledge stops at a certain date. But if you hand them a document before asking, they can use it. RAG does this automatically — it searches your files, finds the relevant pages, and hands them to your friend just before they answer.

**5. Agents — Giving your friend tools:**
Instead of just talking, now your friend can *do things*: search the web, run calculations, send emails. They think ("I need the stock price"), use a tool (search), see the result, think again ("now I have enough info"), and give you the final answer.

**6. Evaluation — Marking your friend's homework:**
How do you know if their answer was good?
- If there's one right answer → check if they got it right (exact match)
- If there are many valid answers → check if the meaning matches (LLM-as-Judge)
- For translations → BLEU score (did they use the right words?)
- For summaries → ROUGE score (did they miss any important parts?)

---

## 26. Advanced Concepts

### 26.1 RLHF and Alignment
- **RLHF** (Reinforcement Learning from Human Feedback) is the process used to turn an instruct model into a safe, helpful chat model
- Humans rank multiple model responses → a "reward model" learns human preferences → the LLM is trained to maximize reward
- Modern alternative: **DPO** (Direct Preference Optimization) — more stable training without a separate reward model

### 26.2 Multi-Model Agentic Workflows
```
User Input
    │
    ▼
Reasoning Model (o1/o3) ──► Decomposes task, creates plan
    │
    ▼
Generation Model (GPT-4o) ──► Executes each step fluently
    │
    ▼
Final polished output
```
Using reasoning models to plan and generation models to write → better quality than either alone

### 26.3 PagedAttention in vLLM
- The KV (Key-Value) cache from transformer attention takes massive GPU memory
- vLLM treats GPU memory like an OS treats RAM: allocates memory in pages
- Eliminates memory fragmentation → 2-4× more requests served per GPU
- Similar to how your computer runs many apps simultaneously with virtual memory

### 26.4 Quantization
- Reducing model weight precision (e.g., float32 → int8) to save memory and speed up inference
- **AWQ, GPTQ:** 4-bit quantization with minimal quality loss
- **GGUF (llama.cpp):** Optimized for CPU inference on consumer hardware
- Trade-off: smaller, faster model vs. slight quality degradation

### 26.5 Speculative Decoding
- Use a tiny "draft model" to predict the next N tokens very quickly
- Then verify with the big model in one pass (parallel computation)
- If the big model agrees → accept all N tokens (big speedup)
- If it disagrees → fall back at the first mismatch
- Result: Up to 2× speedup with identical output quality

### 26.6 BERTScore — Advanced Evaluation
- Uses BERT embeddings to compare generated and reference text
- Each word in generated text is matched to the most semantically similar word in the reference
- Computes Precision (P), Recall (R), and F1 at the embedding level
- Far more robust than BLEU/ROUGE for paraphrase-heavy tasks

### 26.7 DSPy — Programmatic Prompt Optimization
- Instead of manually writing prompts, DSPy optimizes prompts automatically
- You define what you want (input/output format + metric to optimize)
- DSPy runs experiments, evaluates outputs, and automatically improves the prompt
- Think: gradient descent for prompts

---

## 27. One-Page Cheat Sheet

```
╔══════════════════════════════════════════════════════════════════════╗
║              AI ENGINEERING — COMPLETE CHEAT SHEET                  ║
╠══════════════════════════════════════════════════════════════════════╣
║ MODEL TYPES                                                          ║
║   Base        → Next-word prediction only (GPT-2). No instructions  ║
║   Instruct    → Single-turn instruction following (SFT trained)      ║
║   Chat        → Multi-turn, context-aware (SFT + RLHF)              ║
╠══════════════════════════════════════════════════════════════════════╣
║ TEMPERATURE                                                          ║
║   T=0   → Deterministic (always top token)    ← Math/Code           ║
║   T=1   → Default balanced                    ← General             ║
║   T>1   → Creative/Random                     ← Writing             ║
║   Formula: P(i) = e^(logit_i/T) / Σe^(logit_j/T)                   ║
║   Stability trick: Subtract max before e^x                           ║
╠══════════════════════════════════════════════════════════════════════╣
║ DECODING                                                             ║
║   Greedy   → Always pick top-1. Fast, local optima risk             ║
║   Beam     → Track K paths. Slower, better sequences                ║
║   Top-K    → Sample from top K tokens                               ║
║   Top-P    → Sample from tokens summing to P%. Adaptive!            ║
╠══════════════════════════════════════════════════════════════════════╣
║ MODEL SELECTION                                                      ║
║   Generation (GPT-4o, Claude Sonnet)  → Fluency, creativity         ║
║   Reasoning (o1, o3-mini, DeepSeek)   → Logic, math, code           ║
║   Rule: Match model to task, not leaderboard                         ║
╠══════════════════════════════════════════════════════════════════════╣
║ PROMPTING TECHNIQUES                                                 ║
║   Zero-Shot  → Just ask directly                                     ║
║   Few-Shot   → Provide 2-5 examples first (ICL)                     ║
║   CoT        → "Think step by step" → better reasoning              ║
╠══════════════════════════════════════════════════════════════════════╣
║ RAG FLOW                                                             ║
║   Query → Embed → Similarity Search (FAISS) → Retrieve Docs         ║
║   → Inject into Prompt → Generate Grounded Answer                   ║
║   FAISS = similarity search library (NOT for fine-tuning!)          ║
╠══════════════════════════════════════════════════════════════════════╣
║ AI AGENTS                                                            ║
║   Components: LLM + Tools + Control Loop (NOT Training Data)        ║
║   ReAct = Reason + Act                                               ║
║   Loop: Thought → Action → Observation → Thought → Final Answer     ║
║   Key: Agent WAITS for Observation before giving Final Answer        ║
╠══════════════════════════════════════════════════════════════════════╣
║ EVALUATION METRICS                                                   ║
║   One right answer + predictive  → Accuracy, RMSE, NDCG             ║
║   One right answer + extraction  → Exact Match, Unit Tests          ║
║   Many valid answers             → BLEU, ROUGE, BERTScore            ║
║   No reference                   → LLM-as-Judge, Human Eval         ║
║   BLEU=precision (translation), ROUGE=recall (summarization)        ║
╠══════════════════════════════════════════════════════════════════════╣
║ API MESSAGE STRUCTURE                                                ║
║   system    → Sets model behaviour/persona                          ║
║   user      → Human input                                           ║
║   assistant → Model response (for history)                          ║
╠══════════════════════════════════════════════════════════════════════╣
║ KEY TOOLS                                                            ║
║   LangChain/LangGraph → Orchestration & Agents                      ║
║   FAISS/Pinecone/Chroma → Vector databases                          ║
║   vLLM/TGI → Production inference serving                           ║
║   HuggingFace → Open-source model hub                               ║
║   TRL/Unsloth → Fine-tuning                                         ║
╠══════════════════════════════════════════════════════════════════════╣
║ TOP INTERVIEW TRAPS                                                  ║
║   ❌ Training Dataset IS an agent component → ✅ It is NOT          ║
║   ❌ Retrieval step fine-tunes the model → ✅ Only fetches docs      ║
║   ❌ FAISS does fine-tuning → ✅ FAISS does similarity search only   ║
║   ❌ Greedy = best quality → ✅ Beam Search finds better sequences   ║
║   ❌ High T always bad → ✅ High T is good for creative tasks        ║
║   ❌ ReAct = Retrieve+Act → ✅ ReAct = Reason + Act                 ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

*Notes compiled from: Scaler Academy AI Engineering Course | Lecture: Getting Started with Text Generation LLM APIs | 19–28 February 2026*
