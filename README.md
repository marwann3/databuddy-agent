# DataBuddy: an Agentic Data Analyst (LlamaIndex)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marwann3/llamaindex-data-analyst/blob/main/data_analyst_agent.ipynb)

An agent that answers questions about a CSV file (totals, rankings, filters, charts) by **calling tools instead of guessing numbers**. Built with **LlamaIndex** `AgentWorkflow` and an open-source LLM (Llama 3.3 70B) served through the **Groq** free tier.

## 1. Use case

**Problem:** business users have sales data in a CSV but cannot write pandas or SQL. They want to ask in plain language: *"Which region earns the most?"*, *"Plot revenue by category"*.

**Why an agent, not a plain LLM or a fixed script?**
- A plain LLM cannot see the data, so it would hallucinate numbers.
- A fixed script only answers the questions it was coded for.
- An agent decides **which tool** to call and **in what order**, reads the result, and retries when a call fails. Multi-step questions ("compare Cairo and Delta, then plot it") work naturally.

## 2. LLM and provider

| Choice | Value |
|---|---|
| Model | `llama-3.3-70b-versatile` (Meta Llama 3.3 70B) |
| Provider | Groq (free tier, OpenAI-compatible API) |
| Agent loop | Native function calling |
| Temperature | 0.1 |

**Why:** the model is open-source and large enough to pick tools and arguments reliably. Groq has a free tier that needs no credit card, and its API is OpenAI-compatible, so LlamaIndex connects with `OpenAILike`. Because the model supports native function calling, LlamaIndex uses a `FunctionAgent` instead of a text-based ReAct loop. A low temperature keeps tool calls accurate and repeatable. If the model name is no longer available, set `GROQ_MODEL` to another model from Groq's model list.

> The first version used Hugging Face Inference Providers, but those now require pre-paid credits (HTTP 402), so the provider was switched to Groq.

## 3. Tools (designed for this use case)

| Tool | Purpose | Key arguments |
|---|---|---|
| `describe_dataset` | Shape, columns, dtypes, first rows. Called first to learn exact column names | none |
| `column_summary` | Stats for numeric columns, top values for text columns, missing values | `column` |
| `aggregate` | Group by a column and sum/mean/count/min/max/median another | `group_by`, `value_column`, `agg`, `top_n` |
| `filter_rows` | Look at specific records (`==`, `!=`, `>`, `<`, `>=`, `<=`, `contains`) | `column`, `operator`, `value`, `limit` |
| `plot_chart` | Save a bar/line/pie chart as PNG | `kind`, `x`, `y`, `agg` |

**Design decisions**
- **Few, focused tools**, so the model picks the right one easily.
- **Docstrings are written for the model.** LlamaIndex uses the function name, type hints and docstring as the tool schema.
- **Tools never crash.** On bad input they return an error that lists the valid options, so the agent can correct itself.
- **No arbitrary code execution.** The agent cannot run free-form Python, which keeps it safe.

## 4. System prompt

The full prompt is in the notebook. Main ideas:
- Role and goal: data analyst for one CSV.
- Must call `describe_dataset` first, so it never guesses column names.
- **Never invent numbers**: every number must come from a tool result.
- Retry with corrected arguments when a tool returns an error.
- Say clearly when the dataset cannot answer the question.
- Short answers, direct answer first, reply in the user's language.

## 5. Implementation with LlamaIndex

```python
tools = [FunctionTool.from_defaults(fn) for fn in ALL_TOOLS]
agent = AgentWorkflow.from_tools_or_functions(tools, llm=llm, system_prompt=SYSTEM_PROMPT)
ctx = Context(agent)                      # conversation memory
await agent.run("Which region sells most?", ctx=ctx)
```

- `FunctionTool` wraps each Python function as a tool.
- `OpenAILike` connects LlamaIndex to Groq's OpenAI-compatible API.
- `AgentWorkflow` runs the agent loop with native function calling.
- `Context` gives the agent memory across questions.

## Run it in Google Colab

1. Click the **Open in Colab** badge above.
2. Get a free API key at [console.groq.com/keys](https://console.groq.com/keys), then add it in Colab: key icon on the left (**Secrets**), new secret named `GROQ_API_KEY`, and turn on *Notebook access*.
3. **Runtime → Run all**.

The key is read from Colab Secrets and is never stored in the notebook.

## Example questions

- Which region has the highest total revenue?
- What are the top 3 products by units sold?
- Show me all orders with more than 13 units in Cairo.
- Plot revenue by category as a bar chart.

## Files

```
data_analyst_agent.ipynb   the full task (steps 1 to 5)
README.md                  this file
```

The sample dataset (240 synthetic sales rows) is generated inside the notebook, so nothing needs to be uploaded.
