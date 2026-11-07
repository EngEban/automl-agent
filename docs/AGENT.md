# The AutoML Agent

## What is it?
The AutoML Agent is an autonomous AI entity powered by Large Language Models (LLMs). Instead of manually writing code, the user simply asks the agent to "Analyze this dataset" or "Predict the price", and the agent figures out the steps.

## Tools Available
The agent has access to the following tools (defined in `app/agent/tools.py`):
1. `inspect_dataset`: Reads the schema, columns, and sample rows.
2. `run_eda`: Generates exploratory data analysis.
3. `run_preprocessing`: Handles missing values, scaling, and encoding.
4. `train_models`: Trains and compares multiple models.
5. `generate_report`: Creates the PDF and Notebook.
6. `set_target_column`: Changes the target variable.
7. `rerun_pipeline`: Runs the entire process end-to-end.

## The Agent Loop (`agent.py`)
1. User sends a message.
2. The agent sends the conversation + tool schemas to the LLM.
3. If the LLM responds with a tool call, the agent executes it (`executor.py`).
4. The result is appended to the conversation memory.
5. The loop repeats until the LLM returns a final text response.

## Safety & Reliability
- **Loop Breaker**: If a tool fails twice consecutively, the agent stops and reports the error, preventing infinite loops.
- **Token Management**: The conversation history is truncated to fit the LLM's context window.