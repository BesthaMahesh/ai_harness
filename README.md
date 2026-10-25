# AI Agent Email Classification Harness

A resilient email classification agent harness featuring multi-tier validation, security guardrails, retry logic, structured JSON parsing, and evaluation testing using Groq LLM API.

## Features

- **Input Validation**: Sanitizes and validates incoming emails before passing them to the AI agent.
- **AI Classification**: Categorizes emails into `IMPORTANT`, `NORMAL`, or `SPAM` with reasoning via Groq LLMs.
- **Output Validation & Retries**: Validates model output against strict schema requirements with automated retries.
- **Security Policy Guardrail**: Intercepts credential phishing or suspicious requests and routes them to `REVIEW_REQUIRED`.
- **Structured Event Logging**: Logs system events, validation checks, and benchmark latencies in `harness.log`.
- **Evaluation Suite**: Built-in test suite evaluating classification accuracy against edge cases.

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Add your Groq API key:
```env
GROQ_API_KEY="your_groq_api_key_here"
```

### 3. Run the Application

```bash
python app.py
```

From the interactive menu:
1. **Classify an email**: Test individual emails interactively.
2. **Run evaluation tests**: Run benchmark tests against predefined test cases.
3. **Exit**.
