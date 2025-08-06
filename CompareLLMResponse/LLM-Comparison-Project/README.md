# LLM-Comparison-Project

This project compares responses from two large language models: OpenAI GPT-3.5 and Claude 3 Sonnet (Anthropic).

## Features
- Accepts a user prompt
- Queries both OpenAI GPT-3.5 and Claude-3 Sonnet using their APIs
- Prints outputs from both models
- Generates a Markdown report (`report.md`) with:
  - Prompt
  - GPT-3.5 Response
  - Claude Response
  - Comparison Notes

## Setup
1. Clone this repository or copy the project folder.
2. Create and activate a Python virtual environment:
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```
3. Install dependencies:
   ```sh
   pip install openai anthropic markdown2
   ```
4. Set your API keys as environment variables (recommended):
   - `OPENAI_API_KEY` for OpenAI GPT-3.5
   - `ANTHROPIC_API_KEY` for Claude-3 Sonnet
   
   Or, you can enter them at runtime when prompted.

## Usage
Run the main script:
```sh
python main.py
```
Follow the prompts to enter your query and API keys if not set as environment variables.

## Output
- The script prints both model responses to the console.
- A `report.md` file is generated with the prompt, both responses, and a section for comparison notes.