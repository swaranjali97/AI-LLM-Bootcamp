import os
import openai
import anthropic


def get_openai_response(prompt, api_key):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error from OpenAI: {e}"

def get_claude_response(prompt, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"Error from Claude: {e}"

def main():
    print("=== LLM Comparison Project ===")
    prompt = input("Enter your prompt: ")

    openai_key = os.getenv("OPENAI_API_KEY") or input("Enter your OpenAI API key: ")
    claude_key = os.getenv("ANTHROPIC_API_KEY") or input("Enter your Anthropic API key: ")

    print("\nQuerying OpenAI GPT-3.5...")
    gpt_response = get_openai_response(prompt, openai_key)
    print("\nQuerying Claude-3 Sonnet...")
    claude_response = get_claude_response(prompt, claude_key)

    print("\n--- GPT-3.5 Response ---\n", gpt_response)
    print("\n--- Claude-3 Sonnet Response ---\n", claude_response)

    # Prepare for Markdown report
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(f"# LLM Comparison Report\n\n")
        f.write(f"## Prompt\n\n{prompt}\n\n")
        f.write(f"## GPT-3.5 Response\n\n{gpt_response}\n\n")
        f.write(f"## Claude Response\n\n{claude_response}\n\n")
        f.write(f"## Comparison Notes\n\n- ")
    print("\nReport saved to report.md")

if __name__ == "__main__":
    main()