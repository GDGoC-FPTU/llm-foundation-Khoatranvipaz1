"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    latency_seconds = time.time() - start_time
    
    response_text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }
    
    return response_text, latency_seconds, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.
    """
    # Sử dụng Thư viện mới Google GenAI SDK (Option A)
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens
    )
    
    start_time = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )
    latency_seconds = time.time() - start_time
    
    response_text = response.text
    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count
    }
    
    return response_text, latency_seconds, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    start_time = time.time()
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    latency_seconds = time.time() - start_time
    
    response_text = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }
    
    return response_text, latency_seconds, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.
    """
    # 1. Gọi các API để lấy kết quả
    res_gpt4o, lat_gpt4o, use_gpt4o = call_openai(prompt, model=OPENAI_MODEL)
    res_mini, lat_mini, use_mini = call_openai(prompt, model=OPENAI_MINI_MODEL)
    res_gemini, lat_gemini, use_gemini = call_gemini(prompt, model=GEMINI_MODEL)
    
    # 2. Hàm hỗ trợ tính toán chi phí chính xác theo định dạng yêu cầu
    def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
        rates = PRICING_1M_TOKENS[model_name]
        cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
        return cost

    # 3. Đóng gói kết quả đầu ra
    comparison = {
        "gpt4o": {
            "response": res_gpt4o,
            "latency": lat_gpt4o,
            "cost": calculate_cost(OPENAI_MODEL, use_gpt4o["input_tokens"], use_gpt4o["output_tokens"]),
            "input_tokens": use_gpt4o["input_tokens"],
            "output_tokens": use_gpt4o["output_tokens"]
        },
        "gpt4o_mini": {
            "response": res_mini,
            "latency": lat_mini,
            "cost": calculate_cost(OPENAI_MINI_MODEL, use_mini["input_tokens"], use_mini["output_tokens"]),
            "input_tokens": use_mini["input_tokens"],
            "output_tokens": use_mini["output_tokens"]
        },
        "gemini_flash": {
            "response": res_gemini,
            "latency": lat_gemini,
            "cost": calculate_cost(GEMINI_MODEL, use_gemini["input_tokens"], use_gemini["output_tokens"]),
            "input_tokens": use_gemini["input_tokens"],
            "output_tokens": use_gemini["output_tokens"]
        }
    }
    
    return comparison


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Sử dụng cấu trúc lưu trữ Content Object để duy trì hội thoại đa lượt của thư viện genai mới
    # Chỉ giữ tối đa 3 lượt hội thoại gần nhất (mỗi lượt gồm tin nhắn của 'user' và 'model')
    turns_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session...")
            break
            
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
            
        if not user_input:
            continue
            
        # Thêm lượt chat hiện tại của user vào lịch sử tạm thời để gửi đi
        current_contents = list(turns_history)
        current_contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)]
        ))
        
        print("AI: ", end="", flush=True)
        full_response_text = ""
        
        # Gọi API stream nội dung từ model Gemini 2.5 Flash
        response_stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=current_contents
        )
        
        for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response_text += chunk.text
        print() # Xuống dòng khi kết thúc luồng dữ liệu stream
        
        # Cập nhật lịch sử hội thoại thực tế
        turns_history.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)]
        ))
        turns_history.append(types.Content(
            role="model",
            parts=[types.Part.from_text(text=full_response_text)]
        ))
        
        # Giới hạn lịch sử chỉ giữ tối đa 3 lượt (6 Content objects bao gồm cả cặp phản hồi)
        if len(turns_history) > 6:
            turns_history = turns_history[-6:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            if attempt >= max_retries:
                raise e
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
            attempt += 1


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.
    """
    results = []
    for prompt in prompts:
        comparison_result = compare_models(prompt)
        # Inject thêm key "prompt" gốc vào kết quả trả về
        comparison_result["prompt"] = prompt
        results.append(comparison_result)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.
    """
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    # Định nghĩa cấu trúc map dữ liệu để bóc tách từ kết quả của hàm `batch_compare`
    models_to_check = [
        {"key": "gpt4o", "display_name": "gpt-4o"},
        {"key": "gpt4o_mini", "display_name": "gpt-4o-mini"},
        {"key": "gemini_flash", "display_name": "gemini-2.5-flash"}
    ]
    
    for item in results:
        prompt = item["prompt"]
        for m in models_to_check:
            stats = item[m["key"]]
            
            # Xử lý chuỗi xuống dòng thành khoảng trắng và cắt ngắn chuỗi hội thoại còn 50 kí tự
            clean_response = stats["response"].replace("\n", " ")
            if len(clean_response) > 50:
                truncated_res = clean_response[:47] + "..."
            else:
                truncated_res = clean_response
                
            latency_str = f"{stats['latency']:.2f}s"
            tokens_str = f"{stats['input_tokens']}/{stats['output_tokens']}"
            cost_str = f"${stats['cost']:.6f}"
            
            lines.append(f"| {prompt} | {m['display_name']} | {truncated_res} | {latency_str} | {tokens_str} | {cost_str} |")
            
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")