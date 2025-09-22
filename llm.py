import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class GenerateRequest(BaseModel):
    prompt: str
    system: Optional[str] = (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    )
    max_new_tokens: int = 512


class GenerateResponse(BaseModel):
    response: str


model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# Load model and tokenizer once at startup
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def generate_response(prompt: str, system: Optional[str], max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": system or ""},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response_text


app = FastAPI(title="LLM Service", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="'prompt' must be a non-empty string")
    try:
        output = generate_response(req.prompt.strip(), req.system, req.max_new_tokens)
        return GenerateResponse(response=output)
    except Exception as exc:  # Keep broad to return 500 with message
        raise HTTPException(status_code=500, detail=str(exc))


# If executed directly: run a quick demo generation for convenience
if __name__ == "__main__":
    demo = generate_response(
        prompt="Give me a short introduction to large language models.",
        system=(
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        ),
        max_new_tokens=128,
    )
    print(demo)
