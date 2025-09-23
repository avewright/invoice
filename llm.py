import os
import base64
from io import BytesIO
from typing import Optional, List, Any, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageOps
import requests
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from model import ensure_invoice_shape


class GenerateRequest(BaseModel):
    # Optional simple text prompt (backward compatible)
    prompt: Optional[str] = None
    system: Optional[str] = (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    )
    max_new_tokens: int = 2048
    # Preferred: pass multimodal messages (supports image URLs/base64/data URIs)
    messages: Optional[List[Any]] = None


class GenerateResponse(BaseModel):
    response: dict


model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")

# Load VL model and processor once at startup
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)


def _normalize_image_mode(img: Image.Image) -> Image.Image:
    """Convert image to RGB, correctly handling transparency/palette to avoid warnings."""
    try:
        # Respect EXIF orientation
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    try:
        has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
        if has_alpha:
            # Composite on white background
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            return background.convert("RGB")
        if img.mode != "RGB":
            return img.convert("RGB")
        return img
    except Exception:
        return img.convert("RGB")


def _decode_image(image_ref: Any) -> Optional[Image.Image]:
    try:
        if isinstance(image_ref, Image.Image):
            return _normalize_image_mode(image_ref)
        if isinstance(image_ref, bytes):
            return _normalize_image_mode(Image.open(BytesIO(image_ref)))
        if not isinstance(image_ref, str):
            return None
        ref = image_ref.strip()
        if ref.startswith("http://") or ref.startswith("https://"):
            resp = requests.get(ref, timeout=30)
            resp.raise_for_status()
            return _normalize_image_mode(Image.open(BytesIO(resp.content)))
        if ref.startswith("data:image") and ";base64," in ref:
            b64 = ref.split(",", 1)[1]
            return _normalize_image_mode(Image.open(BytesIO(base64.b64decode(b64))))
        # raw base64
        try:
            decoded = base64.b64decode(ref, validate=True)
            return _normalize_image_mode(Image.open(BytesIO(decoded)))
        except Exception:
            pass
        if os.path.exists(ref):
            return _normalize_image_mode(Image.open(ref))
    except Exception:
        return None
    return None


def _build_messages(prompt: Optional[str], system: Optional[str], messages: Optional[List[Any]]) -> List[dict]:
    if messages:
        return messages
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Provide 'messages' or a non-empty 'prompt'")
    msgs: List[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": [{"type": "text", "text": prompt.strip()}]})
    return msgs


def _process_vision_info(messages: List[Any]) -> Tuple[List[Image.Image], List[Any]]:
    image_inputs: List[Image.Image] = []
    video_inputs: List[Any] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                img = _decode_image(item.get("image"))
                if img is not None:
                    image_inputs.append(img)
            # videos not handled here
    return image_inputs, video_inputs


def generate_response(
    prompt: Optional[str],
    system: Optional[str],
    max_new_tokens: int,
    messages: Optional[List[Any]] = None,
) -> str:
    msgs = _build_messages(prompt, system, messages)

    text = processor.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = _process_vision_info(msgs)

    proc_kwargs: dict = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs:
        proc_kwargs["images"] = image_inputs
    if video_inputs:
        proc_kwargs["videos"] = video_inputs

    model_inputs = processor(**proc_kwargs)
    model_inputs = model_inputs.to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return response_text


app = FastAPI(title="LLM Service", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if not req.messages and not (req.prompt and req.prompt.strip()):
        raise HTTPException(status_code=400, detail="Provide 'messages' or a non-empty 'prompt'")
    try:
        output_text = generate_response(
            prompt=(req.prompt.strip() if req.prompt else None),
            system=req.system,
            max_new_tokens=req.max_new_tokens,
            messages=req.messages,
        )
        try:
            normalized = ensure_invoice_shape(output_text)
        except Exception as exc:  # Validation or parsing failed
            raise HTTPException(status_code=422, detail=f"Normalization/validation failed: {exc}")
        return GenerateResponse(response=normalized)
    except Exception as exc:  # Keep broad to return 500 with message
        raise HTTPException(status_code=500, detail=str(exc))


# If executed directly: run a quick demo generation for convenience
if __name__ == "__main__":
    demo = generate_response(
        prompt="Describe the scene in one sentence.",
        system=("You are a helpful assistant."),
        max_new_tokens=64,
        messages=None,
    )
    print(demo)
