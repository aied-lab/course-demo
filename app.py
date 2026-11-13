import os
import io
import base64
import requests
import gradio as gr

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-27b-it")
API_URL = "https://router.huggingface.co/v1/chat/completions"


def analyze(image, prompt):
    if image is None:
        return "請先上傳一張圖片！"
    if not HF_TOKEN:
        return "後端未設定 HF_TOKEN，請在 Space 的 Settings > Variables and secrets 加入 HF_TOKEN。"

    prompt = (prompt or "").strip() or "用一句中文描述這張圖片"

    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG", quality=85)
    b64 = base64.b64encode(buffered.getvalue()).decode()
    data_url = f"data:image/jpeg;base64,{b64}"

    payload = {
        "model": MODEL_ID,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "max_tokens": 300,
    }

    try:
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        data = resp.json()
        if resp.ok and data.get("choices"):
            return data["choices"][0]["message"]["content"]
        err = data.get("error", {})
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return f"發生錯誤：{msg or data}"
    except Exception as e:
        return f"發生錯誤：{e}"


CSS = """
:root {
    --neon-cyan: #00f0ff;
    --neon-purple: #b026ff;
}
.gradio-container {
    background-color: #0b0f19 !important;
    background-image:
        radial-gradient(circle at 10% 20%, rgba(176, 38, 255, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(0, 240, 255, 0.12) 0%, transparent 40%) !important;
    color: #f3f4f6 !important;
    max-width: 780px !important;
    margin: 0 auto !important;
}
#title h1 {
    background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
#subtitle { color: #9ca3af !important; }
#analyze-btn {
    background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple)) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(0, 240, 255, 0.2) !important;
}
#analyze-btn:hover { box-shadow: 0 6px 20px rgba(176, 38, 255, 0.45) !important; }
"""

with gr.Blocks(css=CSS, title="Gemma 多模態影像分析助手") as demo:
    with gr.Column(elem_id="title"):
        gr.Markdown("# Gemma 多模態影像分析助手")
    with gr.Column(elem_id="subtitle"):
        gr.Markdown("透過 Hugging Face Router 呼叫多模態模型進行高精度影像辨識與分析。")

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="上傳圖片")
            prompt_in = gr.Textbox(
                label="輸入提示詞 (Prompt)",
                value="用一句中文描述這張圖片",
            )
            btn = gr.Button("啟動分析", elem_id="analyze-btn", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="模型回應", lines=14, show_copy_button=True)

    btn.click(analyze, inputs=[image_in, prompt_in], outputs=output)

demo.launch()
