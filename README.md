---
title: Demo
emoji: 👁
colorFrom: purple
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
short_description: Gemma multimodal image analysis
---

Gemma 多模態影像分析助手，透過 Hugging Face Router 呼叫多模態模型。

## 設定

請在此 Space 的 **Settings > Variables and secrets** 新增：

- `HF_TOKEN`（Secret）：你的 Hugging Face access token
- `MODEL_ID`（Variable，選填）：預設 `google/gemma-3-27b-it`，可換成其他 Router 支援的多模態模型
