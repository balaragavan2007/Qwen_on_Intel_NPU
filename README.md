# Qwen 3 8B on Intel NPU with Open﻿Vino-genai

This project demonstrates running **Qwen 3 (8B)** on an **Intel Ultra Series 1 & 2 NPU** using **OpenVINO GenAI**.  
The model is optimized with **INT4 quantization**, making it faster ⚡ and smaller 🚀.

---

## ✨ Features
- Runs **Qwen LLM** locally on Intel NPU
- Uses **OpenVINO GenAI**
- **INT4 quantization** for efficient inference
- Works **offline** (no GPU required)
- Built with **Hugging Face Transformers + Optimum-Intel**

---

## 📦 Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/balaragavan2007/Qwen_on_Intel_NPU.git
cd Qwen_on_Intel_NPU
python -m venv llm
llm\Scripts\activate
pip install -r requirements.txt
```

---

## ⚡ Model Download

Download the model locally by the following command

```bash
huggingface-cli download OpenVINO/Qwen3-8B-int4-cw-ov --local-dir Qwen3-8B-NPU-Model
```
- This will create a folder Qwen3-8B-NPU-Model with the OpenVINO-optimized model.

---

## ▶️ Usage

- Run inference with:

```bash
python run.py
```

---

## Performance

Running the Qwen 3-8B model on my [ASUS Vivobook S16 OLED with Intel Core Ultra 5 125H], I achieved the following performance:

| Device | Performance      |
|--------|------------------|
| CPU    | [7.8] tok/s |
| GPU    | [13.11] tok/s |
| NPU    | [8.77] tok/s |


---

## 📸 Demo
<img width="1920" height="1080" alt="Screenbox_20251128_001106" src="https://github.com/user-attachments/assets/85bb9513-c760-44f0-990f-420bc9d0d40a" />

---
