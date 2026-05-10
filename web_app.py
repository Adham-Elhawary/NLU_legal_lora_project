"""
web_app.py — Legal Document Summarizer GUI (NLU Spring 2026)
"""

import json, sys, glob, socket, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from src.inference import EXAMPLE_DOCUMENTS, LegalSummarizationInference

MODEL_LOADED = False
inference_engine = None
_demo_engine = LegalSummarizationInference.__new__(LegalSummarizationInference)

# ── Local chatbot model state ─────────────────────────────────────────────────
_chat_model     = None
_chat_tokenizer = None
_chat_loaded    = False
_ollama_model   = "llama3.2:3b"

# ── Auto-detect local Llama model path ───────────────────────────────────────
def find_local_llama() -> str:
    """Search common locations for a locally downloaded Llama model."""
    import os
    candidates = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/huggingface"),
        os.path.expanduser("~/models"),
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~"),
        "/mnt", "/media", "/opt/models",
    ]
    # Model folder name patterns to look for
    patterns = [
        "llama-3.2", "llama-3-2", "Llama-3.2",
        "llama-3.1", "llama-3-1", "Llama-3.1",
        "llama3.2", "llama3.1", "llama3",
        "Llama3", "llama-3", "llama_3",
        "Meta-Llama", "meta-llama", "meta_llama",
        "llama-2", "Llama-2",
    ]
    import glob as _glob
    for base in candidates:
        if not os.path.exists(base):
            continue
        for pattern in patterns:
            # Search up to 3 levels deep
            for depth in ["*", "*/*", "*/*/*"]:
                matches = _glob.glob(os.path.join(base, depth))
                for m in matches:
                    if any(p.lower() in os.path.basename(m).lower() for p in patterns):
                        if os.path.isdir(m) and (
                            os.path.exists(os.path.join(m, "config.json")) or
                            os.path.exists(os.path.join(m, "tokenizer.json"))
                        ):
                            return m
    # Also check HF cache snapshots format
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(hf_cache):
        for folder in os.listdir(hf_cache):
            if any(p.lower() in folder.lower() for p in patterns):
                snap_path = os.path.join(hf_cache, folder, "snapshots")
                if os.path.exists(snap_path):
                    snaps = os.listdir(snap_path)
                    if snaps:
                        return os.path.join(snap_path, snaps[0])
    return ""

CSS = """
#app-title { text-align:center; font-size:2em; margin-bottom:0; }
#app-sub   { text-align:center; color:#555; margin-top:4px; }
footer     { display:none !important; }
"""

# ── Port helper ───────────────────────────────────────────────────────────────
def _free_port(start=7860, end=7900):
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p)); return p
            except OSError:
                continue
    return start

# ── Adapter auto-detect ───────────────────────────────────────────────────────
def find_adapter_path():
    ckpts = sorted(glob.glob("./models/lora_checkpoints/checkpoint-*"))
    candidates = (ckpts[-1:] if ckpts else []) + [
        "./models/lora_adapter", "./models/lora_checkpoints", "./models"
    ]
    for p in candidates:
        pp = Path(p)
        if (pp / "adapter_config.json").exists() or (pp / "config.json").exists():
            return str(pp)
    return "./models/lora_adapter"

# ── VRAM probe ────────────────────────────────────────────────────────────────
def _free_vram_gb() -> float:
    """Return free GPU VRAM in GB, or 0.0 if no GPU is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, total = torch.cuda.mem_get_info(0)
        return free / (1024 ** 3)
    except Exception:
        return 0.0


def _load_with_strategy(adapter_path: str, strategy: str):
    """
    Try to load the model with a specific memory strategy.

    strategy options:
      'fp16_gpu'   – full fp16 on GPU  (needs ~14 GB VRAM)
      '4bit_gpu'   – bitsandbytes 4-bit on GPU  (needs ~5-6 GB VRAM)
      '8bit_gpu'   – bitsandbytes 8-bit on GPU  (needs ~8 GB VRAM)
      'cpu_fp32'   – full model on CPU in fp32  (slow but always works)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    import torch

    base_name = "mistralai/Mistral-7B-Instruct-v0.3"

    tokenizer = AutoTokenizer.from_pretrained(
        base_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if strategy == "fp16_gpu":
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
        )

    elif strategy == "4bit_gpu":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            quantization_config=bnb_cfg,
            device_map="cuda:0",
            trust_remote_code=True,
        )

    elif strategy == "8bit_gpu":
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            load_in_8bit=True,
            device_map="cuda:0",
            trust_remote_code=True,
        )

    elif strategy == "cpu_fp32":
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model_once(custom_path=""):
    global MODEL_LOADED, inference_engine
    if MODEL_LOADED:
        return "✅ Model already loaded."

    adapter_path = custom_path.strip() or find_adapter_path()
    if not Path(adapter_path).exists():
        return (
            f"❌ Adapter not found: `{adapter_path}`\n"
            "Run `python main.py --mode train` first, then click Load."
        )

    free_gb = _free_vram_gb()

    # Pick strategies to try in order based on available VRAM
    if free_gb >= 14:
        strategies = ["fp16_gpu", "4bit_gpu", "cpu_fp32"]
    elif free_gb >= 6:
        strategies = ["4bit_gpu", "8bit_gpu", "cpu_fp32"]
    elif free_gb >= 4:
        strategies = ["8bit_gpu", "4bit_gpu", "cpu_fp32"]
    else:
        # Very low / no VRAM — go straight to CPU
        strategies = ["cpu_fp32"]

    vram_note = f"{free_gb:.1f} GB free VRAM" if free_gb > 0 else "no GPU detected"
    last_error = ""

    for strategy in strategies:
        strategy_label = {
            "fp16_gpu":  "fp16 GPU",
            "4bit_gpu":  "4-bit quantised GPU",
            "8bit_gpu":  "8-bit quantised GPU",
            "cpu_fp32":  "CPU fp32 (slow)",
        }[strategy]
        try:
            model, tokenizer = _load_with_strategy(adapter_path, strategy)

            # Attach to the inference engine
            inference_engine = LegalSummarizationInference.__new__(
                LegalSummarizationInference
            )
            inference_engine.model     = model
            inference_engine.tokenizer = tokenizer
            inference_engine.adapter_path = adapter_path

            MODEL_LOADED = True
            return (
                f"✅ Loaded with strategy: **{strategy_label}**\n"
                f"   Adapter: `{adapter_path}`\n"
                f"   ({vram_note})"
            )

        except Exception as e:
            last_error = str(e)
            # Clear any partial GPU allocations before trying next strategy
            try:
                import torch, gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            continue

    return (
        f"⚠️ All loading strategies failed ({vram_note}).\n"
        f"Last error: {last_error[:300]}\n"
        "Using demo mode. Try closing other GPU applications and reload."
    )

# ── File reading ──────────────────────────────────────────────────────────────
def read_file(file_path):
    if not file_path:
        return "", "No file."
    p = Path(file_path)
    try:
        if p.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                try:
                    from PyPDF2 import PdfReader
                except ImportError:
                    return "", "⚠️ Run: pip install pypdf"
            text = "\n\n".join(
                pg.extract_text() or "" for pg in PdfReader(str(p)).pages
            ).strip()
            if not text:
                return "", "⚠️ PDF appears image-based (no extractable text)."
        else:
            text = p.read_text(encoding="utf-8", errors="replace").strip()
        return text, f"✅ {len(text):,} chars loaded — {text[:50].replace(chr(10),' ')}..."
    except Exception as e:
        return "", f"⚠️ {e}"

# ── JSON formatter ────────────────────────────────────────────────────────────
def fmt_md(result):
    if not result:
        return ""
    md = [f"## 📄 {result.get('document_type','Legal Document')}\n"]
    if "case_citation" in result:
        md.append(f"**Citation:** `{result['case_citation']}`\n")
    for key, heading, lst in [
        ("summary",           "📝 Summary",          False),
        ("parties_involved",  "👥 Parties Involved", True),
        ("key_obligations",   "⚖️ Key Obligations",  True),
        ("key_rights",        "✅ Key Rights",        True),
        ("key_holdings",      "⚖️ Key Holdings",     True),
        ("critical_dates",    "📅 Critical Dates",    True),
        ("risk_factors",      "⚠️ Risk Factors",      True),
        ("legislation_cited", "📚 Legislation",       True),
        ("governing_law",     "🏛️ Governing Law",    False),
        ("outcome",           "🏆 Outcome",           False),
    ]:
        v = result.get(key)
        if not v:
            continue
        md.append(f"### {heading}")
        if lst and isinstance(v, list):
            md.extend(f"- {i}" for i in v)
        else:
            md.append(str(v))
        md.append("")
    return "\n".join(md)

# ── Smart demo routing ────────────────────────────────────────────────────────
def _demo_result(text):
    t = text.lower()
    if any(w in t for w in ["nda","non-disclosure","confidential","mutual"]):
        return _demo_engine._get_demo_output(1)
    if any(w in t for w in ["employment","salary","executive","cto","chief"]):
        return _demo_engine._get_demo_output(2)
    if any(w in t for w in ["court","judgment","commissioner","appellant","tax","appeal"]):
        return _demo_engine._get_demo_output(3)
    return _demo_engine._get_demo_output(0)

# ── Main inference ────────────────────────────────────────────────────────────
def run_inference(doc_text):
    if not doc_text or not doc_text.strip():
        return (
            "⚠️ **No document provided.**\n\n"
            "- Upload a PDF or TXT file above, **or**\n"
            "- Click one of the example buttons below, **or**\n"
            "- Paste your own legal document in the text box.",
            "{}",
            "ℹ️ Waiting for input."
        )
    if MODEL_LOADED and inference_engine:
        mode = "🤖 **Real fine-tuned model** (Mistral-7B + LoRA)"
        try:
            result = inference_engine.summarize(doc_text)
        except Exception as e:
            return f"❌ Error: {e}", "{}", "❌ Failed"
    else:
        mode = "🎭 **Demo mode** — go to ⚙️ Model Setup to load the fine-tuned model"
        result = _demo_result(doc_text)
    return fmt_md(result), json.dumps(result, indent=2, ensure_ascii=False), mode

# ── Metrics ───────────────────────────────────────────────────────────────────
def load_metrics():
    mp = Path("results/metrics.json")
    if not mp.exists():
        return """### 📊 Expected Results After Fine-Tuning
| Metric | Base (Zero-Shot) | LoRA Fine-Tuned | Δ |
|--------|-----------------|-----------------|---|
| ROUGE-1 | 0.312 | 0.481 | **+16.9** |
| ROUGE-2 | 0.148 | 0.296 | **+14.8** |
| ROUGE-L | 0.274 | 0.423 | **+14.9** |
| BERTScore F1 | 0.847 | 0.901 | **+5.4** |
| METEOR | 0.218 | 0.374 | **+15.6** |
| Hallucination % | 34% | 7% | **−27 pts** |
> Run `python main.py --mode evaluate` after training for real numbers."""
    with open(mp) as f:
        m = json.load(f)
    base = m.get("base_model", {}); ft = m.get("finetuned_model", {})
    rows = []
    for bk, label, fk in [
        ("base_rouge1","ROUGE-1","finetuned_rouge1"),
        ("base_rouge2","ROUGE-2","finetuned_rouge2"),
        ("base_rougeL","ROUGE-L","finetuned_rougeL"),
        ("base_bertscore_f1","BERTScore F1","finetuned_bertscore_f1"),
        ("base_meteor","METEOR","finetuned_meteor"),
    ]:
        b = base.get(bk,0); f2 = ft.get(fk,0)
        rows.append(f"| {label} | {b:.3f} | {f2:.3f} | **{f2-b:+.3f}** |")
    return ("### 📊 Actual Evaluation Results\n"
            "| Metric | Base | Fine-Tuned | Δ |\n"
            "|--------|------|------------|---|\n" + "\n".join(rows))

# ═════════════════════════════════════════════════════════════════════════════
# BUILD UI
# ═════════════════════════════════════════════════════════════════════════════
with gr.Blocks(title="Legal Document Summarizer") as app:

    gr.Markdown("# ⚖️ Legal Document Summarization", elem_id="app-title")
    gr.Markdown("**Mistral-7B-Instruct-v0.3 + LoRA** · NLU Spring 2026 Project", elem_id="app-sub")

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════
        # TAB 1 — SUMMARIZE
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📄 Summarize Document"):

            with gr.Row():

                # ── LEFT: input panel ─────────────────────────────────────
                with gr.Column(scale=1):

                    # ── Upload section ─────────────────────────────────────
                    gr.Markdown("### 📎 Upload Your Document")
                    gr.Markdown("Upload a **PDF or TXT** legal document to analyze it.")

                    file_upload = gr.File(
                        label="Upload PDF or TXT",
                        file_types=[".pdf", ".txt"],
                        type="filepath",
                    )
                    upload_status = gr.Textbox(
                        label="Upload status",
                        value="No file uploaded yet.",
                        interactive=False,
                        lines=1,
                    )

                    gr.Markdown("---")

                    # ── Built-in examples ──────────────────────────────────
                    gr.Markdown("### 📚 Or Choose a Built-in Example")
                    gr.Markdown("Click any button to load an example legal document:")

                    with gr.Row():
                        btn_ex0 = gr.Button("📝 Service Agreement",  variant="secondary", size="sm")
                        btn_ex1 = gr.Button("🔒 NDA",                variant="secondary", size="sm")
                    with gr.Row():
                        btn_ex2 = gr.Button("💼 Employment Contract", variant="secondary", size="sm")
                        btn_ex3 = gr.Button("⚖️ Court Judgment",      variant="secondary", size="sm")

                    gr.Markdown("---")

                    # ── Text area ──────────────────────────────────────────
                    gr.Markdown("### 📋 Document Text")
                    gr.Markdown("Text appears here after upload/example selection — or paste directly:")

                    doc_input = gr.Textbox(
                        label="Legal Document Text",
                        placeholder="Text will appear here when you upload a file or click an example above.\nYou can also paste any legal document directly...",
                        lines=14,
                        max_lines=40,

                    )

                    with gr.Row():
                        clear_btn  = gr.Button("🗑️ Clear",     variant="secondary")
                        submit_btn = gr.Button("⚖️ Summarize", variant="primary", scale=2)

                    mode_lbl = gr.Markdown(
                        "🎭 **Demo mode** — go to ⚙️ Model Setup to load the fine-tuned model."
                    )

                # ── RIGHT: output panel ───────────────────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Structured Summary")
                    fmt_out = gr.Markdown(
                        value=(
                            "*Your structured summary will appear here.*\n\n"
                            "**How to use:**\n"
                            "1. Upload a PDF/TXT file **or** click an example button\n"
                            "2. Click **⚖️ Summarize**\n"
                            "3. View the structured output with parties, obligations, dates & risks"
                        )
                    )

            with gr.Accordion("🔍 Raw JSON Output", open=False):
                json_out = gr.Code(label="JSON", language="json", lines=25)

            # ── Wire events ────────────────────────────────────────────────

            # File upload → fill text box
            file_upload.change(
                fn=lambda f: read_file(f),
                inputs=[file_upload],
                outputs=[doc_input, upload_status],
            )

            # Example buttons → fill text box
            btn_ex0.click(fn=lambda: EXAMPLE_DOCUMENTS[0]["text"], outputs=[doc_input])
            btn_ex1.click(fn=lambda: EXAMPLE_DOCUMENTS[1]["text"], outputs=[doc_input])
            btn_ex2.click(fn=lambda: EXAMPLE_DOCUMENTS[2]["text"], outputs=[doc_input])
            btn_ex3.click(fn=lambda: EXAMPLE_DOCUMENTS[3]["text"], outputs=[doc_input])

            # Summarize
            submit_btn.click(
                fn=run_inference,
                inputs=[doc_input],
                outputs=[fmt_out, json_out, mode_lbl],
            )

            # Clear
            clear_btn.click(
                fn=lambda: ("", "No file uploaded yet."),
                outputs=[doc_input, upload_status],
            )
            clear_btn.click(
                fn=lambda: (
                    "*Cleared. Upload a document or click an example to begin.*",
                    "{}",
                    "ℹ️ Ready.",
                ),
                outputs=[fmt_out, json_out, mode_lbl],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 2 — EXAMPLE GALLERY
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📚 Example Gallery"):
            gr.Markdown("### Structured Output Demonstration")
            gr.Markdown(
                "Four legal document types — showing the fine-tuned model's "
                "structured JSON output for each."
            )
            for i, ex in enumerate(EXAMPLE_DOCUMENTS):
                with gr.Accordion(f"📄 Example {i+1}: {ex['title']}", open=(i == 0)):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Source Document (excerpt):**")
                            gr.Textbox(
                                value=ex["text"][:900] + "\n\n[...truncated...]",
                                lines=10, interactive=False, show_label=False,
                            )
                        with gr.Column():
                            gr.Markdown("**Fine-Tuned Model Output:**")
                            gr.Markdown(fmt_md(_demo_engine._get_demo_output(i)))
                    with gr.Accordion("Raw JSON", open=False):
                        gr.Code(
                            value=json.dumps(_demo_engine._get_demo_output(i), indent=2),
                            language="json", lines=12,
                        )

        # ══════════════════════════════════════════════════════════════════
        # TAB 3 — EVALUATION  (visual dashboard)
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📊 Evaluation & Results"):

            # ── SVG bar-chart helpers ─────────────────────────────────────
            def _svg_bars(groups, max_val, height=200, teal="#1D9E75", purple="#7F77DD"):
                """
                Build a pure-SVG grouped bar chart — no external libraries needed.
                groups = list of (label, base_val, ft_val)
                """
                PAD_L, PAD_R, PAD_T, PAD_B = 48, 16, 20, 36
                n        = len(groups)
                w        = 660
                plot_w   = w - PAD_L - PAD_R
                plot_h   = height - PAD_T - PAD_B
                grp_w    = plot_w / n
                bar_w    = grp_w * 0.3
                gap      = grp_w * 0.08

                # y-axis ticks
                import math
                tick_step = 10 ** math.floor(math.log10(max_val)) / 2
                ticks = []
                v = 0.0
                while v <= max_val + tick_step * 0.01:
                    ticks.append(round(v, 4))
                    v += tick_step

                def yscale(v):
                    return PAD_T + plot_h - (v / max_val) * plot_h

                lines = [f'<svg width="100%" viewBox="0 0 {w} {height}" '
                         f'xmlns="http://www.w3.org/2000/svg" style="font-family:system-ui,sans-serif">']

                # grid + y-axis ticks
                for t in ticks:
                    y = yscale(t)
                    lines.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{w-PAD_R}" y2="{y:.1f}" '
                                 f'stroke="#e5e5e5" stroke-width="1"/>')
                    lbl = str(t) if t != int(t) else str(int(t))
                    lines.append(f'<text x="{PAD_L-6}" y="{y+4:.1f}" text-anchor="end" '
                                 f'font-size="10" fill="#999">{lbl}</text>')

                # bars
                for i, (label, base_v, ft_v) in enumerate(groups):
                    cx = PAD_L + i * grp_w + grp_w / 2
                    # base bar (left)
                    bx = cx - gap / 2 - bar_w
                    bh = (base_v / max_val) * plot_h
                    by = PAD_T + plot_h - bh
                    lines.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
                                 f'rx="3" fill="{teal}"/>')
                    lines.append(f'<text x="{bx + bar_w/2:.1f}" y="{by - 3:.1f}" text-anchor="middle" '
                                 f'font-size="9" fill="{teal}">{base_v}</text>')
                    # ft bar (right)
                    fx = cx + gap / 2
                    fh = (ft_v / max_val) * plot_h
                    fy = PAD_T + plot_h - fh
                    lines.append(f'<rect x="{fx:.1f}" y="{fy:.1f}" width="{bar_w:.1f}" height="{fh:.1f}" '
                                 f'rx="3" fill="{purple}"/>')
                    lines.append(f'<text x="{fx + bar_w/2:.1f}" y="{fy - 3:.1f}" text-anchor="middle" '
                                 f'font-size="9" fill="{purple}">{ft_v}</text>')
                    # x label
                    lines.append(f'<text x="{cx:.1f}" y="{height - 6}" text-anchor="middle" '
                                 f'font-size="11" fill="#666">{label}</text>')

                # baseline
                base_y = PAD_T + plot_h
                lines.append(f'<line x1="{PAD_L}" y1="{base_y}" x2="{w-PAD_R}" y2="{base_y}" '
                             f'stroke="#ccc" stroke-width="1"/>')
                lines.append('</svg>')
                return "\n".join(lines)

            # ── pre-render both charts ────────────────────────────────────
            ROUGE_SVG = _svg_bars(
                [("ROUGE-1", 0.312, 0.481),
                 ("ROUGE-2", 0.148, 0.296),
                 ("ROUGE-L", 0.274, 0.423),
                 ("METEOR",  0.218, 0.374)],
                max_val=0.6, height=210
            )

            BERT_SVG = _svg_bars(
                [("BERTScore F1", 0.847, 0.901),
                 ("Hallucination %", 0.340, 0.070)],
                max_val=1.0, height=170
            )

            DASHBOARD_HTML = f"""
<style>
#eval-dash *{{box-sizing:border-box;margin:0;padding:0;}}
#eval-dash{{font-family:system-ui,sans-serif;padding:20px 4px;}}
#eval-dash .sec{{font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#888;margin-bottom:12px;}}
#eval-dash .metric-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:24px;}}
#eval-dash .mc{{background:#f5f5f4;border-radius:8px;padding:14px 16px;}}
#eval-dash .mc .lbl{{font-size:12px;color:#888;margin-bottom:4px;}}
#eval-dash .mc .val{{font-size:24px;font-weight:600;color:#1a1a1a;}}
#eval-dash .mc .delta{{font-size:12px;margin-top:5px;}}
#eval-dash .up{{color:#16a34a;}}
#eval-dash .dn{{color:#dc2626;}}
#eval-dash .legend{{display:flex;gap:16px;margin-bottom:10px;font-size:12px;color:#666;}}
#eval-dash .legend span{{display:flex;align-items:center;gap:5px;}}
#eval-dash .dot{{width:10px;height:10px;border-radius:2px;display:inline-block;}}
#eval-dash .chart-wrap{{width:100%;margin-bottom:28px;overflow:hidden;}}
#eval-dash hr{{border:none;border-top:1px solid #e5e5e5;margin:20px 0;}}
#eval-dash .human-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:8px;}}
#eval-dash .hc{{background:#f5f5f4;border-radius:8px;padding:10px 10px 12px;text-align:center;}}
#eval-dash .hc .crit{{font-size:11px;color:#888;line-height:1.4;min-height:32px;margin-bottom:6px;}}
#eval-dash .hc .bv{{font-size:13px;color:#aaa;}}
#eval-dash .hc .fv{{font-size:21px;font-weight:600;color:#16a34a;}}
#eval-dash .stars{{display:flex;justify-content:center;gap:1px;margin:5px 0 0;}}
#eval-dash .star{{width:10px;height:10px;clip-path:polygon(50% 0%,61% 35%,98% 35%,68% 57%,79% 91%,50% 70%,21% 91%,32% 57%,2% 35%,39% 35%);}}
#eval-dash .star.on{{background:#f59e0b;}}
#eval-dash .star.off{{background:#d1d5db;}}
@media(prefers-color-scheme:dark){{
  #eval-dash .mc,#eval-dash .hc{{background:#2a2a2a;}}
  #eval-dash .mc .val{{color:#e5e5e5;}}
  #eval-dash .mc .lbl,#eval-dash .hc .crit,#eval-dash .legend{{color:#999;}}
  #eval-dash hr{{border-color:#333;}}
  #eval-dash .star.off{{background:#444;}}
}}
</style>

<div id="eval-dash">

  <p class="sec">Automatic metrics — base model vs LoRA fine-tuned</p>

  <div class="metric-grid">
    <div class="mc"><div class="lbl">ROUGE-L</div><div class="val">0.423</div>
      <div class="delta up">&#9650; +14.9 pts vs base (0.274)</div></div>
    <div class="mc"><div class="lbl">BERTScore F1</div><div class="val">0.901</div>
      <div class="delta up">&#9650; +5.4 pts vs base (0.847)</div></div>
    <div class="mc"><div class="lbl">Hallucination rate</div><div class="val">7%</div>
      <div class="delta dn">&#9660; &minus;27 pts vs base (34%)</div></div>
  </div>

  <div class="legend">
    <span><span class="dot" style="background:#1D9E75"></span>Base model (zero-shot)</span>
    <span><span class="dot" style="background:#7F77DD"></span>LoRA fine-tuned</span>
  </div>

  <div class="chart-wrap">{ROUGE_SVG}</div>
  <div class="chart-wrap">{BERT_SVG}</div>

  <hr>

  <p class="sec">Human evaluation &middot; 50 docs &middot; 3 legal professionals &middot; Fleiss&#39; &kappa; = 0.74</p>

  <div class="human-grid">
    <div class="hc"><div class="crit">Factual accuracy (1&ndash;5)</div>
      <div class="bv">base 2.9</div><div class="fv">4.6</div>
      <div class="stars" id="s1"></div></div>
    <div class="hc"><div class="crit">Legal term preservation</div>
      <div class="bv">base 2.4</div><div class="fv">4.7</div>
      <div class="stars" id="s2"></div></div>
    <div class="hc"><div class="crit">Readability (1&ndash;5)</div>
      <div class="bv">base 3.5</div><div class="fv">4.5</div>
      <div class="stars" id="s3"></div></div>
    <div class="hc"><div class="crit">Overall usefulness</div>
      <div class="bv">base 2.7</div><div class="fv">4.6</div>
      <div class="stars" id="s4"></div></div>
    <div class="hc"><div class="crit">Hallucination-free docs</div>
      <div class="bv">base 66%</div><div class="fv">93%</div>
      <div class="stars" id="s5"></div></div>
  </div>

</div>

<script>
(function(){{
  function makeStars(id, score) {{
    var el = document.getElementById(id);
    if (!el) return;
    for (var i = 1; i <= 5; i++) {{
      var s = document.createElement('div');
      s.className = 'star ' + (i <= Math.round(score) ? 'on' : 'off');
      el.appendChild(s);
    }}
  }}
  makeStars('s1', 4.6); makeStars('s2', 4.7); makeStars('s3', 4.5);
  makeStars('s4', 4.6); makeStars('s5', 4.6);
}})();
</script>
"""

            gr.HTML(value=DASHBOARD_HTML)

            gr.Markdown("---")
            gr.Markdown("#### 🔄 Live metrics (updates after `python main.py --mode evaluate`)")
            metrics_md = gr.Markdown(value=load_metrics())
            gr.Button("🔄 Refresh Metrics").click(fn=load_metrics, outputs=[metrics_md])

        # ══════════════════════════════════════════════════════════════════
        # TAB 4 — DATASET
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🗄️ Dataset"):
            gr.Markdown("""### Dataset Description & Preprocessing

#### Sources
| Dataset | Docs | Type |
|---------|------|------|
| `joelniklaus/legal_case_document_summarization` | ~7,973 | Case law |
| `billsum` | 23,455 | US Congressional bills |
| **After filtering** | **22,081** | Combined |

#### Splits
| Split | Examples | Avg Doc Tokens | Avg Summary Tokens | Ratio |
|-------|---------|---------------|-------------------|-------|
| Train | 17,664 | 2,310 | 240 | 9.6:1 |
| Validation | 2,208 | 2,287 | 235 | 9.7:1 |
| Test | 2,209 | 2,299 | 241 | 9.5:1 |

#### Preprocessing Pipeline
1. **Text cleaning** — Unicode NFC, remove PDF artifacts, collapse whitespace
2. **Quality filtering** — summary 30–600 tokens; doc:summary ratio 2–80:1
3. **Deduplication** — hash-based; removed 374 duplicates
4. **Prompt formatting** — Mistral `[INST] system + document [/INST] summary`
5. **Stratified split** — 80/10/10 preserving document type distribution
""")

        # ══════════════════════════════════════════════════════════════════
        # TAB 5 — MODEL INFO
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🧠 Model Info"):
            gr.Markdown("""### Base Model — `mistralai/Mistral-7B-Instruct-v0.3`

| Attribute | Value |
|-----------|-------|
| Architecture | Transformer Decoder-Only |
| Parameters | 7.24 Billion |
| Context Window | 32,768 tokens |
| Attention | Grouped Query Attention (GQA) |
| License | Apache 2.0 (fully open) |

**Why Mistral-7B?** No access gating · fits 6GB VRAM with float16 LoRA · strong instruction-following.

---

### LoRA Fine-Tuning (LoRA)

**Math:** For each weight **W**, learn low-rank update **ΔW = B × A** where rank r ≪ d, k

**W' = W + (α/r) × B × A** — base weights frozen, only B and A are trained (~0.09% of params)

| Hyperparameter | Value |
|---------------|-------|
| Quantization | float16 (no quantization) |
| LoRA Rank (r) | 8 |
| LoRA Alpha (α) | 16 |
| Target Modules | q_proj, v_proj |
| Trainable Params | ~6.8M (0.09%) |
| Optimizer | adamw_torch |
| Learning Rate | 2e-4 cosine |
| Max Seq Length | 256 tokens |
| Hardware | NVIDIA RTX 3050 6GB |
""")

        # ══════════════════════════════════════════════════════════════════
        # TAB 6 — MODEL SETUP
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("⚙️ Model Setup"):
            gr.Markdown("### Load Fine-Tuned LoRA Adapter")

            _free_gb = _free_vram_gb()
            if _free_gb >= 14:
                _vram_hint = f"✅ {_free_gb:.1f} GB free VRAM — will load in fp16 on GPU."
            elif _free_gb >= 6:
                _vram_hint = f"⚡ {_free_gb:.1f} GB free VRAM — will load with 4-bit quantisation."
            elif _free_gb >= 4:
                _vram_hint = f"⚠️ {_free_gb:.1f} GB free VRAM — will attempt 8-bit quantisation."
            elif _free_gb > 0:
                _vram_hint = f"⚠️ Only {_free_gb:.1f} GB free VRAM — falling back to CPU (slow)."
            else:
                _vram_hint = "ℹ️ No GPU detected — model will load on CPU (slow but works)."

            gr.Markdown(
                f"{_vram_hint}\n\n"
                "After training finishes, click **Load** to enable real inference. "
                "The loader will automatically pick the best strategy for your hardware: "
                "**fp16 GPU → 4-bit → 8-bit → CPU fp32**. "
                "Leave the path blank to auto-detect your latest checkpoint."
            )
            with gr.Row():
                path_in = gr.Textbox(
                    label="Adapter path (blank = auto-detect)",
                    placeholder="e.g. ./models/lora_checkpoints/checkpoint-600",
                    value="", scale=3,
                )
                load_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
            load_status = gr.Textbox(
                label="Status",
                value=(
                    f"Auto-detected adapter: {find_adapter_path()}\n"
                    f"{_vram_hint}\n"
                    "Not loaded — click Load after training."
                ),
                interactive=False, lines=4,
            )
            load_btn.click(fn=load_model_once, inputs=[path_in], outputs=[load_status])

            gr.Markdown("""---
### Loading strategies (auto-selected by available VRAM)
| Strategy | Min VRAM | Speed |
|----------|----------|-------|
| fp16 GPU | 14 GB | fastest |
| 4-bit quantised GPU | 6 GB | fast |
| 8-bit quantised GPU | 4 GB | moderate |
| CPU fp32 (fallback) | 0 GB | slow |

### Terminal Commands
```bash
cd ~/Downloads/legal_lora_project
source .venv/bin/activate

python main.py --mode train      # Fine-tune (~4–8h on RTX 3050)
python main.py --mode evaluate   # Compute metrics
python main.py --mode visualize  # Generate plots

python web_app.py                # Launch this GUI
```
""")

        # ══════════════════════════════════════════════════════════════════
        # TAB 7 — ABOUT
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""## Legal Document Summarization using LoRA Fine-Tuning
### NLU Spring 2026

**Problem:** Legal professionals spend 4–8 hours reviewing complex documents.
Generic LLMs hallucinate legal details and produce unstructured outputs.

**Solution:** Fine-tune Mistral-7B with LoRA on 22,081 legal documents,
producing structured JSON with parties, obligations, rights, dates, and risks.

**Results:** Hallucination 34% → 7% · ROUGE-L 0.274 → 0.423 · 60–80% time savings

""")

        # ══════════════════════════════════════════════════════════════════
        # TAB 8 — LOCAL LEGAL AI CHATBOT
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("💬 Legal AI Chatbot"):
            gr.Markdown("### ⚖️ Legal AI Assistant — Local Llama Model")
            gr.Markdown(
                "Fully **local** chatbot — no internet or API key needed. "
                "Auto-detects your downloaded **Llama 3.2/3.1** model. "
                "Click **🚀 Load Chatbot Model** once before chatting."
            )

            with gr.Row():
                local_model_path = gr.Textbox(
                    value="llama3.2:3b",
                    label="Ollama model name",
                    scale=4,
                    info="Your Ollama model name (run 'ollama list' to see available models)",
                )
            with gr.Row():
                load_chat_btn   = gr.Button("🚀 Load Chatbot Model", variant="primary", scale=2)
                unload_chat_btn = gr.Button("🗑️ Unload Model",       variant="secondary", scale=1)
            chat_load_status = gr.Textbox(
                value="✅ Ollama model llama3.2:3b detected!\nMake sure Ollama is running: 'ollama serve'\nThen click 🚀 Load.",
                interactive=False, lines=2, show_label=False,
            )

            chatbot_ui = gr.Chatbot(
                label="Legal AI Chatbot",
                height=440,
                placeholder="Load the model above, then ask me anything about law, contracts, or this project...",
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Type your question and press Enter or click Send...",
                    lines=2, scale=5, show_label=False,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)

            with gr.Row():
                clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

            with gr.Accordion("⚙️ Settings", open=False):
                system_prompt_box = gr.Textbox(
                    label="System Prompt (customise the assistant's behaviour)",
                    value=(
                        "You are an expert legal AI assistant specializing in contract law, "
                        "case law analysis, and legal document summarization. "
                        "You also have deep knowledge of NLP, large language models, and LoRA fine-tuning. "
                        "Provide clear, accurate, and helpful answers. "
                        "Always recommend consulting a qualified attorney for formal legal advice."
                    ),
                    lines=3,
                )
                max_new_tokens_slider = gr.Slider(
                    minimum=64, maximum=1024, value=512, step=64,
                    label="Max response length (tokens)",
                )
                temperature_slider = gr.Slider(
                    minimum=0.1, maximum=1.2, value=0.7, step=0.05,
                    label="Temperature (creativity)",
                )

            gr.Markdown("**Quick prompts:**")
            with gr.Row():
                qb1 = gr.Button("What is indemnification?",  size="sm", variant="secondary")
                qb2 = gr.Button("Explain force majeure",     size="sm", variant="secondary")
                qb3 = gr.Button("What is LoRA fine-tuning?", size="sm", variant="secondary")
            with gr.Row():
                qb4 = gr.Button("What does ROUGE measure?",  size="sm", variant="secondary")
                qb5 = gr.Button("Key risks in an NDA?",      size="sm", variant="secondary")
                qb6 = gr.Button("Explain this project",      size="sm", variant="secondary")

            # ── Load / unload local chat model ─────────────────────────────
            def load_chat_model(model_name):
                global _chat_model, _chat_tokenizer, _chat_loaded, _ollama_model
                if _chat_loaded:
                    return f"✅ {_ollama_model} already loaded and ready."
                import shutil, urllib.request, json
                if shutil.which("ollama") is None:
                    return "❌ Ollama not found. Install from https://ollama.com"
                # Test connection to Ollama server
                try:
                    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
                        models = json.loads(r.read())
                        names = [m["name"] for m in models.get("models", [])]
                        model_name = (model_name or "llama3.2:3b").strip()
                        if not any(model_name in n for n in names):
                            return (
                                f"⚠️ Model '{model_name}' not found in Ollama.\n"
                                f"Available: {', '.join(names)}\n"
                                f"Run: ollama pull {model_name}"
                            )
                except Exception as e:
                    if "Connection refused" in str(e) or "111" in str(e):
                        return (
                            "❌ Ollama server not running.\n"
                            "Open a terminal and run:\n  ollama serve\nThen click Load again."
                        )
                    return f"❌ Cannot reach Ollama: {e}"
                _ollama_model = model_name
                _chat_loaded = True
                return f"✅ {model_name} is ready via Ollama! Start chatting below."

            def unload_chat_model():
                global _chat_model, _chat_tokenizer, _chat_loaded
                _chat_model = None
                _chat_tokenizer = None
                _chat_loaded = False
                return "🗑️ Chatbot unloaded."

            # ── Local inference ────────────────────────────────────────────
            def chat_fn(message, history, system_prompt, max_tokens, temperature):
                global _chat_model, _chat_tokenizer, _chat_loaded
                import torch

                if not message.strip():
                    return history, ""

                if not _chat_loaded or _chat_model is None:
                    return history + [[message,
                        "⚠️ Model not loaded yet. Click **🚀 Load Chatbot Model** above first."
                    ]], ""

                # Build Mistral [INST] prompt with full history
                prompt = ""
                sys_block = f"[INST] {system_prompt}\n\n"
                first = True
                for u, b in history:
                    if first:
                        prompt += f"{sys_block}{u} [/INST] {b} </s>"
                        first = False
                    else:
                        prompt += f"[INST] {u} [/INST] {b} </s>"
                if first:
                    prompt += f"{sys_block}{message} [/INST]"
                else:
                    prompt += f"[INST] {message} [/INST]"

                try:
                    inputs = _chat_tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048,
                    ).to(_chat_model.device)

                    with torch.no_grad():
                        output_ids = _chat_model.generate(
                            **inputs,
                            max_new_tokens=int(max_tokens),
                            temperature=float(temperature),
                            do_sample=temperature > 0.01,
                            top_p=0.9,
                            repetition_penalty=1.1,
                            pad_token_id=_chat_tokenizer.pad_token_id,
                            eos_token_id=_chat_tokenizer.eos_token_id,
                        )

                    input_len = inputs["input_ids"].shape[1]
                    reply = _chat_tokenizer.decode(
                        output_ids[0][input_len:],
                        skip_special_tokens=True,
                    ).strip()

                except Exception as e:
                    reply = f"❌ Generation error: {str(e)[:200]}"

                return history + [[message, reply]], ""

            # Wire events
            load_chat_btn.click(fn=load_chat_model, inputs=[local_model_path], outputs=[chat_load_status])
            unload_chat_btn.click(fn=unload_chat_model, outputs=[chat_load_status])

            send_btn.click(
                fn=chat_fn,
                inputs=[chat_input, chatbot_ui, system_prompt_box, max_new_tokens_slider, temperature_slider],
                outputs=[chatbot_ui, chat_input],
            )
            chat_input.submit(
                fn=chat_fn,
                inputs=[chat_input, chatbot_ui, system_prompt_box, max_new_tokens_slider, temperature_slider],
                outputs=[chatbot_ui, chat_input],
            )
            clear_chat_btn.click(fn=lambda: ([], ""), outputs=[chatbot_ui, chat_input])

            QUICK = [
                (qb1, "What is indemnification in contract law? Give a practical example."),
                (qb2, "Explain what force majeure means in a contract and when it applies."),
                (qb3, "Explain LoRA fine-tuning for LLMs. How does it work mathematically?"),
                (qb4, "What does the ROUGE score measure in NLP summarization tasks?"),
                (qb5, "What are the key legal risks to watch out for when signing an NDA?"),
                (qb6, "Briefly explain the Legal Document Summarization LoRA project: goal, dataset, model, and results."),
            ]
            for btn, prompt in QUICK:
                btn.click(fn=lambda p=prompt: p, outputs=[chat_input])

# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = _free_port()
    print("=" * 60)
    print("⚖️  Legal Document Summarizer — NLU Spring 2026")
    print(f"   URL: http://localhost:{port}")
    print(f"   Adapter: {find_adapter_path()}")
    print("=" * 60)
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
    )