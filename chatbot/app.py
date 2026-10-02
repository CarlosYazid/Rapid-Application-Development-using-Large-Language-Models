#!/usr/bin/env python3
"""
NVIDIA Course Content Processor
"""

import os
import re
import io
import json
import base64
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Generator

import gradio as gr
import nbformat
import PIL.Image
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 7860))
ROOT_PATH = os.environ.get("ROOT_PATH", None)

BASE_DIR = os.environ.get("NOTEBOOK_DIR", "/notebooks")
IMG_DIR = os.path.join(BASE_DIR, "imgs")
SLIDES_DIR = os.path.join(BASE_DIR, "slides")
SYNTH_DIR = os.environ.get("SYNTH_DIR", "/docs/synth")
DEV_DIR = os.environ.get("DEV_DIR", "/docs/dev")

LLM_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "http://llm_client:9000/v1")
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://lab:9004/v1")
DONE_STR = "✅ **Done**"
ERR_STR = "❌ Error"
GEN_STR = "🤖 Generating Response"
def pulsing(string):
    return f"<span class='thinking'>{string}</span>"

for d in [SYNTH_DIR, DEV_DIR]:
    os.makedirs(d, exist_ok=True)

CHARS_PER_TOKEN = 4
MAX_RECOMMENDED_TOKENS = 32000

# Match the hosted text and local vision services used by the notebooks.
LLM_MODELS = {
    "nvidia/nemotron-3.5-lightning-30b-a3b": "nvidia/nemotron-3.5-lightning-30b-a3b",
}

VLM_MODELS = {
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
}

RECIPE_PRESETS = {
    "Summary": (
        "Provide a comprehensive technical summary. Include overview, key concepts, prerequisites, tools used, code patterns, and applications."
        " Make sure that a professor would be able to perfectly understand what notebooks are available, what they all teach, and accurately"
        " reconstruct the course's narrative from the summary. Do not oversimplify anything, and assume nothing from the reasoning trace will be seen."
        " Make sure to mention every notebook and dredge out its intuitions. Enumerate and explain every notebook."
    ),
    "Quiz": "Create 5 MCQs (4 options each, mark correct, explain) and 2 coding challenges (problem, example, solution).",
    "Study Guide": "Create a study guide: learning objectives, concept relationships, explanations, takeaways, practice suggestions.",
    "Key Points": "Extract key points: main arguments, definitions, critical code, concept relationships. Be concise.",
    "Explain Code": "Explain each code block: what it does, how, why, pitfalls. Assume intermediate Python.",
    "Custom": ""
}

SYSTEM_PRESETS = {
    "Teaching Assistant": "You are an expert teaching assistant for NVIDIA Deep Learning Institute. Provide clear, accurate explanations. Be helpful and educational.",
    "Code Reviewer": "You are a senior code reviewer. Analyze code for correctness, efficiency, style, and issues. Suggest improvements.",
    "Q&A": "Answer questions based on provided context. If unsure, say so. Be concise and accurate.",
    "Socratic": "Guide learning through questions. Don't give direct answers - ask probing questions to help discovery.",
    "Custom": ""
}

# =============================================================================
# UTILITIES
# =============================================================================

def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN if text else 0

def safe_read(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""

def safe_write(path: str, content: str) -> Optional[str]:
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return None
    except Exception as e:
        return str(e)

def ts() -> str:
    return datetime.now().strftime('%H:%M:%S')

def fmt_static(text: str) -> str:
    """Rewrite image paths for static serving."""
    for old, new in [('](imgs/', '](/static/imgs/'), ('](slides/', '](/static/slides/'),
                      ('src="imgs/', 'src="/static/imgs/'), ('src="slides/', 'src="/static/slides/')]:
        text = text.replace(old, new)
    return text

def format_thinking(text: str) -> str:
    """Format <think> tags for display, fixing the dark theme issue."""
    # Convert thinking tags to collapsible details
    text = re.sub(
        r'<think>(.*?)</think>',
        r'<details class="think-box"><summary>💭 Thinking...</summary>\n\n\1\n\n</details>',
        text,
        flags=re.DOTALL
    )
    return text

# =============================================================================
# FILE OPERATIONS
# =============================================================================

def discover_files() -> Dict[str, List[str]]:
    def scan(d: str, exts: tuple) -> List[str]:
        if not os.path.exists(d):
            return []
        return sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith(exts) and not f.startswith('.')])
    
    return {
        "notebooks": scan(BASE_DIR, ('.ipynb', '.md', '.txt', '.py', '.json')),
        "dev": scan(DEV_DIR, ('.ipynb', '.md', '.txt', '.py', '.json')),
        "synth": scan(SYNTH_DIR, ('.md', '.txt', '.json')),
        "images": scan(IMG_DIR, ('.png', '.jpg', '.jpeg', '.gif')) + scan(SLIDES_DIR, ('.png', '.jpg', '.jpeg', '.gif')),
    }

def notebook_to_md(path: str) -> str:
    if not os.path.exists(path):
        return f"[File not found: {path}]"
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        return f"[Parse error: {e}]"
    
    parts = [f"# {os.path.basename(path)}\n---\n"]
    for cell in nb.cells:
        src = cell.source.strip()
        if not src:
            continue
        if cell.cell_type == 'markdown':
            parts.append(f"{src}\n\n")
        elif cell.cell_type == 'code':
            parts.append(f"```python\n{src}\n```\n\n")
    return "".join(parts)

def read_content(path: str, max_chars: Optional[int] = None) -> str:
    content = notebook_to_md(path) if path.endswith('.ipynb') else safe_read(path)
    
    if max_chars and max_chars > 0 and len(content) > max_chars:
        h = max_chars // 2
        content = f"{content[:h]}\n\n[...truncated {len(content)-max_chars:,} chars...]\n\n{content[-h:]}"
        
    return content

# Function to extract all code blocks
def extract_code_blocks(content: str) -> List[str]:
    """Extracts content inside ```python ... ``` blocks."""
    return re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)

def extract_urls(text: str) -> List[str]:
    return list(set(re.findall(r'https?://[^\s<>"\')\]]+[^\s<>"\')\].,;:!?]', text)))

def extract_images(text: str) -> List[Tuple[str, Optional[str]]]:
    refs, seen = [], set()
    for pattern in [r'!\[[^\]]*\]\(([^)]+)\)', r'<img[^>]+src=["\']([^"\']+)["\']']:
        for m in re.finditer(pattern, text):
            p = m.group(1).split('?')[0].strip()
            if p in seen or p.startswith(('data:', 'http')):
                continue
            seen.add(p)
            resolved = None
            for c in [p, os.path.join(BASE_DIR, p), os.path.join(IMG_DIR, os.path.basename(p)), os.path.join(SLIDES_DIR, os.path.basename(p))]:
                if os.path.exists(c):
                    resolved = os.path.abspath(c)
                    break
            refs.append((p, resolved))
    return refs

# =============================================================================
# STREAMING LLM CLIENTS (No Change)
# =============================================================================

class LLM:
    def __init__(self, model: str, base_url: str = LLM_BASE_URL, temp: float = 0.2, max_tokens: int = 4000):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temp = temp
        self.max_tokens = max_tokens
    
    def stream(self, prompt: str, system: str = "") -> Generator[str, None, None]:
        """Stream response chunks. Handles thinking tags."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json={"model": self.model, "messages": msgs, "temperature": self.temp, 
                      "max_tokens": self.max_tokens, "stream": True},
                headers={"Content-Type": "application/json"},
                stream=True, timeout=300
            )
            r.raise_for_status()
            
            for line in r.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            if 'content' in delta and delta['content']:
                                yield delta['content']
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"\n\n❌ Error: {e}"

class VLM(LLM):
    def analyze_image(self, path: str, prompt: str) -> Generator[str, None, None]:
        """Stream image analysis."""
        try:
            with PIL.Image.open(path) as img:
                fmt = 'PNG' if path.lower().endswith('.png') else 'JPEG'
                if img.mode == 'RGBA' and fmt == 'JPEG':
                    img = img.convert('RGB')
                if max(img.size) > 1024:
                    r = 1024 / max(img.size)
                    img = img.resize((int(img.width*r), int(img.height*r)), PIL.Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format=fmt, quality=85)
                b64 = f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"
        except Exception as e:
            yield f"❌ Image error: {e}"
            return
        
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": b64}},
                        {"type": "text", "text": prompt}
                    ]}],
                    "temperature": self.temp, "max_tokens": self.max_tokens, "stream": True
                },
                headers={"Content-Type": "application/json"},
                stream=True, timeout=300
            )
            r.raise_for_status()
            
            for line in r.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: ') and line[6:] != '[DONE]':
                        try:
                            delta = json.loads(line[6:]).get('choices', [{}])[0].get('delta', {})
                            if 'content' in delta and delta['content']:
                                yield delta['content']
                        except:
                            pass
        except Exception as e:
            yield f"\n\n❌ Error: {e}"

# =============================================================================
# PROCESS BOT (streaming to chatbot)
# =============================================================================

def process_bot_stream(sources, ops, recipe, llm_name, vlm_name, max_chars, history):
    """
    Process sources and stream results to chatbot.
    Structured output is streamed using <details> tags.
    """
    history = list(history or [])
    
    if not sources:
        history.append({"role": "assistant", "content": "Please select at least one source file."})
        yield history
        return
    
    # 1. LOAD CONTENT & INITIAL STATUS
    
    # Load and concatenate content
    source_names = [os.path.basename(s) for s in sources]
    combined_parts, all_urls, all_imgs, all_code = [], dict(), dict(), dict()
    
    for src in sources:
        content = read_content(src, int(max_chars) if max_chars and max_chars > 0 else None)
        combined_parts.append(content)
        
        # Extract everything up front for transparency
        all_urls[src] = all_urls.get(src, []) + extract_urls(content)
        all_imgs[src] = all_imgs.get(src, []) + extract_images(content)
        all_code[src] = all_code.get(src, []) + extract_code_blocks(content)

    full_content = "\n\n---\n\n".join(combined_parts)
    tokens = estimate_tokens(full_content) + estimate_tokens(recipe)
    
    # Start status block
    status = f"**Processing:** {', '.join(source_names)}\n"
    status += f"**Operations:** {', '.join(ops) if ops else 'Preview only'}\n"
    status += f"**Tokens:** <span class='stats-value {('warn' if tokens > MAX_RECOMMENDED_TOKENS else '')}'>~{tokens:,}</span>\n\n---\n\n"
    
    # Start streaming results (in assistant's role)
    result_content = ""
    history.append({"role": "assistant", "content": status + pulsing("⏳ Loading sources and metadata...")})
    yield history
    
    # --- PROMPT/CONTEXT TRANSPARENCY (User role) ---
    # Show the user the exact recipe and context used for generation.
    prompt_used = f"<b>RECIPE:</b> {recipe}"
    context_used = f"Context:\n{full_content}"

    # Encapsulate all source content and prompt in a single user message for traceability
    summary_msg = f"Source Context Sent to LLM ({len(sources)} files, {estimate_tokens(full_content):,} tokens)"
    traceability_message = (
        f"**Generation Request**\n---\n{prompt_used}"
        f"\n\n<details><summary>{summary_msg}</summary>"
        f"\n\n{context_used[:5000]}... (Truncated to avoid slowdown)</details>"
    )
    
    # Create a user entry detailing the process input (Recipe + Context)
    history.append({"role": "user", "content": fmt_static(traceability_message)})
    history.append({"role": "assistant", "content": status + GEN_STR})
    yield history

    # 2. EXTRACTIONS (Structure is still built inside the Assistant message)
    
    # A. Source Content (Optional inclusion in final output)
    if "Output Full Content" in ops:
        result_content += f"<details><summary>📝 Full Source Content (Sent as input)</summary>\n\n{full_content}\n\n</details>\n\n"
        history[-1]["content"] = status + result_content + GEN_STR
        yield history

    # B. Code Extraction
    if "Output Code" in ops and all_code:
        code_block = "\n\n".join(key + "\n" + "\n".join(f"```python\n{entry}\n```" for entry in values) for key, values in all_code.items())
        num_code = sum(len(v) for k, v in all_code.items())
        result_content += f"<details><summary>📄 Extracted Code ({num_code} block(s))</summary>\n\n{code_block}\n\n</details>\n\n"
        history[-1]["content"] = status + result_content + GEN_STR
        yield history
        
    # C. URL Extraction
    if "Output URLs" in ops and all_urls:
        url_list = "\n\n".join(key + "\n" + "\n".join(f" * {entry}" for entry in values) for key, values in all_urls.items())
        num_urls = sum(len(v) for k, v in all_urls.items())
        result_content += f"<details><summary>🔗 Extracted URLs ({num_urls} unique)</summary>\n\n{url_list}\n\n</details>\n\n"
        history[-1]["content"] = status + result_content + GEN_STR
        yield history
        
    # D. Image Extraction (Metadata only)
    if "Output Images" in ops and all_imgs:
        img_list = "\n\n".join(key + "\n" + "\n".join(
            f"* {orig} ({'Resolved' if resolved else 'Not Found'})" 
            for (orig, resolved) in values) for key, values in all_imgs.items()
        )
        num_imgs = sum(len(v) for k, v in all_imgs.items())
        result_content += f"<details><summary>🖼️ Extracted Images/Slides ({num_imgs} refs)</summary>\n\n{img_list}\n\n</details>\n\n"
        history[-1]["content"] = status + result_content + GEN_STR
        yield history

    # 3. LLM GENERATION
    
    llm_output_placeholder = GEN_STR
    
    if recipe.strip() and "Generate" in ops:
        result_content += f"## 🤖 Generated Content\n\n"
        history[-1]["content"] = status + result_content + llm_output_placeholder
        yield history
        
        model_id = LLM_MODELS.get(llm_name, llm_name)
        llm = LLM(model=model_id, base_url=LLM_BASE_URL, max_tokens=4096)
        
        # We send the combined prompt + context to the LLM stream
        final_prompt_input = f"{prompt_used}\n\n---\n\n{context_used}"
        
        llm_output = ""
        for chunk in llm.stream(final_prompt_input):
            llm_output += chunk
            # Apply formatting dynamically (thinking tags)
            display = result_content + format_thinking(llm_output)
            history[-1]["content"] = status + display
            yield history
            
        result_content += format_thinking(llm_output)
        history[-1]["content"] = status + result_content
        yield history
    else:
        # If no generation, the final output is just the extracted/preview content
        history.pop() # Remove the "Generating response..." assistant message
        history[-1]["content"] = status + result_content

    # 4. VLM IMAGE ANALYSIS
    
    if "Analyze Images" in ops and all_imgs:
        resolved = [(o, r) for o, r in all_imgs if r][:5]
        if resolved:
            result_content += "\n\n## 🖼️ Visual Analysis\n\n"
            history[-1]["content"] = status + result_content
            yield history
            
            vlm_id = VLM_MODELS.get(vlm_name, vlm_name)
            vlm = VLM(model=vlm_id, base_url=VLM_BASE_URL, max_tokens=1024)
            
            for orig, rpath in resolved:
                name = os.path.basename(rpath)
                result_content += f"### <img src='{rpath.replace('notebooks/', 'static/')}'></img>\n"
                llm_analysis = ""
                
                for chunk in vlm.analyze_image(rpath, "Describe this figure in technical detail."):
                    llm_analysis += chunk
                    history[-1]["content"] = status + result_content + llm_analysis
                    yield history
                
                result_content += llm_analysis + "\n\n"
                history[-1]["content"] = status + result_content
                yield history
    
    # 5. FINALIZATION
    # Get the LAST assistant message (which holds the final generated content)
    final_output = history[-1]["content"]
    
    # Ensure the FINAL content is formatted correctly and marked as done.
    history[-1]["content"] = fmt_static(final_output) + "\n\n" + DONE_STR
    yield history

# =============================================================================
# CHAT BOT (streaming)
# =============================================================================

def chat_bot_stream(message, history, context, system, temp, max_tokens, model):
    """Stream chat response, enhancing multi-turn conversation context."""
    if not message.strip():
        yield history
        return
    
    history = list(history or [])
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    yield history
    
    model_id = LLM_MODELS.get(model, model)
    llm = LLM(model=model_id, base_url=LLM_BASE_URL, temp=temp, max_tokens=max_tokens)
    
    # Build the list of messages for the API call
    api_messages = []
    
    # Add System Prompt
    if system.strip():
        api_messages.append({"role": "system", "content": system})
        
    # Add History for multi-turn (excluding the current user/assistant turn)
    MAX_HISTORY_MESSAGES = 4 
    
    # Use only the content of the history items
    for m in history[-(MAX_HISTORY_MESSAGES + 1):-1]: 
        api_messages.append({"role": m['role'], "content": m['content'][:1024]}) # Truncate history messages
        
    # Build the user prompt (Context is ONLY sent in the first user message)
    user_prompt = message
    if context.strip():
        # Prepend the context to the current question
        # NOTE: The context is a raw string from raw_context_store (not HTML)
        user_prompt = f"CONTEXT:\n{context}\n\n---\n\nQUESTION: {message}"

    api_messages.append({"role": "user", "content": user_prompt})

    # Find the system message, if any
    system_msg_content = api_messages[0]['content'] if api_messages and api_messages[0]['role'] == 'system' else ""
    
    # 2. Stream Response
    response = ""
    history[-1]["content"] = pulsing("🤖 Thinking…")
    yield history
    for chunk in llm.stream(api_messages[-1]['content'], system=system_msg_content):
        if chunk:
            response += chunk
            history[-1]["content"] = format_thinking(response)
            yield history

    # 3. Final cleanup and image path fixing
    history[-1]["content"] = fmt_static(format_thinking(response))
    yield history

# =============================================================================
# INTERFACE
# =============================================================================

# Updated CSS remains the same for better visual appeal

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap');

* { 
    font-family: 'Inter', system-ui, sans-serif !important; 
    color: #333; /* Default body text color for light theme */
}

/* Dark mode compatibility fixes for text and containers */
body { 
    background-color: #f7f7f7; /* Light background */
}

.dark * {
    color: #ddd !important; /* General text color in dark theme */
}

code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

/* Token/Stats Box */
.stats {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px;
    padding: 6px 10px;
    background: #e9ecef;
    border-left: 3px solid #76b900;
    border-radius: 4px;
    color: #333;
}
.dark .stats {
    background: #1a1a2e;
    color: #76b900;
}
.stats-value { font-weight: bold; }
.warn { color: #ff6b6b !important; }

/* Collapsible Details/Thinking Box */
details summary { cursor: pointer; color: #666; font-style: italic; }
details { 
    border: 1px solid #ddd; 
    padding: 8px; 
    border-radius: 4px; 
    margin: 8px 0;
    background: #fff;
}
.dark details {
    background: #1a1a2e; /* Dark background for details */
    border-color: #76b90044;
    color: #ddd !important;
}
.dark details summary {
    color: #aaa !important;
}
.think-box pre {
    white-space: pre-wrap;
    background: #f0f0f0; /* Light code block background */
    padding: 8px;
    border-radius: 3px;
    border: 1px solid #ccc;
    color: #333;
}
.dark .think-box pre {
    background: #2d2d3d; /* Dark code block background */
    color: #ddd;
    border-color: #444;
}
.thinking {
    font-weight: 600;
    animation: pulse 1.2s infinite;
}

@keyframes pulse {
    0% { opacity: 0.4; }
    50% { opacity: 1; }
    100% { opacity: 0.4; }
}
"""


def build_interface() -> gr.Blocks:
    files = discover_files()
    all_sources = files['notebooks'] + files['synth'] + files["dev"]
    
    with gr.Blocks(title="Content Processor") as app:
        
        gr.Markdown("# 🚀 DLI Content Processor [BETA]")
        raw_context_store = gr.State("")

        with gr.Tabs():
            # ═══════════════════════════════════════════════════════════════
            # TAB 1: CHAT
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### System & Settings")
                        
                        ctx_preset = gr.Dropdown(
                            choices=list(SYSTEM_PRESETS.keys()),
                            value="Teaching Assistant", label="System Preset"
                        )
                        system_prompt = gr.Textbox(
                            value=SYSTEM_PRESETS["Teaching Assistant"],
                            label="System Prompt (LLM's Persona)", lines=2, show_label=True
                        )
                        chat_model = gr.Dropdown(
                            choices=list(LLM_MODELS.keys()),
                            value=list(LLM_MODELS.keys())[0], label="Model",
                            allow_custom_value=True,
                        )
                        with gr.Row():
                            chat_temp = gr.Slider(0, 1, value=0.3, step=0.05, label="Temperature")
                            max_tokens = gr.Slider(1, 10000, value=4000, step=1, label="Max Tokens")
                        
                        with gr.Row():
                            gr.Markdown("#### File Context")
                            toggle_select1 = gr.Button("🔃 Flip Selects", size="sm", scale=1)
                            refresh_btn1 = gr.Button("🔄 Refresh", size="sm", scale=1)
                        ctx_files = gr.Dropdown(
                            choices=[(os.path.basename(f), f) for f in all_sources],
                            multiselect=True, label="Source Files",
                        )
                        ctx_max = gr.Slider(0, 30000, value=15000, step=1000, label="Max Chars/File (for LLM)", visible="hidden")

                        with gr.Row():
                            load_btn = gr.Button("📁 Load Context", variant="secondary", size="sm")
                            clear_ctx_btn = gr.Button("🗑️ Empty Context", size="sm")

                        context_box = gr.Textbox(
                            label="Context (Loaded & Editable)", lines=6,
                            placeholder="Loaded file content will appear here...",
                            interactive=True # Ensure user can edit the loaded context
                        )
                        ctx_stats = gr.Markdown("", elem_classes=["stats"])
                    
                    with gr.Column(scale=2):
                        chat_bot = gr.Chatbot(
                            value=[{"role": "assistant", "content": "Hello! Load some context or just ask a question. I'll stream my response."}],
                            height="75vh",
                        )
                        with gr.Row():
                            chat_input = gr.Textbox(placeholder="Ask anything...", show_label=False, scale=5)
                            chat_send = gr.Button("Send", variant="primary", scale=1)
                        chat_clear = gr.Button("Clear Chat", size="sm")
            
            # ═══════════════════════════════════════════════════════════════
            # TAB 2: PROCESS
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("⚡ Process"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            gr.Markdown("#### Sources & Configuration")
                            toggle_select2 = gr.Button("🔃 Flip Selects", size="sm", scale=1)
                            refresh_btn2 = gr.Button("🔄 Refresh", size="sm", scale=1)
                        with gr.Row():
                            source_select = gr.Dropdown(
                                choices=[(os.path.basename(f), f) for f in all_sources],
                                multiselect=True, label="Source Files"
                            )
                            max_chars = gr.Number(value=15000, label="Max Chars/File", minimum=0, visible="hidden")
                        
                        gr.Markdown("#### Recipe & Operations")
                        preset = gr.Dropdown(
                            choices=list(RECIPE_PRESETS.keys()),
                            value="Summary", label="Generation Preset"
                        )
                        recipe = gr.Textbox(
                            value=RECIPE_PRESETS["Summary"],
                            label="Custom Instruction", lines=4, show_label=True
                        )
                        ops = gr.CheckboxGroup(
                            choices=["Generate", "Output URLs", "Output Images", "Output Code", "Analyze Images", "Output Full Content"],
                            value=["Generate"], label="Data Extractions"
                        )
                        
                        with gr.Row():
                            proc_llm = gr.Dropdown(choices=list(LLM_MODELS.keys()), value=list(LLM_MODELS.keys())[0], 
                                label="LLM", scale=1, allow_custom_value=True)
                            proc_vlm = gr.Dropdown(choices=list(VLM_MODELS.keys()), value=list(VLM_MODELS.keys())[0], 
                                label="VLM", scale=1, allow_custom_value=True)
                        
                        token_est = gr.Markdown("Tokens: 0", elem_classes=["stats"])
                        process_btn = gr.Button("🚀 Process Files", variant="primary", size="lg")
                        busy = gr.Markdown(pulsing("⏳Processing…"), visible=False)

                        gr.Markdown("#### 💾 Save")
                        with gr.Row():
                            save_name = gr.Textbox(placeholder="output.md", show_label=False, scale=3)
                            save_btn = gr.Button("Save Output", variant="secondary", scale=1)
                        save_status = gr.Markdown("")
                    
                    with gr.Column(scale=2):
                        proc_bot = gr.Chatbot(
                            value=[{"role": "assistant", "content": "Select sources, choose a recipe, and click Process. I'll stream the structured results here."}],
                            height="75vh",
                        )
                        proc_clear = gr.Button("Clear Output", size="sm")
        
        # ═══════════════════════════════════════════════════════════════
        # HANDLERS (Binding logic)
        # ═══════════════════════════════════════════════════════════════
        
        # Presets
        ctx_preset.change(lambda p: SYSTEM_PRESETS.get(p, ""), inputs=[ctx_preset], outputs=[system_prompt])
        preset.change(lambda p: RECIPE_PRESETS.get(p, ""), inputs=[preset], outputs=[recipe])
        
        # Context loading
        def update_ctx_stats(ctx):
            # Only update stats based on the raw content inside the details tag
            if not ctx: return ""
            # Safely extract content from the HTML context box, stripping the wrapper
            raw_content = re.sub(r'<details.*?<summary>.*?</summary>\s*(.*?)\s*</details>', r'\1', ctx, flags=re.DOTALL | re.MULTILINE)

            t = estimate_tokens(raw_content)
            class_cat = 'warn' if t > MAX_RECOMMENDED_TOKENS else ''
            return f"<span class='{class_cat} stats-value'>~{t:,} tokens</span>"

        def load_context_and_store(file_paths, mc=0):
            if not file_paths: return "", "", ""
            mc = int(mc) if mc else None
            
            raw_parts = []
            for f in file_paths:
                raw_parts.append(read_content(f, mc))
            raw_ctx = "\n\n---\n\n".join(raw_parts)

            t = estimate_tokens(raw_ctx)
            class_cat = 'warn' if t > MAX_RECOMMENDED_TOKENS else ''
            display_ctx = f"<details open><summary>Loaded Context ({len(file_paths)} files, {t:,} tokens)</summary>\n\n{raw_ctx}\n\n</details>"
            
            return display_ctx, f"<span class='{class_cat} stats-value'>~{t:,} tokens</span>", raw_ctx
        
        # Bindings
        load_btn.click(load_context_and_store, inputs=[ctx_files, ctx_max], outputs=[context_box, ctx_stats, raw_context_store])
        clear_ctx_btn.click(lambda: ("", "", ""), outputs=[context_box, ctx_stats, raw_context_store])
        context_box.change(update_ctx_stats, inputs=[context_box], outputs=[ctx_stats]) 
        
        # Token estimation for process
        def est_tokens(sources, recipe_text, mc=0):
            if not sources:
                return "Tokens: 0"
            mc = int(mc) if mc else None
            total = sum(estimate_tokens(read_content(s, mc)) for s in sources)
            total += estimate_tokens(recipe_text or "")
            class_cat = 'warn' if total > MAX_RECOMMENDED_TOKENS else ''
            return f"Tokens: <span class='{class_cat} stats-value'>~{total:,}</span>"
        
        source_select.change(est_tokens, inputs=[source_select, recipe, max_chars], outputs=[token_est])
        recipe.change(est_tokens, inputs=[source_select, recipe, max_chars], outputs=[token_est])
        max_chars.change(est_tokens, inputs=[source_select, recipe, max_chars], outputs=[token_est])
        
        def get_file_choices():
            f = discover_files()
            all_f = f['notebooks'] + f['synth'] + f['dev']
            choices = [(os.path.basename(x), x) for x in all_f]
            return choices

        # Refresh files
        def do_refresh():
            choices = get_file_choices()
            return gr.update(choices=choices), gr.update(choices=choices)
        
        def do_flip(x, y):
            choices = get_file_choices()
            return (
                gr.update(choices=choices, value=[v[1] for v in get_file_choices() if v[1] not in x]), 
                gr.update(choices=choices)
            )

        refresh_btn1.click(do_refresh, outputs=[source_select, ctx_files])
        refresh_btn2.click(do_refresh, outputs=[source_select, ctx_files])
        toggle_select1.click(do_flip, inputs=[ctx_files, source_select], outputs=[ctx_files, source_select])
        toggle_select2.click(do_flip, inputs=[source_select, ctx_files], outputs=[source_select, ctx_files])
        
        # Chat handlers
        chat_send.click(
            chat_bot_stream,
            inputs=[chat_input, chat_bot, raw_context_store, system_prompt, chat_temp, max_tokens, chat_model],
            outputs=[chat_bot]
        ).then(lambda: "", outputs=[chat_input])
        
        chat_input.submit(
            chat_bot_stream,
            inputs=[chat_input, chat_bot, raw_context_store, system_prompt, chat_temp, max_tokens, chat_model],
            outputs=[chat_bot]
        ).then(lambda: "", outputs=[chat_input])
        
        chat_clear.click(
            lambda: [{"role": "assistant", "content": "Chat cleared."}],
            outputs=[chat_bot]
        )
        
        # Process handlers
        process_btn.click(
            lambda: gr.update(visible=True),
            outputs=[busy],
        ).then(
            process_bot_stream,
            inputs=[source_select, ops, recipe, proc_llm, proc_vlm, max_chars, proc_bot],
            outputs=[proc_bot],
        ).then(
            lambda: gr.update(visible=False),
            outputs=[busy],
        )

        process_btn.click(
            lambda: gr.update(interactive=False),
            outputs=[process_btn],
        ).then(
            process_bot_stream,
            inputs=[source_select, ops, recipe, proc_llm, proc_vlm, max_chars, proc_bot],
            outputs=[proc_bot],
        ).then(
            lambda: gr.update(interactive=True),
            outputs=[process_btn],
        )
        
        proc_clear.click(
            lambda: [{"role": "assistant", "content": "Ready to process."}],
            outputs=[proc_bot]
        )
        
        # Save (extracts last assistant message) - FIXED
        def do_save(history, filename):
            if not history: return "Nothing to save"
            
            content = ""
            # 1. Iterate backwards to find the LAST Assistant message.
            for msg in reversed(history):
                # 2. Extract the content string.
                content = msg.get("content", "")
                while not isinstance(content, str):
                    if isinstance(content, list):
                        content = content[-1]
                    elif isinstance(content, dict):
                        content = content.get("text", list(content.values())[-1])
                    else:
                        content = "No Response"
                
                # 3. Clean up HTML tags and status wrappers from the saved output.
                # Remove final status markers
                content = content.replace(DONE_STR, "").strip()
                # Remove initial status block and generation placeholder
                content = content.replace(pulsing("⏳ Loading sources and metadata..."), "").strip()
                break
            
            if not content or content.startswith("Ready") or content.startswith("Select"):
                return "Nothing to save"
            
            fn = filename.strip() or f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            if not fn.endswith('.md'):
                fn += '.md'
            
            path = os.path.join(SYNTH_DIR, fn)
            err = safe_write(path, content)
            return f"❌ {err}" if err else f"✅ Saved: {fn}"
        
        save_btn.click(do_save, inputs=[proc_bot, save_name], outputs=[save_status])
            
    return app

# =============================================================================
# FASTAPI & SERVER
# =============================================================================

def create_app() -> FastAPI:
    app = FastAPI(title="Content Processor")
    
    for name, path in [("imgs", IMG_DIR), ("slides", SLIDES_DIR)]:
        if os.path.exists(path):
            app.mount(f"/static/{name}", StaticFiles(directory=path), name=f"static-{name}")
    
    interface = build_interface().queue()
    app = gr.mount_gradio_app(
        app, interface, path="/",
        root_path=ROOT_PATH,
        theme=gr.themes.Soft(primary_hue="green"),
        css=CSS
    )
    return app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--standalone", action="store_true")
    args = parser.parse_args()
    
    logger.info(f"Starting on port {args.port}")
    
    if args.standalone:
        build_interface().launch(server_name=args.host, server_port=args.port)
    else:
        uvicorn.run(create_app(), host=args.host, port=args.port)
