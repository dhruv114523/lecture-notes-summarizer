"""Notes Q&A agent: ask questions about your own lecture notes.

Supported files: PDF, Word (.docx), PowerPoint (.pptx), plain text (.txt) and Markdown (.md).
Everything runs locally. The model never opens files itself: this script extracts the
text, and the model only sees what its tools hand back.

Setup (once):
    pip install ollama pypdf python-docx python-pptx
    ollama pull qwen3.5:4b

Run:
    python notes_agent.py "C:/path/to/your/notes"
    python notes_agent.py "C:/path/to/your/notes" --model qwen3.5:9b
    (or just `python notes_agent.py` to use a folder called ./notes)

Then ask things like:  what did I write about photosynthesis?
Type 'quit' to exit.
"""
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import ollama

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
try:
    from docx import Document
except ImportError:
    Document = None
try:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
except ImportError:
    Presentation = None
    MSO_SHAPE_TYPE = None

MODEL = "qwen3.5:4b"
MAX_STEPS = 8
CHUNK_CHARS = 2500
MAX_RESULTS = 6
SNIPPET_CHARS = 220
READ_LIMIT = 3000
MAX_FILE_MB = 150
KEEP_QUESTIONS = 6
NUM_CTX = 8192

SUPPORTED = {".pdf", ".docx", ".pptx", ".txt", ".md"}
LEGACY = {".doc": "Word (.docx)", ".ppt": "PowerPoint (.pptx)"}


@dataclass
class Doc:
    parts: list
    unit: str


DOCS: dict = {}

# Extractors: turn each file type into a list of text parts

def split_text(text: str) -> list:
    """Split long text into chunks, breaking at paragraph boundaries."""
    parts, current = [], ""
    for para in text.split("\n\n"):
        if current and len(current) + len(para) > CHUNK_CHARS:
            parts.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts or [""]


def extract_pdf(path: Path):
    if PdfReader is None:
        raise RuntimeError("pypdf not installed (pip install pypdf)")
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            ok = reader.decrypt("")
        except Exception:
            ok = 0
        if not ok:
            raise ValueError("password-protected PDF")
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    if not any(p.strip() for p in parts):
        raise ValueError("no extractable text (scanned PDF? it needs OCR first)")
    return parts, "page"


def _table_text(table) -> str:
    rows = []
    for row in table.rows:
        cells, last = [], None
        for cell in row.cells:
            t = " ".join(cell.text.split())
            if t != last:
                cells.append(t)
            last = t
        line = " | ".join(c for c in cells if c)
        if line:
            rows.append(line)
    return "\n".join(rows)


def extract_docx(path: Path):
    if Document is None:
        raise RuntimeError("python-docx not installed (pip install python-docx)")
    doc = Document(str(path))
    try:
        items = list(doc.iter_inner_content())
    except AttributeError:
        items = list(doc.paragraphs) + list(doc.tables)
    blocks = []
    for item in items:
        try:
            if hasattr(item, "rows"):
                text = _table_text(item)
            else:
                text = item.text.strip()
            if text:
                blocks.append(text)
        except Exception:
            continue
    if not blocks:
        raise ValueError("no text found in document")
    return split_text("\n\n".join(blocks)), "part"


def _collect_shape_text(shapes, out: list) -> None:
    for shape in shapes:
        try:
            if MSO_SHAPE_TYPE is not None and shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                _collect_shape_text(shape.shapes, out)
            elif getattr(shape, "has_table", False) and shape.has_table:
                text = _table_text(shape.table)
                if text:
                    out.append(text)
            elif shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    out.append(text)
        except Exception:
            continue


def extract_pptx(path: Path):
    if Presentation is None:
        raise RuntimeError("python-pptx not installed (pip install python-pptx)")
    prs = Presentation(str(path))
    parts = []
    for slide in prs.slides:
        out = []
        _collect_shape_text(slide.shapes, out)
        try:
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    out.append("Speaker notes: " + notes)
        except Exception:
            pass
        parts.append("\n".join(out))
    if not any(p.strip() for p in parts):
        raise ValueError("no text found (slides may be images only)")
    return parts, "slide"


def extract_plain(path: Path):
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        text = raw.decode("utf-8", errors="ignore")
    if not text.strip():
        raise ValueError("file is empty")
    return split_text(text), "part"


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".pptx": extract_pptx,
    ".txt": extract_plain,
    ".md": extract_plain,
}


def load_notes(folder: Path) -> None:
    """Read every supported file in the folder (and subfolders). Never crashes on a bad file."""
    skipped = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        name = str(path.relative_to(folder))
        suffix = path.suffix.lower()
        if path.name.startswith(("~$", ".")):
            continue
        if suffix in LEGACY:
            skipped.append((name, f"old format; open it and Save As {LEGACY[suffix]}"))
            continue
        if suffix not in SUPPORTED:
            continue
        try:
            if path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
                raise ValueError(f"file larger than {MAX_FILE_MB} MB")
            parts, unit = EXTRACTORS[suffix](path)
            DOCS[name] = Doc(parts, unit)
            print(f"  loaded {name} ({len(parts)} {unit}{'s' if len(parts) != 1 else ''})")
        except Exception as e:
            reason = str(e) or type(e).__name__
            if suffix in {".docx", ".pptx"} and "not found" in reason.lower():
                reason = "could not open (corrupt, or not a real Office file)"
            skipped.append((name, reason))

    if skipped:
        print("\nSkipped files:")
        for name, reason in skipped:
            print(f"  ! {name}: {reason}")


# Tools the model can call. All read-only, and limited to files loaded above.

def _resolve(filename) -> str:
    """Forgiving file lookup: small models often get names slightly wrong."""
    filename = str(filename).strip().strip("\"'")
    if filename in DOCS:
        return filename
    low = filename.lower()
    exact = [n for n in DOCS if n.lower() == low or Path(n).name.lower() == low]
    if len(exact) == 1:
        return exact[0]
    partial = [n for n in DOCS if low and low in n.lower()]
    return partial[0] if len(partial) == 1 else ""


def _stem(word: str) -> str:
    """Crude stemming so 'photosynthesis' also matches 'photosynthetic'."""
    return word if len(word) <= 5 else word[: max(5, len(word) - 3)]


def list_notes() -> str:
    """List all available note files and how many pages/slides/parts each has."""
    if not DOCS:
        return "No notes loaded."
    return "\n".join(
        f"{n} ({len(d.parts)} {d.unit}{'s' if len(d.parts) != 1 else ''})" for n, d in DOCS.items()
    )


def search_notes(keywords: str) -> str:
    """Search all notes for keywords and return the best matching pages/slides with a short snippet.

    Args:
        keywords: One or more keywords to look for, e.g. "photosynthesis light reactions".
    """
    if isinstance(keywords, (list, tuple)):
        keywords = " ".join(map(str, keywords))
    words = [w for w in re.findall(r"\w+", str(keywords).lower()) if len(w) > 2]
    if not words:
        return "Please give at least one keyword of 3+ letters."
    stems = [_stem(w) for w in words]

    hits = []
    for name, doc in DOCS.items():
        for i, text in enumerate(doc.parts, start=1):
            lower = text.lower()
            matched = [s for s in stems if s in lower]
            if not matched:
                continue
            score = sum(lower.count(s) for s in matched)
            first = min(lower.find(s) for s in matched)
            start = max(0, first - SNIPPET_CHARS // 2)
            snippet = " ".join(text[start:start + SNIPPET_CHARS].split())
            hits.append((len(matched), score, name, doc.unit, i, snippet))

    if not hits:
        return f"No matches for: {keywords}. Try different or simpler keywords."

    hits.sort(key=lambda h: (h[0], h[1]), reverse=True)
    return "\n".join(
        f"[{name}, {unit} {i}] ...{snippet}..." for _, _, name, unit, i, snippet in hits[:MAX_RESULTS]
    )


def read_note(filename: str, page: int = 1) -> str:
    """Read one page, slide or part of a note file in full. Use after search_notes.

    Args:
        filename: File name as shown by list_notes or search_notes.
        page: Page, slide or part number, starting at 1.
    """
    name = _resolve(filename)
    if not name:
        return f"No such file: {filename}. Use list_notes to see available files."
    doc = DOCS[name]
    try:
        page = int(page)
    except (TypeError, ValueError):
        return f"Page must be a number from 1 to {len(doc.parts)}."
    if not 1 <= page <= len(doc.parts):
        return f"{name} has {len(doc.parts)} {doc.unit}s. Choose 1 to {len(doc.parts)}."
    text = doc.parts[page - 1].strip() or f"(this {doc.unit} has no text)"
    return f"[{name}, {doc.unit} {page} of {len(doc.parts)}]\n{text[:READ_LIMIT]}"


TOOLS = {"list_notes": list_notes, "search_notes": search_notes, "read_note": read_note}

SYSTEM_PROMPT = (
    "You answer questions about the user's personal study notes. "
    "Always use search_notes first, then read_note on the most relevant page or slide if you need more detail. "
    "Answer ONLY from what the tools return. Do not use outside knowledge. "
    "Cite the file name and page/slide number for each claim, like (biology.pdf, page 4). "
    "If the notes do not contain the answer, say so plainly instead of guessing."
)

# The agent loop

def trim_history(messages: list) -> None:
    """Keep only the last few questions so a small model's context doesn't overflow."""
    user_idx = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"]
    if len(user_idx) > KEEP_QUESTIONS:
        del messages[1:user_idx[-KEEP_QUESTIONS]]


def answer(messages: list) -> str:
    for step in range(1, MAX_STEPS + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=list(TOOLS.values()),
                options={"num_ctx": NUM_CTX},
            )
        except ollama.ResponseError as e:
            return f"Ollama returned an error: {e}. Check the model name with `ollama list`."
        except Exception as e:
            return f"Could not reach Ollama ({e}). Is the Ollama app running?"

        msg = response.message
        messages.append(msg)

        if not msg.tool_calls:
            text = (msg.content or "").strip() or "(The model returned an empty answer. Try rephrasing.)"
            if getattr(response, "done_reason", None) == "length":
                text += ("\n\n[Warning: the answer was cut off because the context window filled up. "
                         "Raise NUM_CTX at the top of the script, or ask a narrower question.]")
            return text

        for call in msg.tool_calls:
            name, args = call.function.name, call.function.arguments
            print(f"  [step {step}] {name}({args})")
            try:
                result = TOOLS[name](**args) if name in TOOLS else f"Unknown tool: {name}"
            except Exception as e:
                result = f"Tool error in {name}: {e}"
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})

    return "I hit the step limit without finding a clear answer. Try rephrasing your question."


def check_ollama() -> None:
    try:
        installed = ollama.list().models
    except Exception as e:
        sys.exit(f"Cannot reach Ollama ({e}).\nStart the Ollama app (or run `ollama serve`) and try again.")
    names = [getattr(m, "model", None) or getattr(m, "name", "") or "" for m in installed]
    if MODEL not in names and f"{MODEL}:latest" not in names:
        print(f"Warning: model '{MODEL}' not found. Installed: {', '.join(names) or 'none'}")
        print(f"Fix: run `ollama pull {MODEL}` or start with --model <name>\n")


def main() -> None:
    global MODEL
    parser = argparse.ArgumentParser(description="Ask questions about your notes.")
    parser.add_argument("folder", nargs="?", default="notes", help="folder containing your notes")
    parser.add_argument("--model", default=MODEL, help=f"Ollama model name (default: {MODEL})")
    args = parser.parse_args()
    MODEL = args.model
    folder = Path(args.folder).expanduser()

    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")

    check_ollama()
    print(f"Loading notes from {folder} ...")
    load_notes(folder)
    if not DOCS:
        sys.exit("\nNo readable .pdf, .docx, .pptx, .txt or .md files found.")

    print(f"\nReady with {len(DOCS)} file(s). Model: {MODEL}. Type 'quit' to exit.\n")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        messages.append({"role": "user", "content": question})
        trim_history(messages)
        print("\nAgent:", answer(messages), "\n")


if __name__ == "__main__":
    main()