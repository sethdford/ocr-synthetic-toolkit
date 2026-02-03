# OCR Synthetic Document Toolkit

Generate synthetic business documents (invoices, forms, contracts, receipts) with configurable OCR-like errors for training post-OCR correction models.

## What it does

1. **Generates** realistic document content via Claude API
2. **Renders** structured PDFs (4 document types)
3. **Degrades** rendered images (blur, noise, rotation, JPEG artifacts, ink bleed)
4. **Injects** character-level errors using a visual-similarity confusion matrix
5. **Evaluates** with Tesseract OCR and computes CER/WER metrics

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install poppler tesseract  # macOS
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

```bash
# Generate 10 invoices with default error levels
python pipeline.py --doc-type invoice --count 10 --output-dir ./output

# All document types
python pipeline.py --doc-type all --count 5 --error-levels 0.3,1.0,3.0,7.0

# Skip API, use pre-generated content
python pipeline.py --skip-api --input-json content.json --doc-type invoice

# Heavy degradation
python pipeline.py --doc-type invoice --count 5 --blur 1.5 --noise 15 --rotation 1.0 --jpeg-quality 60 --ink-bleed 1.0
```

## Output structure

```
output/
├── invoice_001/
│   ├── clean.pdf
│   ├── ground_truth.txt
│   ├── content.json
│   ├── degraded.png
│   ├── level_0.3/
│   │   ├── degraded.png
│   │   ├── noisy_text.txt
│   │   ├── ocr_output.txt
│   │   └── metrics.json
│   └── level_7.0/...
└── summary.json
```

## Modules

| File | Purpose |
|---|---|
| `confusions.py` | Confusion matrix, error injection, CER/WER metrics |
| `templates.py` | PDF rendering (4 types) + image degradation pipeline |
| `content.py` | Claude API content generation |
| `pipeline.py` | CLI orchestrator |

## Error model

Based on the Guan et al. (EMNLP 2024) approach — OCR errors are modeled as weighted character substitutions from visual similarity (`rn`→`m`, `l`→`1`, `O`→`0`, etc.) with a 5:1:1 substitution:deletion:insertion ratio. Multi-level error injection at configurable CER targets (default: 0.3%, 1%, 3%, 7%, 11%, 15%) produces training pairs across a range of noise levels.
