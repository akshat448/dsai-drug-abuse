## Setup

1. **Install dependencies:**
   ```sh
   pip install python-docx PyMuPDF requests python-dotenv
   ```

2. **Configure API key:**
   - Add your Gemini API key to `.env`:
     ```
     GEMINI_API_KEY=your_key_here
     ```

3. **Verify directory structure:**
   ```
   data/              # Input documents (.docx, .pdf)
   output/processed/  # Output JSON files (auto-created)
   scripts/           # Pipeline scripts
   ```

## Usage

### Process a Single Document
```sh
python scripts/process_docs.py --file data/YourFile.docx
```

Output saves to `output/processed/YourFile.json`

### Process All Files in `data/` Directory
```sh
for file in data/*.docx data/*.pdf; do
  [ -f "$file" ] && python scripts/process_docs.py --file "$file"
done
```

### Process with Custom Language
```sh
python scripts/process_docs.py --file data/YourFile.docx --language Punjabi
```

## Pipeline Steps

1. **extract_text.py** - Extracts text from .docx/.pdf files
2. **build_prompt.py** - Creates optimized Gemini prompt
3. **call_gemini.py** - Calls Gemini API with retries
4. **save_output.py** - Validates and saves JSON output

## Output Schema

Each output JSON contains:
- `source_file`: Input filename
- `language`: Document language
- `segments`: Array of analyzed segments with:
  - `original_text`: Unmodified transcript
  - `generalized_text`: Anonymized/generalized version
  - `segment_type`: Theme classification (background, triggers, coping, etc.)
  - `tags`: Relevant tags (2–5 per segment)
  - `safety`: Classification (safe, sensitive, unsafe, red_flag)
  - `metadata`: Confidence level and notes