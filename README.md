# Multi-LLM Anki Deck Generator v0

Automagically creates Anki vocabulary decks (`.apkg`) from a word list using:
*   LLM #1 for source example sentences.
*   LLM #2 for translation (word + sentence).
*   TTS (XTTSv2) for source & target audio.
*   SQLite DB for **resumable** progress.

**Requires:** Python, PyTorch, `transformers`, `TTS`, `genanki`, `icecream`, `espeak-ng`.

**Run:**
```bash
# Basic:
python anki_deck_generator.py your_words.txt --lang-src de --lang-target en

# Options (--help for more):
python anki_deck_generator.py words.txt --llm-model-example-gen <model_id> --llm-model-translate <model_id> --tts-speaker <name> --use-quantization