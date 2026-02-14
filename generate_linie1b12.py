import asyncio
import hashlib
import re
from pathlib import Path
from typing import List, Tuple

import edge_tts
import genanki

# --- Configuration ---
INPUT_DIR = Path("linie1B12")
OUTPUT_BASE_DIR = Path("output_linie1b12")
TTS_VOICE = "de-DE-KatjaNeural"  # or de-DE-ConradNeural, de-DE-AmalaNeural, de-DE-KillianNeural

# Chapter names for each file (Kapitel 9-16)
CHAPTER_TOPICS = {
    "09": "9 – Arbeit und Beruf",
    "10": "10 – Sport und Fitness",
    "11": "11 – Familie und Elternzeit",
    "12": "12 – Bildung und Anerkennung",
    "13": "13 – Ehrenamt und Engagement",
    "14": "14 – Wohnen und Nachbarschaft",
    "15": "15 – Integration und Zusammenleben",
    "16": "16 – Prüfungen und Lernen",
}

MODEL_ID = 1774089500
PARENT_DECK_NAME = "Linie1 B1.2"

ANKI_FIELDS = [
    {"name": "RussianWord"},
    {"name": "RussianExample"},
    {"name": "GermanWord"},
    {"name": "GermanExample"},
    {"name": "GermanAudio"},
]

ANKI_TEMPLATES = [
    {
        "name": "RU -> DE",
        "qfmt": (
            '<div style="font-family: Arial; font-size: 30px; text-align: center;">{{RussianWord}}</div>'
            "<hr>"
            '<div style="font-family: Arial; font-size: 20px; text-align: center; color:grey;">{{RussianExample}}</div>'
        ),
        "afmt": (
            "{{FrontSide}}"
            '<hr id=answer>'
            '<div style="font-family: Arial; font-size: 30px; text-align: center;">{{GermanWord}}</div>'
            "<hr>"
            '<div style="font-family: Arial; font-size: 20px; text-align: center; color:grey;">{{GermanExample}}</div>'
            "<br>{{GermanAudio}}"
        ),
    },
]

ANKI_CSS = ".card { font-family: arial; font-size: 20px; text-align: center; color: black; background-color: white; }"


def filename_to_id(name: str) -> int:
    hash_hex = hashlib.sha256(name.encode()).hexdigest()
    return int(hash_hex, 16) % (2**31 - 1) + 1


def clean_text_for_tts(text: str) -> str:
    """Clean German text for TTS: remove gender suffixes, plural markers, etc."""
    if not text:
        return ""
    t = text
    t = re.sub(r"\b([Dd]er)\s*/\s*[Dd]ie\b", r"\1", t)
    t = re.sub(r"\b([Ee]in)\s*/\s*[Ee]ine\b", r"\1", t)
    t = re.sub(r"\b(\w+)\s*/\s*(?:[Ii]n|[Ff]rau)\b(?:,\s*[-/\w]+)?", r"\1", t)
    t = re.sub(r",\s*(Sg|Pl)\.", "", t)
    t = re.sub(r",\s*-\w+(?:/-\w+)*", "", t)
    t = re.sub(r"\b(Sg|Pl)\.\s*", " ", t)
    t = re.sub(r"\((?:Sg|Pl)\.?\)", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    t = t.strip(" .,;:-_()/")
    return t.strip()


def parse_file(filepath: Path) -> List[Tuple[str, str, str, str]]:
    """Parse a tab-separated vocabulary file."""
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 0 and line.startswith("Deutsch"):
                continue
            if line.startswith("==="):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            de_word = parts[0].strip()
            ru_word = parts[1].strip()
            de_example = parts[2].strip()
            ru_example = parts[3].strip()
            if not de_word or not ru_word:
                continue
            entries.append((de_word, ru_word, de_example, ru_example))
    return entries


async def generate_tts(text: str, filepath: Path) -> bool:
    """Generate TTS audio using edge-tts. Returns True on success."""
    cleaned = clean_text_for_tts(text)
    if not cleaned:
        return False
    try:
        communicate = edge_tts.Communicate(cleaned, TTS_VOICE)
        await communicate.save(str(filepath))
        if filepath.is_file() and filepath.stat().st_size > 500:
            return True
        else:
            if filepath.is_file():
                filepath.unlink()
            return False
    except Exception as e:
        print(f"  TTS error for '{text[:40]}': {e}")
        return False


async def process_file(
    filepath: Path, deck_name: str, anki_model: genanki.Model, media_dir: Path
) -> Tuple[genanki.Deck, List[str]]:
    """Process a single vocabulary file, return deck and media files."""
    print(f"\n{'='*60}")
    print(f"Processing: {filepath.name} -> {deck_name}")
    print(f"{'='*60}")

    entries = parse_file(filepath)
    deck_id = filename_to_id(f"linie1b12_{filepath.stem}")
    anki_deck = genanki.Deck(deck_id, deck_name)
    media_files = []

    if not entries:
        print(f"  No entries found in {filepath.name}, skipping.")
        return anki_deck, media_files

    print(f"  Found {len(entries)} word entries.")

    for idx, (de_word, ru_word, de_example, ru_example) in enumerate(entries):
        print(f"  [{idx+1}/{len(entries)}] {de_word} -> {ru_word}")

        tts_text = f"{clean_text_for_tts(de_word)}. {de_example}"
        audio_filename = f"tts_de_{filepath.stem}_{idx:03d}.mp3"
        audio_path = media_dir / audio_filename

        audio_tag = ""
        if not audio_path.is_file() or audio_path.stat().st_size < 500:
            if await generate_tts(tts_text, audio_path):
                audio_tag = f"[sound:{audio_filename}]"
                media_files.append(str(audio_path.resolve()))
            else:
                print(f"    TTS failed for: {de_word}")
        else:
            audio_tag = f"[sound:{audio_filename}]"
            media_files.append(str(audio_path.resolve()))

        note = genanki.Note(
            model=anki_model,
            fields=[ru_word, ru_example, de_word, de_example, audio_tag],
            tags=[f"linie1b12-kap{filepath.stem}"],
        )
        anki_deck.add_note(note)

    print(f"  -> {len(entries)} cards added to subdeck")
    return anki_deck, media_files


async def main():
    files = sorted(INPUT_DIR.glob("[0-9][0-9].txt"))
    if not files:
        print(f"No numbered .txt files found in {INPUT_DIR}")
        return

    print(f"Found {len(files)} chapter files: {[f.name for f in files]}")
    print(f"TTS voice: {TTS_VOICE}")

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    media_dir = OUTPUT_BASE_DIR / "_media"
    media_dir.mkdir(parents=True, exist_ok=True)

    anki_model = genanki.Model(
        MODEL_ID, "Linie1B12_RU_DE",
        fields=ANKI_FIELDS, templates=ANKI_TEMPLATES, css=ANKI_CSS,
    )

    all_decks = []
    all_media = []
    total_notes = 0

    for filepath in files:
        stem = filepath.stem
        topic = CHAPTER_TOPICS.get(stem, stem)
        # Subdeck: "Linie1 B1.2::16 – Prüfungen und Lernen"
        deck_name = f"{PARENT_DECK_NAME}::{topic}"

        deck, media_files = await process_file(filepath, deck_name, anki_model, media_dir)
        all_decks.append(deck)
        all_media.extend(media_files)
        total_notes += len(deck.notes)

    if not total_notes:
        print("\nNo notes generated. Skipping package.")
        return

    # Single .apkg with all subdecks
    apkg_path = OUTPUT_BASE_DIR / "Linie1_B1.2.apkg"
    package = genanki.Package(all_decks)
    package.media_files = all_media
    package.write_to_file(str(apkg_path))

    print(f"\n{'='*60}")
    print(f"Done! Saved {total_notes} cards across {len(all_decks)} subdecks")
    print(f"Package: {apkg_path.resolve()}")
    print(f"Media files: {len(all_media)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
