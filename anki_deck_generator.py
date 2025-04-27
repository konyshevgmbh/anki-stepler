import argparse
import hashlib
import random
import re
import sqlite3
import gc
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, List, Set

# Use icecream for better debugging output
from icecream import ic

import genanki
import torch
import torch.serialization
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# --- Icecream Configuration ---
ic.configureOutput(prefix='|', includeContext=False)
# ic.disable() # Uncomment this line to disable all icecream logging


class AnkiDeckGeneratorMultiLLM_DBState:
    """
    Generates Anki decks using separate LLMs for example generation and translation,
    followed by TTS, operating in distinct phases. Uses the SQLite cache as the primary
    state manager, eliminating the need for an in-memory processed_data list.
    Removes statistics tracking for a cleaner implementation.
    """

    # --- Class Constants ---
    _MEDIA_CACHE_DIR_NAME = "_media_cache"
    _PROMPTS_DIR_NAME = "prompts"
    _PLACEHOLDER_INPUT_WORD = "{input_word}"
    _PLACEHOLDER_SOURCE_EXAMPLE = "{source_example}"
    _LANG_FULL_NAME_MAP = {
        "de": "german", "en": "english", "ru": "russian", "fr": "french",
        "es": "spanish", "it": "italian", "pt": "portuguese", "pl": "polish",
        "tr": "turkish", "nl": "dutch", "cs": "czech", "ar": "arabic",
        "zh-cn": "chinese", "ja": "japanese", "ko": "korean", "hu": "hungarian"
    }
    _MODEL_ID = 1998089417 # New base ID for this version (incremented)
    _ANKI_FIELDS = [
        {'name': 'SourceWord'}, {'name': 'SourceExample'},
        {'name': 'TargetWord'}, {'name': 'TargetExample'},
        {'name': 'SourceAudio'}, {'name': 'TargetAudio'},
    ]
    _ANKI_TEMPLATES = [
        {   'name': 'Card 1 (Source -> Target)',
            'qfmt': """<div style='font-family: Arial; font-size: 30px; text-align: center;'>{{SourceWord}}</div><hr><div style='font-family: Arial; font-size: 20px; text-align: center; color:grey;'>{{SourceExample}}</div><br>{{SourceAudio}}""",
            'afmt': """{{FrontSide}}<hr id=answer><div style='font-family: Arial; font-size: 30px; text-align: center;'>{{TargetWord}}</div><hr><div style='font-family: Arial; font-size: 20px; text-align: center; color:grey;'>{{TargetExample}}</div><br>{{TargetAudio}}""",
        },]
    _CSS = ".card { font-family: arial; font-size: 20px; text-align: center; color: black; background-color: white; }"

    def __init__(self,
                 input_filepath: str,
                 deck_name: str,
                 output_dir: str,
                 llm_model_example_gen: str,
                 llm_model_translate: str,
                 tts_model_name: str,
                 tts_speaker: str,
                 quantization_requested: bool,
                 lang_src_code: str,
                 lang_target_code: str):
        ic.enable()
        # -- Store Configuration --
        self.input_file = Path(input_filepath)
        self._DECK_ID = self._filename_to_number(self.input_file.name)
        self.deck_name = deck_name if deck_name else f"Vocab {self.input_file.stem}"
        self.output_path = Path(output_dir)
        self.llm_model_example_gen = llm_model_example_gen
        self.llm_model_translate = llm_model_translate
        self.tts_model_name = tts_model_name
        self.tts_speaker = tts_speaker
        self.quantization_requested = quantization_requested
        self.lang_src_code = lang_src_code.lower()
        self.lang_target_code = lang_target_code.lower()

        if not self.input_file.is_file(): raise FileNotFoundError(f"Input file not found: {self.input_file}")

        self.lang_src_full = self._LANG_FULL_NAME_MAP.get(self.lang_src_code)
        self.lang_target_full = self._LANG_FULL_NAME_MAP.get(self.lang_target_code)
        if not self.lang_src_full or not self.lang_target_full: raise ValueError(f"Language code invalid/unsupported. Available: {list(self._LANG_FULL_NAME_MAP.keys())}")
        if self.lang_src_code == self.lang_target_code: raise ValueError("Source and target languages cannot be the same.")

        self.script_dir = Path(__file__).resolve().parent
        self.prompts_dir = self.script_dir / self._PROMPTS_DIR_NAME
        self.media_path = self.output_path / self._MEDIA_CACHE_DIR_NAME
        db_filename = self.input_file.stem + ".db"
        self.db_path = self.output_path / db_filename

        try:
            self.output_path.mkdir(parents=True, exist_ok=True)
            self.media_path.mkdir(parents=True, exist_ok=True)
        except OSError as e: raise OSError(f"Failed to create output/media directories: {e}")

        self.device: Optional[torch.device] = None
        self.quantization_active: bool = False
        self.quantization_config: Optional[BitsAndBytesConfig] = None
        self.llm_pipeline: Optional[pipeline] = None
        self.tts_model: Optional[TTS] = None
        self.db_conn: Optional[sqlite3.Connection] = None

        self._setup_anki_model_deck()

        self.source_words: List[str] = [] # Still need the list of words to process
        self.anki_notes: List[genanki.Note] = []
        self.media_files_added: Set[str] = set()

    # --- Core Methods (Unchanged) ---
    def _filename_to_number(self, filename: str) -> int:
        hash_object = hashlib.sha256(filename.encode())
        hash_hex = hash_object.hexdigest()
        return int(hash_hex, 16) % (2**31 - 1) + 1

    def _setup_anki_model_deck(self):
        self.anki_model = genanki.Model(
            self._MODEL_ID, 'Multilang Vocab Model v7 (Multi-LLM DBState NoStats)', # Updated name
            fields=self._ANKI_FIELDS, templates=self._ANKI_TEMPLATES, css=self._CSS
        )
        self.anki_deck = genanki.Deck(self._DECK_ID, self.deck_name)

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        if not text: return "unnamed"
        sanitized = re.sub(r'[\\/*?:"<>|]', '_', text)
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = re.sub(r'[^\w\-]+', '', sanitized)
        sanitized = sanitized.strip('_-')
        return sanitized[:50] if sanitized else "unnamed"

    def _clean_text(self, text: str) -> str:
        """
        Cleans text for TTS, including simplifying complex German gendered nouns.
        E.g., "der/die Architekt/in, -en/-nen" -> "der Architekt"
        """
        if not text: return ""
        cleaned_text = text
        # Optional: Log original text for debugging complex cases
        # ic("Original for cleaning:", cleaned_text)

        # --- Step 1: Simplify Complex Gendered Nouns ---

        # Pattern 1: Article/Article Noun/Suffix[, Suffixes]
        # Example: der/die Architekt/in, -en/-nen
        #          ein/eine Lehrer/in
        # Captures: (Article1), (Article2), (BaseNoun), (NounSuffix), (Optional Plural Junk)
        pattern1 = r"\b([Dd]er|[Ee]in)\s*/\s*([Dd]ie|[Ee]ine)\s+(\w+)\s*/\s*([Ii]n|[Ff]rau)\b(,\s*[-/\w]+)?"
        # Replace with: Article1 BaseNoun
        cleaned_text = re.sub(pattern1, r"\1 \3", cleaned_text)

        # Pattern 2: Noun/Suffix[, Suffixes] (No preceding article combo)
        # Example: Architekt/in, -en/-nen
        #          Kaufmann/frau
        # Captures: (BaseNoun), (NounSuffix), (Optional Plural Junk)
        pattern2 = r"\b(\w+)\s*/\s*([Ii]n|[Ff]rau)\b(,\s*[-/\w]+)?"
        # Replace with: BaseNoun
        cleaned_text = re.sub(pattern2, r"\1", cleaned_text)

        # Pattern 3: Handle remaining article slashes separately if Pattern 1 didn't catch them
        # Example: der/die Arzt
        cleaned_text = re.sub(r"\b([Dd]er)/[Dd]ie\b", r"\1", cleaned_text)
        cleaned_text = re.sub(r"\b([Ee]in)/[Ee]ine\b", r"\1", cleaned_text)

        # Optional: Log after specific noun cleaning
        # ic("After complex noun cleaning:", cleaned_text)

        # --- Step 2: Remove Simpler Plural/Gender Annotations ---
        # Handles ", Sg.", ", Pl.", " -en", " /e" etc. (if not caught above)
        # Remove standard annotations like ", Sg." or ", Pl."
        cleaned_text = re.sub(r",\s*(Sg|Pl)\.", "", cleaned_text)
        # Remove suffixes indicated by space/comma + hyphen/slash
        cleaned_text = re.sub(r'(?:,\s*|\s+)(?:["\'\s]*-?/?\s*(?:nen|en|n|s|e))\b(?:/\w+)?\.?', '', cleaned_text)
        # Remove standalone Sg./Pl. as whole words
        cleaned_text = re.sub(r'\b(Sg|Pl)\.\s*', ' ', cleaned_text)

        # Optional: Log after simple annotation removal
        # ic("After simple annotation removal:", cleaned_text)

        # --- Step 3: General Cleanup ---
        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text) # Consolidate multiple spaces
        cleaned_text = re.sub(r'\s+([,.!?])', r'\1', cleaned_text) # Remove space before punctuation
        # More aggressive trailing character removal
        cleaned_text = cleaned_text.rstrip(' .,!?;:-_()[]{}"\'\n\t/') # Added /
        cleaned_text = cleaned_text.strip() # Final trim

        # Optional: Log final result
        # ic("Final Cleaned:", cleaned_text)

        return cleaned_text

    def _build_prompt(self, word: str) -> str:
        """
        Dynamically build the prompt for sentence generation based on source language.
        """
        lang = self.lang_src_code.lower()

        if lang == "en":
            return f"Write a simple short sentence (max 4 words) using the word '{word}'. Sentence:"
        elif lang == "de":
            return f"Schreibe einen einfachen, kurzen Satz (max. 4 Wörter) mit dem Wort '{word}'. Satz:"
        elif lang == "ru":
            return f"Напиши простое короткое предложение (макс. 4 слова) со словом '{word}'. Предложение:"
        elif lang == "fr":
            return f"Écris une phrase courte et simple (4 mots maximum) avec le mot '{word}'. Phrase:"
        elif lang == "es":
            return f"Escribe una frase sencilla y corta (máximo 4 palabras) usando la palabra '{word}'. Frase:"
        elif lang == "it":
            return f"Scrivi una frase breve e semplice (massimo 4 parole) usando la parola '{word}'. Frase:"
        elif lang == "pt":
            return f"Escreva uma frase simples e curta (máximo 4 palavras) usando a palavra '{word}'. Frase:"
        elif lang == "pl":
            return f"Napisz krótkie, proste zdanie (maksymalnie 4 słowa) używając słowa '{word}'. Zdanie:"
        elif lang == "tr":
            return f"'{word}' kelimesini kullanarak kısa ve basit bir cümle yazın (maksimum 4 kelime). Cümle:"
        elif lang == "nl":
            return f"Schrijf een korte, eenvoudige zin (maximaal 4 woorden) met het woord '{word}'. Zin:"
        elif lang == "cs":
            return f"Napiš krátkou, jednoduchou větu (maximálně 4 slova) se slovem '{word}'. Věta:"
        elif lang == "ar":
            return f"اكتب جملة بسيطة وقصيرة (بحد أقصى 4 كلمات) باستخدام الكلمة '{word}'. الجملة:"
        elif lang == "zh-cn":
            return f"用'{word}'写一个简单的短句（最多4个词）。句子："
        elif lang == "ja":
            return f"単語「{word}」を使って、簡単で短い文（最大4語）を書いてください。文："
        elif lang == "ko":
            return f"단어 '{word}'를 사용하여 짧고 간단한 문장(최대 4단어)을 작성하세요. 문장:"
        elif lang == "hu":
            return f"Írj egy egyszerű, rövid mondatot (max. 4 szó) a következő szóval: '{word}'. Mondat:"
        else:
            return f"Write a short simple sentence (max 4 words) using the word '{word}' in {lang}. Sentence:"
    
    # --- LLM Content Generation ---
    def _generate_source_example(self, source_word: str) -> Optional[str]:
        if not self.llm_pipeline: 
            ic("Example Gen LLM unavailable.")
            return None
        
        cleaned_word = self._clean_text(source_word)
        ic(cleaned_word)
        prompt =  self._build_prompt(cleaned_word)
        for attempt in range(4):  
            try:
                
                outputs = self.llm_pipeline(prompt, max_new_tokens=25, do_sample=True, temperature=0.9, repetition_penalty=1.1, return_full_text=False, pad_token_id=self.llm_pipeline.tokenizer.eos_token_id, eos_token_id=self.llm_pipeline.tokenizer.eos_token_id)
                full_text = outputs[0]['generated_text'] if outputs and 'generated_text' in outputs[0] else ""
                full_text = self._clear_llm_text(full_text)
                words_count = len(full_text.split())
                if words_count > 10 or words_count < 2:
                    ic("Example Gen Error: Invalid word count") 
                    ic(attempt)
                    ic(full_text)  
                    continue
                return full_text 

    
            except Exception as e:
                ic("Example Gen Error: LLM failed"); ic(source_word); ic(e)
                return None

        return None 
    

    def _clear_llm_text(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r'[\*\_\n\r]+', ' ', text).strip()
    

    def _generate_translations(self, source_word: str, source_example: str) -> Optional[Dict]:
        if not self.llm_pipeline:
            ic("Translation LLM unavailable.")
            return None

        try:
            # Batch translation: send word and sentence together
            clear_text = self._clean_text(source_word)
            inputs = [clear_text, source_example]
            outputs = self.llm_pipeline(inputs, max_length=100)

            if not outputs or len(outputs) < 2:
                ic("Translation Error: Incomplete output")
                return None

            translated_word = self._clear_llm_text( outputs[0]["translation_text"] )
            translated_example =  self._clear_llm_text( outputs[1]["translation_text"] )
            return {
                "source_word": source_word,
                "source_example": source_example,
                "target_word": translated_word,
                "target_example": translated_example,
            }

        except Exception as e:
            ic("Translation Error: LLM failed")
            ic(source_word)
            ic(source_example)
            ic(e)
            return None


    # --- TTS Generation ---
    def _generate_tts_audio(self, text: str, lang_code: str, base_filename: str) -> Optional[str]:
        if not self.tts_model: return None
        if not text: return None
        try:
            timestamp = f"{int(time.time())}"
            filename = f"tts_{lang_code}_{timestamp}.wav"
            filepath = self.media_path / filename
            cleaned_tts_text = self._clean_text(text)
            if not cleaned_tts_text: ic("TTS Warning: Skipping empty text"); ic(text); return None
            ic(lang_code)
            ic(filename)
            self.tts_model.tts_to_file(text=cleaned_tts_text, speaker=self.tts_speaker, language=lang_code, file_path=str(filepath))
            time.sleep(0.1)
            if filepath.is_file() and filepath.stat().st_size > 1000:
                 ic(filename)
                 return filename
            else:
                 ic("TTS Error: Invalid/empty file"); ic(filename)
                 if filepath.is_file():
                     try: filepath.unlink()
                     except OSError as e: ic("TTS Warning: Delete failed"); ic(e)
                 return None
        except Exception as e:
            ic("TTS Error: Generation failed"); ic(text[:30]); ic(e)
            return None

    # --- Caching (Simplified - Read Only in Phases) ---
    def _get_cached_entry(self, source_word: str) -> Optional[Dict]:
        """Retrieves the full cache entry (or None) for a word."""
        if not self.db_conn: return None
        try:
            cursor = self.db_conn.cursor()
            # Select all potentially generated fields
            cursor.execute("""
                SELECT source_example, target_word, target_example,
                       source_audio_filename, target_audio_filename
                FROM word_cache WHERE source_word = ? AND source_lang = ? AND target_lang = ?
            """, (source_word, self.lang_src_full, self.lang_target_full))
            row = cursor.fetchone()
            if row:
                # Return dict with all fields, using None if not set in DB
                return {
                    "source_word": source_word, # Always include the key
                    "source_example": row[0],
                    "target_word": row[1],
                    "target_example": row[2],
                    "source_audio_filename": row[3],
                    "target_audio_filename": row[4]
                }
            else:
                # Return a minimal dict if no entry exists
                return {"source_word": source_word}
        except sqlite3.Error as e:
            ic("Cache Error: Retrieval failed"); ic(source_word); ic(e)
            # Return minimal dict on error to avoid halting processing
            return {"source_word": source_word}

    def _upsert_cached_data(self, data: dict):
        """Inserts/replaces data in cache. Requires source_word."""
        if not self.db_conn: return
        source_word = data.get("source_word")
        if not source_word: ic("Cache Error: Cannot upsert without source_word"); return

        # Prepare values, using None if not available in the input data dict
        src_example = data.get("source_example")
        tgt_word = data.get("target_word")
        tgt_example = data.get("target_example")
        src_audio = data.get("source_audio_filename") # Key used in this function
        tgt_audio = data.get("target_audio_filename") # Key used in this function

        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO word_cache ( source_word, source_lang, target_lang, source_example, target_word, target_example, source_audio_filename, target_audio_filename, created_at, updated_at )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(source_word, source_lang, target_lang) DO UPDATE SET
                    source_example = coalesce(excluded.source_example, word_cache.source_example),
                    target_word = coalesce(excluded.target_word, word_cache.target_word),
                    target_example = coalesce(excluded.target_example, word_cache.target_example),
                    source_audio_filename = coalesce(excluded.source_audio_filename, word_cache.source_audio_filename),
                    target_audio_filename = coalesce(excluded.target_audio_filename, word_cache.target_audio_filename),
                    updated_at = CURRENT_TIMESTAMP
                -- Only trigger update if at least one NEW value is provided AND different
                WHERE
                    (excluded.source_example IS NOT NULL AND ifnull(word_cache.source_example, '') != excluded.source_example) OR
                    (excluded.target_word IS NOT NULL AND ifnull(word_cache.target_word, '') != excluded.target_word) OR
                    (excluded.target_example IS NOT NULL AND ifnull(word_cache.target_example, '') != excluded.target_example) OR
                    (excluded.source_audio_filename IS NOT NULL AND ifnull(word_cache.source_audio_filename, '') != excluded.source_audio_filename) OR
                    (excluded.target_audio_filename IS NOT NULL AND ifnull(word_cache.target_audio_filename, '') != excluded.target_audio_filename)
            """, (
                source_word, self.lang_src_full, self.lang_target_full,
                src_example, tgt_word, tgt_example,
                src_audio, tgt_audio
            ))
            if cursor.rowcount > 0: # Check if update actually happened
                 ic("Cache updated")
            self.db_conn.commit()

        except sqlite3.Error as e:
            ic("Cache Error: Upsert failed"); ic(source_word); ic(e)

    # --- Resource Loading/Unloading (Unchanged) ---
    def _determine_device_and_quantization(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.quantization_active = self.quantization_requested
            if self.quantization_active:
                try: self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                except ImportError: print("[Warning] `bitsandbytes` not installed. Cannot use quantization."); self.quantization_active = False
                except Exception as e: print(f"[Warning] Failed init BitsAndBytesConfig: {e}. Disabling quantization."); self.quantization_active = False
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
            if self.quantization_requested: print("Warning: Quantization with MPS limited. Disabling."); self.quantization_active = False
        else:
            self.device = torch.device("cpu")
            if self.quantization_requested: print("Warning: Quantization needs GPU. Using CPU without."); self.quantization_active = False
        ic(self.device)
        ic(self.quantization_active)

    def _load_llm_pipeline(self, model_name: str) -> bool:
        if self.llm_pipeline: ic("Warning: Unloading existing LLM"); self._unload_llm()
        if not self.device: raise RuntimeError("Device not determined before loading LLM.")
        ic(model_name)
        try:
            torch_dtype = torch.float16
            current_quant_config = None
            if self.quantization_active and self.device.type == 'cuda':
                torch_dtype = torch.bfloat16; current_quant_config = self.quantization_config
            elif self.device.type == 'mps': torch_dtype = torch.float16
            elif self.device.type != 'cuda': torch_dtype = torch.float32

            model_kwargs = {"torch_dtype": torch_dtype}
            using_device_map = False
            if current_quant_config and self.device.type == 'cuda':
                 model_kwargs["quantization_config"] = current_quant_config
                 model_kwargs["device_map"] = "auto"; using_device_map = True
            elif self.device.type != 'cpu':
                 try: import accelerate; model_kwargs["device_map"] = "auto"; using_device_map = True
                 except ImportError: print("Note: `accelerate` not found. Loading model directly.")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            if not using_device_map and self.device.type != 'cpu': model.to(self.device); ic(self.device)

            pipeline_device = -1
            if not using_device_map:
                if self.device.type == 'cuda': pipeline_device = self.device.index if self.device.index is not None else 0
                elif self.device.type == 'mps': pipeline_device = 0

            if using_device_map: self.llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
            else: self.llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=pipeline_device)
            ic(model_name)
            return True
        except Exception as e:
            print(f"[Critical Error] Failed load LLM '{model_name}': {e}")
            if "out of memory" in str(e).lower(): print(">>> OOM error. <<<")
            self.llm_pipeline = None; return False

    _LANG_NLLB_CODE_MAP = {
        "en": "eng_Latn",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "it": "ita_Latn",
        "pt": "por_Latn",
        "pl": "pol_Latn",
        "ru": "rus_Cyrl",
        "tr": "tur_Latn",
        "nl": "nld_Latn",
        "cs": "ces_Latn",
        "ar": "arb_Arab",
        "zh-cn": "zho_Hans",
        "ja": "jpn_Jpan",
        "ko": "kor_Hang",
        "hu": "hun_Latn"
    }
    def _load_translations_pipeline(self, model_name: str) -> bool:
        """Load a translation pipeline (for NLLB models)."""
        if self.llm_pipeline:
            ic("Warning: Unloading existing LLM")
            self._unload_llm()
        if not self.device:
            raise RuntimeError("Device not determined before loading translation model.")
        ic(model_name)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            if self.device.type != "cpu":
                model.to(self.device)
                ic(self.device)

            # === Fix the language codes for NLLB ===
            src_lang_code = self._LANG_NLLB_CODE_MAP.get(self.lang_src_code, self.lang_src_code)
            tgt_lang_code = self._LANG_NLLB_CODE_MAP.get(self.lang_target_code, self.lang_target_code)

            self.llm_pipeline = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                src_lang=src_lang_code,
                tgt_lang=tgt_lang_code,
                device=0 if self.device.type != "cpu" else -1
            )
            ic( src_lang_code )
            ic( tgt_lang_code )
            return True

        except Exception as e:
            print(f"[Critical Error] Failed load Translation Model '{model_name}': {e}")
            self.llm_pipeline = None
            return False

    def _load_tts_model(self):
        if not self.device: raise RuntimeError("Device not determined before loading TTS.")
        ic(self.tts_model_name)
        try:
            warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
            tts_device_str = "cuda" if self.device.type == 'cuda' else "cpu"
            if self.device.type == 'mps': print("Note: Mapping MPS to CPU for Coqui TTS."); tts_device_str = "cpu"
            self.tts_model = TTS(model_name=self.tts_model_name, progress_bar=True).to(tts_device_str)
            ic(tts_device_str)
            warnings.filterwarnings("default", category=UserWarning, module='transformers')
            return True
        except Exception as e:
            print(f"[Critical Error] Failed load XTTS '{self.tts_model_name}': {e}")
            print(">>> Ensure 'espeak-ng' is installed. <<<")
            return False

    def _init_db(self):
        ic(self.db_path)
        try:
            self.db_conn = sqlite3.connect(self.db_path)
            cursor = self.db_conn.cursor(); cursor.execute(""" CREATE TABLE IF NOT EXISTS word_cache ( source_word TEXT NOT NULL, source_lang TEXT NOT NULL, target_lang TEXT NOT NULL, source_example TEXT, target_word TEXT, target_example TEXT, source_audio_filename TEXT, target_audio_filename TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (source_word, source_lang, target_lang) )""")
            cursor.execute(""" CREATE TRIGGER IF NOT EXISTS update_word_cache_updated_at AFTER UPDATE ON word_cache FOR EACH ROW WHEN ifnull(OLD.source_example,'') IS NOT ifnull(NEW.source_example,'') OR ifnull(OLD.target_word,'') IS NOT ifnull(NEW.target_word,'') OR ifnull(OLD.target_example,'') IS NOT ifnull(NEW.target_example,'') OR ifnull(OLD.source_audio_filename,'') IS NOT ifnull(NEW.source_audio_filename,'') OR ifnull(OLD.target_audio_filename,'') IS NOT ifnull(NEW.target_audio_filename,'') BEGIN UPDATE word_cache SET updated_at=CURRENT_TIMESTAMP WHERE source_word=OLD.source_word AND source_lang=OLD.source_lang AND target_lang=OLD.target_lang; END; """)
            self.db_conn.commit()
            return True
        except sqlite3.Error as e: print(f"[Critical Error] Failed init DB '{self.db_path}': {e}"); self.db_conn=None; return False

    def _unload_llm(self):
        if hasattr(self, 'llm_pipeline') and self.llm_pipeline:
            current_model_name = "Unknown"
            try:
                if hasattr(self.llm_pipeline, 'model') and hasattr(self.llm_pipeline.model, 'config'):
                     current_model_name = getattr(self.llm_pipeline.model.config, '_name_or_path', 'Unknown')
            except Exception: pass
            if hasattr(self.llm_pipeline, 'model'): del self.llm_pipeline.model
            if hasattr(self.llm_pipeline, 'tokenizer'): del self.llm_pipeline.tokenizer
            del self.llm_pipeline; self.llm_pipeline = None
            ic(current_model_name)
            if self.device and self.device.type == 'cuda':
                try: torch.cuda.empty_cache()
                except Exception as e: ic("CUDA Warning: Cache clear failed"); ic(e)
            gc.collect()

    def close_resources(self):
        self._unload_llm()
        if hasattr(self, 'tts_model') and self.tts_model:
             del self.tts_model; self.tts_model = None
             if self.device and self.device.type == 'cuda':
                 try: torch.cuda.empty_cache()
                 except Exception as e: ic("CUDA Warning: Cache clear failed"); ic(e)
        if self.db_conn:
            try: self.db_conn.close()
            except sqlite3.Error as e: ic("DB Warning: Close failed"); ic(e)
            self.db_conn = None
        collected = gc.collect(); ic(collected)

    # --- Phased Processing Logic (Using DB as State) ---
    def _run_phase_0_init(self) -> bool:
        """Loads initial resources and reads source words."""
        print("\n--- Phase 0: Initial Setup ---")
        self._determine_device_and_quantization()
        if not self._init_db(): return False
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                self.source_words = [line.strip() for line in f if line.strip()]
            ic(len(self.source_words))
            if not self.source_words: print("Input file empty."); return False
        except Exception as e: print(f"[Critical] Failed read input file: {e}"); return False
        return True

    def _run_phase_1_example_generation(self) -> bool:
        print("\n--- Phase 1: Source Example Generation ---")
        if not self._load_llm_pipeline(self.llm_model_example_gen): return False

        for i, source_word in enumerate(self.source_words):
            print(f"\n--- Phase 1: Processing '{source_word}' ({i+1}/{len(self.source_words)}) ---")
            ic(source_word)
            cached_entry = self._get_cached_entry(source_word) # Get current state from DB

            # Skip if example already exists
            if cached_entry.get("source_example"):
                ic(cached_entry.get("source_example"))
                continue # Skip to next word

            example = self._generate_source_example(source_word)
            ic(example)
            if example:
                # Prepare data for update (only the new field)
                update_data = {"source_word": source_word, "source_example": example}
                self._upsert_cached_data(update_data) # Update cache
            # else: failure logged by _generate_source_example

        self._unload_llm()
        return True

    def _run_phase_2_translation(self) -> bool:
        print("\n--- Phase 2: Translation ---")
        if not self._load_translations_pipeline(self.llm_model_translate): return False

        for i, source_word in enumerate(self.source_words):
            print(f"\n--- Phase 2: Processing '{source_word}' ({i+1}/{len(self.source_words)}) ---")
            cached_entry = self._get_cached_entry(source_word)

            # Skip if translation already exists OR if source example is missing
            source_example = cached_entry.get("source_example")
            if not source_example:
                ic("Translate Warning: Skipping missing src example")
                continue
            if cached_entry.get("target_word") and cached_entry.get("target_example"):
                ic("Translation found in cache.")
                ic(cached_entry.get("target_word"))
                ic(cached_entry.get("target_example"))
                continue

            translations = self._generate_translations(source_word, source_example)
            ic(translations)
            if translations:
                self._upsert_cached_data(translations) # Update cache

        self._unload_llm()
        return True

    def _run_phase_3_tts(self) -> bool:
        print("\n--- Phase 3: TTS ---")
        if not self._load_tts_model(): return False

        for i, source_word in enumerate(self.source_words):
            print(f"\n--- Phase 3: Processing '{source_word}' ({i+1}/{len(self.source_words)}) ---")
            ic(source_word)
            cached_entry = self._get_cached_entry(source_word)

            # Skip if text generation failed earlier
            required_text = ["source_example", "target_word", "target_example"]
            if not all(cached_entry.get(k) for k in required_text):
                ic("TTS Warning: Skipping missing text")
                continue

            needs_generation = False
            update_data = {"source_word": source_word} # Start update dict

            # --- Source Audio Check ---
            src_audio_filename = cached_entry.get("source_audio_filename")
            src_audio_valid = False
            if src_audio_filename:
                full_path = self.media_path / src_audio_filename
                if full_path.is_file() and full_path.stat().st_size > 1000:
                    src_audio_valid = True
                    update_data["source_audio_filename"] = src_audio_filename # Keep existing valid filename
                else: ic("TTS Warning: Invalid cached src audio"); ic(src_audio_filename)

            if not src_audio_valid:
                needs_generation = True
                tts_text_src = f"{source_word}. {cached_entry['source_example']}"
                generated_fname = self._generate_tts_audio(tts_text_src, self.lang_src_code, source_word)
                if generated_fname: update_data["source_audio_filename"] = generated_fname
                # Else: failure logged, filename will be None in update_data

            # --- Target Audio Check ---
            tgt_audio_filename = cached_entry.get("target_audio_filename")
            tgt_audio_valid = False
            if tgt_audio_filename:
                full_path = self.media_path / tgt_audio_filename
                if full_path.is_file() and full_path.stat().st_size > 1000:
                    tgt_audio_valid = True
                    update_data["target_audio_filename"] = tgt_audio_filename # Keep existing valid filename
                else: ic("TTS Warning: Invalid cached tgt audio"); ic(tgt_audio_filename)

            if not tgt_audio_valid:
                needs_generation = True # Flag that something needs generating
                tts_text_tgt = f"{cached_entry['target_word']}. {cached_entry['target_example']}"
                generated_fname = self._generate_tts_audio(tts_text_tgt, self.lang_target_code, cached_entry['target_word'])
                if generated_fname: update_data["target_audio_filename"] = generated_fname
                # Else: failure logged, filename will be None in update_data

            # Update cache only if new audio was generated
            if needs_generation and ("source_audio_filename" in update_data or "target_audio_filename" in update_data):
                 self._upsert_cached_data(update_data)


 
        return True

    def _run_phase_4_finalize_and_package(self):
        print("\n--- Phase 4: Finalize & Package ---")
        total_words = len(self.source_words)
        notes_added = 0
        ic(total_words)

        for i, source_word in enumerate(self.source_words):
             if (i + 1) % 50 == 0 or i == total_words - 1:
                 print(f"\n--- Phase 4: Finalizing '{source_word}' ({i+1}/{total_words}) ---")

             # Read final complete data from cache for note generation
             final_data = self._get_cached_entry(source_word)

             required_keys = ["source_example", "target_word", "target_example", "source_audio_filename", "target_audio_filename"]
             if not final_data or not all(final_data.get(k,"").strip() for k in required_keys):
                 # ic("Finalize Warning: Skipping incomplete data"); ic(source_word) # Can be noisy
                 continue # Skip if any required field is missing/empty

             # === Prepare Anki Note ===
             src_audio_file = final_data["source_audio_filename"]
             tgt_audio_file = final_data["target_audio_filename"]
             source_audio_anki = f"[sound:{src_audio_file}]"
             target_audio_anki = f"[sound:{tgt_audio_file}]"

             fpath_src = self.media_path / src_audio_file
             if fpath_src.is_file() and fpath_src.stat().st_size > 1000: self.media_files_added.add(str(fpath_src.resolve()))
             else: print(f"   ! Warning: Source audio '{src_audio_file}' invalid/missing for packaging.")
             fpath_tgt = self.media_path / tgt_audio_file
             if fpath_tgt.is_file() and fpath_tgt.stat().st_size > 1000: self.media_files_added.add(str(fpath_tgt.resolve()))
             else: print(f"   ! Warning: Target audio '{tgt_audio_file}' invalid/missing for packaging.")

             note_fields = [ source_word, final_data["source_example"], final_data["target_word"],
                             final_data["target_example"], source_audio_anki, target_audio_anki ]
             tags = [ self.input_file.stem.replace(" ", "_"), f"{self.lang_src_code}-{self.lang_target_code}", 'multi-llm-dbstate-nostats' ] # Update tag
             anki_note = genanki.Note(model=self.anki_model, fields=note_fields, tags=tags)
             self.anki_notes.append(anki_note)
             notes_added += 1

        ic(notes_added)

        # === Generate Anki Package ===
        if not self.anki_notes: print("\nNo notes generated. Skipping package."); return
        ic("Generating Anki package...")
        final_media_list = sorted(list(self.media_files_added))
        ic(len(final_media_list))
        anki_package = genanki.Package(self.anki_deck); anki_package.media_files = final_media_list
        try:
            safe_deck_name = self._sanitize_filename(self.deck_name)
            output_apkg_filename = self.output_path / f"{safe_deck_name}.apkg"
            ic(output_apkg_filename)
            anki_package.write_to_file(output_apkg_filename)
            print("\n--- Anki Package Generation Success! ---")
            print(f"Package saved to: {output_apkg_filename.resolve()}")
            print(f"Added {notes_added} notes with {len(final_media_list)} media files.")
        except Exception as e: print(f"\n[Error] Failed to write Anki package: {e}")

    # --- Main Execution Method ---
    def run(self):
        start_time = time.time()
        print("\n" + "="*60); print(" Starting OFFLINE Anki Deck Generation (Multi-LLM, DB State, No Stats)") # Updated Title
        print(f" Source: {self.lang_src_full} | Target: {self.lang_target_full}")
        print(f" Input: {self.input_file.name} | Deck: {self.deck_name}")
        print(f" LLM Example Gen: {self.llm_model_example_gen}")
        print(f" LLM Translate:   {self.llm_model_translate}")
        print(f" TTS: {self.tts_model_name} (Speaker: {self.tts_speaker}) | Quantization: {self.quantization_requested}")
        print(f" Output Dir: {self.output_path.resolve()}"); print("="*60)

        resources_loaded = False
        try:
            if not self._run_phase_0_init(): return # Just init DB and read words
            resources_loaded = True
            if not self._run_phase_1_example_generation(): return
            if not self._run_phase_2_translation(): return
            if not self._run_phase_3_tts(): return
            self._run_phase_4_finalize_and_package()

        except KeyboardInterrupt: print("\n[Interrupted] Processing stopped by user."); print("Rerun script to resume based on cache.")
        except Exception as e: print(f"\n[Critical Error] Unexpected error: {e}"); import traceback; traceback.print_exc()
        finally:
            if resources_loaded: self.close_resources()
            else: print("Skipping resource cleanup (initial load failed).")
            end_time = time.time()
            print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")


# =============================================================================
# Command Line Interface / Main function wrapper
# =============================================================================
def main():
    print(torch.__version__)
    print(torch.cuda.is_available())

    """Parses arguments and runs the generator."""
    parser = argparse.ArgumentParser(description="Generate Anki deck (Multi-LLM Phased, DB State, No Stats).", formatter_class=argparse.ArgumentDefaultsHelpFormatter) # Updated description
    parser.add_argument("input_file", help="Path to input text file (source words).")
    parser.add_argument("-d", "--deck-name", default="", help="Anki deck name (default: derived from input filename).")
    parser.add_argument("-o", "--output-dir", default="output", help="Directory for .apkg, .db, media cache.")
    parser.add_argument("--lang-src", default="de", help=f"Source language code. Available: {list(AnkiDeckGeneratorMultiLLM_DBState._LANG_FULL_NAME_MAP.keys())}")
    parser.add_argument("--lang-target", default="en", help=f"Target language code. Available: {list(AnkiDeckGeneratorMultiLLM_DBState._LANG_FULL_NAME_MAP.keys())}")
    parser.add_argument("--llm-model-example-gen", default="google/gemma-2-2b-it", help="LLM for source example generation (HF Hub).")
    parser.add_argument("--llm-model-translate", default="facebook/nllb-200-distilled-600M", help="LLM for translation (HF Hub). NLLB recommended.")
    parser.add_argument("--tts-model", default="tts_models/multilingual/multi-dataset/xtts_v2", help="TTS (XTTS) model identifier.")
    parser.add_argument("--tts-speaker", default="Ana Florence", help="Speaker name/ID for XTTSv2 (`tts --list_models`).")
    parser.add_argument("--use-quantization", action="store_true", help="Request 4-bit LLM quantization (requires bitsandbytes & CUDA). Applies to both LLMs.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.is_file(): print(f"[Error] Input file not found: {args.input_file}"); exit(1)
    if args.lang_src.lower() == args.lang_target.lower(): print("[Error] Source and target languages cannot be the same."); exit(1)
    if args.lang_src.lower() not in AnkiDeckGeneratorMultiLLM_DBState._LANG_FULL_NAME_MAP: print(f"[Error] Source language code '{args.lang_src}' not supported."); exit(1)
    if args.lang_target.lower() not in AnkiDeckGeneratorMultiLLM_DBState._LANG_FULL_NAME_MAP: print(f"[Error] Target language code '{args.lang_target}' not supported."); exit(1)

    generator = None
    try:
        default_deck_name = f"Vocab {input_path.stem}"
        deck_name_to_use = args.deck_name if args.deck_name else default_deck_name
        generator = AnkiDeckGeneratorMultiLLM_DBState( # Use the new class name
            input_filepath=args.input_file, deck_name=deck_name_to_use, output_dir=args.output_dir,
            llm_model_example_gen=args.llm_model_example_gen, llm_model_translate=args.llm_model_translate,
            tts_model_name=args.tts_model, tts_speaker=args.tts_speaker,
            quantization_requested=args.use_quantization, lang_src_code=args.lang_src, lang_target_code=args.lang_target
        )
        generator.run()
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as e: print(f"\n[Initialization Error] {e}")
    except Exception as e: print(f"\n[Unhandled Error] {e}"); import traceback; traceback.print_exc()

    print("\nScript finished.")

if __name__ == "__main__":
    main() # Call the main function
