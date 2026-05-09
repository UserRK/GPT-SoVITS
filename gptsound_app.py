import os
import sys
import json
import numpy as np
import soundfile as sf

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "GPT_SoVITS")

PROFILES_DIR = "voice_profiles"
os.makedirs(PROFILES_DIR, exist_ok=True)

# ─── ASR (Faster-Whisper) ─────────────────────────────────────────────────────

import torch
from faster_whisper import WhisperModel

_asr_model = None

def get_asr_model():
    global _asr_model
    if _asr_model is None:
        device   = "cuda" if torch.cuda.is_available() else "cpu"
        compute  = "int8"  # float16 не підтримується на картах з compute capability < 7.0
        # large-v3-turbo: ~800 MB, відмінно розпізнає українську та англійську
        _asr_model = WhisperModel("large-v3-turbo", device=device, compute_type=compute)
    return _asr_model


def transcribe(audio_path):
    """Повертає (text, detected_language)"""
    if not audio_path:
        return "", ""
    model = get_asr_model()
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=None,          # auto-detect
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    text = "".join(seg.text for seg in segments).strip()
    return text, info.language


# ─── TTS backend ──────────────────────────────────────────────────────────────

import inference_webui as tts_backend

# ─── Voice profile helpers ────────────────────────────────────────────────────

def list_profiles():
    return sorted(f[:-5] for f in os.listdir(PROFILES_DIR) if f.endswith(".json"))


def load_profile(name):
    path = os.path.join(PROFILES_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Gradio handlers ──────────────────────────────────────────────────────────

import gradio as gr


def on_audio_upload(audio_path):
    """Автоматично транскрибує завантажений файл."""
    if not audio_path:
        return gr.update(), gr.update(value="")
    try:
        text, lang = transcribe(audio_path)
        lang_label = f"Розпізнана мова: {lang.upper()}" if lang else ""
        return gr.update(value=text), gr.update(value=lang_label)
    except Exception as e:
        return gr.update(), gr.update(value=f"❌ Помилка транскрипції: {e}")


MAX_REF_SEC = 15  # оптимальна довжина референсу для клонування голосу

def save_profile(audio_path, ref_text, name):
    if not audio_path:
        return "❌ Завантажте аудіо файл", gr.update()
    if not name or not name.strip():
        return "❌ Введіть назву профілю", gr.update()

    name = name.strip()
    dest = os.path.join(PROFILES_DIR, f"{name}.wav")

    # Обрізаємо до MAX_REF_SEC секунд і конвертуємо у WAV
    audio, sr = sf.read(audio_path)
    max_samples = MAX_REF_SEC * sr
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    sf.write(dest, audio, sr)

    with open(os.path.join(PROFILES_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump({"audio_path": dest, "ref_text": ref_text.strip()}, f,
                  ensure_ascii=False, indent=2)

    profiles = list_profiles()
    return f"✅ Профіль «{name}» збережено", gr.update(choices=profiles, value=name)


def refresh_profiles():
    return gr.update(choices=list_profiles())


LANG_MAP = {"🇺🇦 Українська": "all_uk", "🇬🇧 English": "en"}


def generate(profile_name, text, lang_label):
    if not profile_name:
        return None, "❌ Оберіть профіль голосу"
    if not text or not text.strip():
        return None, "❌ Введіть текст"

    profile = load_profile(profile_name)
    if profile is None:
        return None, f"❌ Профіль «{profile_name}» не знайдено"

    ref_text   = profile["ref_text"]
    language   = LANG_MAP.get(lang_label, "en")
    ref_lang   = "all_uk" if any("Ѐ" <= c <= "ӿ" for c in ref_text) else "en"

    try:
        chunks, sr = [], 32000
        for sr, chunk in tts_backend.get_tts_wav(
            ref_wav_path    = profile["audio_path"],
            prompt_text     = ref_text,
            prompt_language = ref_lang,
            text            = text.strip(),
            text_language   = language,
            how_to_cut      = "凑四句一切",
            top_k=15, top_p=1.0, temperature=1.0,
            ref_free        = not ref_text,
            speed           = 1.0,
        ):
            chunks.append(chunk)

        if not chunks:
            return None, "❌ Порожній результат"

        audio = np.concatenate(chunks)
        # повертаємо (sample_rate, array) — Gradio Audio опрацьовує напряму
        return (sr, audio), "✅ Готово"
    except Exception as e:
        return None, f"❌ {e}"


# ─── UI ───────────────────────────────────────────────────────────────────────

CSS = """
#save-status { font-size: 1.05em; font-weight: bold; }
#gen-status  { font-size: 1.05em; font-weight: bold; }
"""

with gr.Blocks(title="GPT-SoVITS", theme=gr.themes.Soft(), css=CSS) as app:
    gr.Markdown("# 🎙 GPT-SoVITS — Клонування та синтез голосу")

    # ── Tab 1: Voice profile ──────────────────────────────────────────────────
    with gr.Tab("1 · Зліпок голосу"):
        gr.Markdown(
            "Завантажте **3–30 секунд** чистого запису голосу. "
            "Текст розпізнається автоматично — перевірте і виправте якщо потрібно."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio_in   = gr.Audio(label="Аудіо (WAV / MP3)", type="filepath")
                asr_status = gr.Textbox(label="", interactive=False, show_label=False,
                                        placeholder="Тут з'явиться розпізнана мова…")
                ref_text   = gr.Textbox(label="Розпізнаний текст (можна відредагувати)",
                                        lines=4,
                                        placeholder="Текст заповниться автоматично після завантаження…")
                name_in    = gr.Textbox(label="Назва профілю",
                                        placeholder="наприклад: Роман")
                save_btn   = gr.Button("💾  Зберегти профіль", variant="primary")
                save_status = gr.Textbox(label="", interactive=False, show_label=False,
                                         elem_id="save-status")

    # ── Tab 2: Generate ───────────────────────────────────────────────────────
    with gr.Tab("2 · Генерація голосу"):
        gr.Markdown("Оберіть збережений профіль, введіть текст і натисніть **Генерувати**.")
        with gr.Row():
            with gr.Column(scale=1):
                profile_dd  = gr.Dropdown(label="Профіль голосу",
                                          choices=list_profiles(), interactive=True)
                refresh_btn = gr.Button("🔄  Оновити список")
                lang_dd     = gr.Dropdown(label="Мова тексту",
                                          choices=["🇺🇦 Українська", "🇬🇧 English"],
                                          value="🇺🇦 Українська")
                text_in     = gr.Textbox(label="Текст для озвучення", lines=7,
                                         placeholder="Введіть текст…")
                gen_btn     = gr.Button("▶  Генерувати", variant="primary")
            with gr.Column(scale=1):
                audio_out  = gr.Audio(label="Результат", type="numpy")
                gen_status = gr.Textbox(label="", interactive=False, show_label=False,
                                        elem_id="gen-status")

    # ── Wiring ────────────────────────────────────────────────────────────────
    audio_in.change(
        on_audio_upload,
        inputs=[audio_in],
        outputs=[ref_text, asr_status],
    )
    save_btn.click(
        save_profile,
        inputs=[audio_in, ref_text, name_in],
        outputs=[save_status, profile_dd],
    )
    refresh_btn.click(refresh_profiles, outputs=[profile_dd])
    gen_btn.click(
        generate,
        inputs=[profile_dd, text_in, lang_dd],
        outputs=[audio_out, gen_status],
    )

if __name__ == "__main__":
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=9873,
        inbrowser=False,
    )
