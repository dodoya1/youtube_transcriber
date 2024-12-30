import json
import logging
import os
from datetime import datetime
from typing import Optional, Tuple

import google.generativeai as genai
import torch
import whisper
import yt_dlp
from dotenv import load_dotenv
from googletrans import Translator

load_dotenv()

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)

# 定数定義
OUTPUT_DIR = "./output/"
PROMPTS_DIR = "./prompts/"
VIDEO_INFO_FILENAME = "video_info.json"
WHISPER_MODEL_SIZE = "small"
TRANSCRIPT_FILENAME = "transcript.txt"
TRANSCRIPT_REFINED_FILENAME = "transcript_refined.txt"
TRANSLATED_TRANSCRIPT_FILENAME = "transcript_{lang}.txt"
SUMMARY_FILENAME = "summary.txt"
AUDIO_FILENAME = "audio.mp3"
TARGET_LANGUAGE = "ja"
SYSTEM_PROMPT_FILENAME = "system_prompt.txt"
SYSTEM_PROMPT_REFINE_FILENAME = "system_prompt_refine.txt"

# Gemini API の設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("Gemini APIキーが設定されていません。環境変数 'GEMINI_API_KEY' を設定してください。")
    raise ValueError("Gemini APIキーが設定されていません")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-1.5-flash"  # または "gemini-1.5-pro" など


def create_output_folder() -> str:
    """タイムスタンプに基づいた新しい出力フォルダを作成します。

    Returns:
        str: 作成された出力フォルダのパス。
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_path = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(folder_path, exist_ok=True)
    logging.info(f"出力フォルダを作成しました: {folder_path}")
    return folder_path


def download_audio(url: str, folder_path: str) -> Tuple[str, str]:
    """YouTube動画のURLから音声をダウンロードします。

    Args:
        url (str): YouTube動画のURL。
        folder_path (str): 音声ファイルを保存するフォルダのパス。

    Returns:
        Tuple[str, str]: ダウンロードした音声ファイルのパスと動画のタイトル。

    Raises:
        Exception: 音声のダウンロードに失敗した場合。
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(folder_path, "audio.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logging.info(f"URL: {url} の音声ダウンロードを開始します。")
            info_dict = ydl.extract_info(url, download=True)
            title = info_dict.get("title", "Unknown Title")
            audio_path = os.path.join(folder_path, AUDIO_FILENAME)
            logging.info(f"音声のダウンロードが完了しました。タイトル: {title}, 保存先: {audio_path}")
            return audio_path, title
    except Exception as e:
        logging.error(f"音声ダウンロードに失敗しました: {e}")
        raise


def _save_text_to_file(filepath: str, text: str) -> None:
    """テキストを指定したファイルに保存するヘルパー関数です。

    Args:
        filepath (str): 保存先のファイルのパス。
        text (str): 保存するテキスト。

    Raises:
        IOError: ファイルへの書き込みに失敗した場合。
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"内容を保存しました: {filepath}")
    except IOError as e:
        logging.error(f"ファイルへの書き込みに失敗しました: {filepath} - {e}")
        raise


def save_video_info(url: str, title: str, folder_path: str) -> None:
    """YouTube動画の情報をJSONファイルに保存します。

    Args:
        url (str): YouTube動画のURL。
        title (str): YouTube動画のタイトル。
        folder_path (str): JSONファイルを保存するフォルダのパス。

    Raises:
        IOError: 動画情報の保存に失敗した場合。
    """
    video_info = {"url": url, "title": title}
    info_path = os.path.join(folder_path, VIDEO_INFO_FILENAME)
    try:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(video_info, f, ensure_ascii=False, indent=4)
        logging.info(f"動画情報を保存しました: {info_path}")
    except IOError as e:
        logging.error(f"動画情報の保存に失敗しました: {e}")
        raise


def transcribe_audio(audio_path: str, folder_path: str) -> Tuple[str, Optional[str]]:
    """Whisperモデルを用いて音声データを文字起こしします。

    Args:
        audio_path (str): 文字起こしする音声ファイルのパス。
        folder_path (str): 文字起こし結果を保存するフォルダのパス。

    Returns:
        Tuple[str, Optional[str]]: 文字起こしされたテキストと検出された言語（もしあれば）。

    Raises:
        RuntimeError: 音声の文字起こしに失敗した場合。
    """
    try:
        logging.info(f"Whisperモデル（{WHISPER_MODEL_SIZE}）の読み込みを開始します。")
        device = "cpu"  # 強制的に CPU を使用
        logging.info(f"使用デバイス (強制): {device}")
        model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
        logging.info(f"音声ファイルの文字起こしを開始します: {audio_path}")
        result = model.transcribe(audio_path)
        transcript_path = os.path.join(folder_path, TRANSCRIPT_FILENAME)
        _save_text_to_file(transcript_path, result["text"])
        return result["text"], result.get("language")
    except Exception as e:
        logging.error(f"文字起こしに失敗しました: {e}")
        raise RuntimeError("Whisperモデルの文字起こしに失敗しました。")


def refine_transcript_with_llm(text: str, folder_path: str) -> str:
    """言語モデル（Gemini）を用いて文字起こしテキストを修正します。

    Args:
        text (str): 修正する文字起こしテキスト。
        folder_path (str): 修正されたテキストを保存するフォルダのパス。

    Returns:
        str: 修正された文字起こしテキスト。

    Raises:
        ValueError: LLMによる文字起こし修正に失敗した場合。
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        system_prompt_path = os.path.join(
            PROMPTS_DIR, SYSTEM_PROMPT_REFINE_FILENAME)
        system_prompt = load_prompt(system_prompt_path)

        prompt = f"{system_prompt}\n\n---\n\n{text}"

        logging.info("Gemini API による文字起こし修正を開始します。")
        response = model.generate_content(prompt)
        refined_text = response.text
        refined_transcript_path = os.path.join(
            folder_path, TRANSCRIPT_REFINED_FILENAME)
        _save_text_to_file(refined_transcript_path, refined_text)
        return refined_text
    except Exception as e:
        logging.error(f"Gemini API による文字起こし修正に失敗しました: {e}")
        raise ValueError("LLMによる文字起こし修正に失敗しました。")


def translate_text(text: str, target_lang: str, folder_path: str) -> None:
    """指定された言語にテキストを翻訳します。

    Args:
        text (str): 翻訳するテキスト。
        target_lang (str): 翻訳先の言語コード（例: 'ja', 'en'）。
        folder_path (str): 翻訳結果を保存するフォルダのパス。

    Raises:
        ValueError: 翻訳処理に失敗した場合。
    """
    try:
        translator = Translator()
        logging.info(f"{target_lang}への翻訳を開始します。")
        translated_text = translator.translate(text, dest=target_lang).text
        transcript_translated_path = os.path.join(
            folder_path, TRANSLATED_TRANSCRIPT_FILENAME.format(
                lang=target_lang)
        )
        _save_text_to_file(transcript_translated_path, translated_text)
    except Exception as e:
        logging.error(f"翻訳に失敗しました: {e}")
        raise ValueError("翻訳処理に失敗しました。")


def load_prompt(filepath: str) -> str:
    """指定されたファイルからプロンプトを読み込みます。

    Args:
        filepath (str): プロンプトファイルのパス。

    Returns:
        str: プロンプトファイルの内容。
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.error(f"プロンプトファイルが見つかりません: {filepath}")
        return ""
    except Exception as e:
        logging.error(f"プロンプトファイルの読み込みに失敗しました: {e}")
        return ""


def summarize_text_with_gemini(text: str, folder_path: str) -> None:
    """Gemini APIを用いてテキストを要約します。

    Args:
        text (str): 要約するテキスト。
        folder_path (str): 要約結果を保存するフォルダのパス。
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        system_prompt_path = os.path.join(PROMPTS_DIR, SYSTEM_PROMPT_FILENAME)
        system_prompt = load_prompt(system_prompt_path)

        prompt = f"{system_prompt}\n\n---\n\n{text}"

        logging.info("Gemini API による要約を開始します。")
        response = model.generate_content(prompt)
        summary = response.text
        summary_path = os.path.join(folder_path, SUMMARY_FILENAME)
        _save_text_to_file(summary_path, summary)
    except Exception as e:
        logging.error(f"Gemini API による要約に失敗しました: {e}")


def main() -> None:
    """YouTube音声のダウンロード、文字起こし、修正、翻訳、要約処理を実行するメイン関数です。"""
    try:
        url = input("YouTubeのURLを入力してください: ")
        logging.info(f"入力されたURL: {url}")

        folder_path = create_output_folder()

        audio_path, title = download_audio(url, folder_path)
        save_video_info(url, title, folder_path)

        raw_text, detected_lang = transcribe_audio(audio_path, folder_path)

        # LLMで文字起こしを修正
        refined_text = refine_transcript_with_llm(raw_text, folder_path)

        if detected_lang and detected_lang != TARGET_LANGUAGE:
            translate_text(refined_text, TARGET_LANGUAGE, folder_path)

        summarize_text_with_gemini(refined_text, folder_path)

        logging.info(f"処理が完了しました。結果は出力フォルダに保存されています: {folder_path}")
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
