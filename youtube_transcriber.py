import json
import logging
import os
from datetime import datetime
from typing import Tuple

import whisper
import yt_dlp
from googletrans import Translator

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 定数定義
OUTPUT_DIR = "./output/"
VIDEO_INFO_FILENAME = "video_info.json"
WHISPER_MODEL_SIZE = "medium"
TRANSCRIPT_FILENAME = "transcript.txt"
AUDIO_FILENAME = "audio.mp3"
TARGET_LANGUAGE = "ja"


def create_output_folder() -> str:
    """
    タイムスタンプに基づいた新しい出力フォルダを作成します。

    Returns:
        str: 作成されたフォルダのパス。
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        folder_path = os.path.join(OUTPUT_DIR, timestamp)
        os.makedirs(folder_path, exist_ok=True)
        logging.info(f"出力フォルダ {folder_path} が作成されました。")
        return folder_path
    except Exception as e:
        logging.error(f"出力フォルダの作成に失敗しました: {e}")
        raise


def download_audio(url: str, folder_path: str) -> Tuple[str, str]:
    """
    YouTubeから動画の音声をダウンロードし、ファイルパスとタイトルを返します。

    Args:
        url (str): ダウンロードするYouTube動画のURL。
        folder_path (str): 音声ファイルの保存先フォルダのパス。

    Returns:
        Tuple[str, str]: 音声ファイルのパスと動画のタイトル。

    Raises:
        Exception: ダウンロードエラー時に発生。
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
            logging.info("音声のダウンロードを開始します。")
            info_dict = ydl.extract_info(url, download=True)
            title = info_dict.get("title", "Unknown Title")
            audio_path = os.path.join(folder_path, AUDIO_FILENAME)
            logging.info(f"音声のダウンロードが完了しました。タイトル: {title}")
            return audio_path, title
    except Exception as e:
        logging.error(f"音声ダウンロードに失敗しました: {e}")
        raise


def save_video_info(url: str, title: str, folder_path: str) -> None:
    """
    YouTube動画の情報をJSON形式で保存します。

    Args:
        url (str): YouTube動画のURL。
        title (str): 動画のタイトル。
        folder_path (str): 保存先フォルダのパス。

    Raises:
        IOError: ファイルの書き込みエラー時に発生。
    """
    video_info = {"url": url, "title": title}
    info_path = os.path.join(folder_path, VIDEO_INFO_FILENAME)
    try:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(video_info, f, ensure_ascii=False, indent=4)
            logging.info(f"動画情報を{VIDEO_INFO_FILENAME}に保存しました。")
    except IOError as e:
        logging.error(f"動画情報の保存に失敗しました: {e}")
        raise


def transcribe_audio(audio_path: str, folder_path: str) -> Tuple[str, str]:
    """
    音声データを文字起こしし、文字起こしテキストと言語を返します。

    Args:
        audio_path (str): 音声ファイルのパス。
        folder_path (str): 文字起こし結果の保存先フォルダのパス。

    Returns:
        Tuple[str, str]: 文字起こし結果のテキストと検出された言語。

    Raises:
        RuntimeError: Whisperモデルの読み込みまたは文字起こしエラー時に発生。
    """
    try:
        logging.info(f"Whisperモデル（{WHISPER_MODEL_SIZE}）の読み込みを開始します。")
        model = whisper.load_model(WHISPER_MODEL_SIZE)
        logging.info("音声の文字起こしを開始します。")
        result = model.transcribe(audio_path)
        transcript_path = os.path.join(folder_path, TRANSCRIPT_FILENAME)
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
            logging.info(f"文字起こし結果を{TRANSCRIPT_FILENAME}に保存しました。")
        return result["text"], result["language"]
    except Exception as e:
        logging.error(f"文字起こしに失敗しました: {e}")
        raise RuntimeError("Whisperモデルの文字起こしに失敗しました。")


def translate_text(text: str, target_lang: str, folder_path: str) -> None:
    """
    指定された言語に文字起こしテキストを翻訳し、保存します。

    Args:
        text (str): 翻訳するテキスト。
        target_lang (str): 目標言語（ISO 639-1形式）。
        folder_path (str): 翻訳結果の保存先フォルダのパス。

    Raises:
        ValueError: 翻訳エンジンエラー時に発生。
    """
    try:
        translator = Translator()
        logging.info(f"{target_lang}への翻訳を開始します。")
        translated_text = translator.translate(text, dest=target_lang).text
        transcript_translated_path = os.path.join(
            folder_path, f"transcript_{target_lang}.txt"
        )
        with open(transcript_translated_path, "w", encoding="utf-8") as f:
            f.write(translated_text)
            logging.info(f"翻訳結果をtranscript_{target_lang}.txtに保存しました。")
    except Exception as e:
        logging.error(f"翻訳に失敗しました: {e}")
        raise ValueError("翻訳処理に失敗しました。")


def main() -> None:
    """
    メイン処理: ユーザー入力に基づいてYouTubeの音声をダウンロード、文字起こし、翻訳を実行します。
    """
    try:
        url = input("YouTubeのURLを入力してください: ")
        logging.info(f"入力されたURL: {url}")

        folder_path = create_output_folder()

        audio_path, title = download_audio(url, folder_path)
        save_video_info(url, title, folder_path)

        text, detected_lang = transcribe_audio(audio_path, folder_path)

        if detected_lang != TARGET_LANGUAGE:
            translate_text(text, TARGET_LANGUAGE, folder_path)

        logging.info(
            f"処理が完了しました。結果は出力フォルダ({folder_path})に保存されています。"
        )
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
