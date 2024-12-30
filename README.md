# YouTube 音声の文字起こしと翻訳ツール

このツールは、YouTube 動画から音声をダウンロードし、Whisper モデルを使用して文字起こしを行い、必要に応じて翻訳も行うシステムです。Gemini API を利用してテキストの要約に加え、文字起こしミスの修正も行えます。

## 機能

- YouTube から音声をダウンロード
- Whisper による音声の文字起こし
- Gemini API を用いた文字起こしテキストの修正
- 修正された文字起こし結果を指定した言語に翻訳
- Gemini API によるテキストの要約 (システムプロンプトをカスタマイズ可能)

## 実行方法

1. 必要な Python ライブラリをインストールします。

   ```bash
   pip install -r requirements.txt
   ```

2. 仮想環境を作成し、アクティブ化します。

   仮想環境の作成:

   ```bash
   python -m venv venv
   ```

   仮想環境のアクティブ化:

   - **Windows (Command Prompt):**
     ```bash
     venv\Scripts\activate
     ```
   - **Windows (PowerShell):**
     ```bash
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

3. **Gemini API キーの設定:**

   - Google Cloud Platform で Gemini API を有効にし、API キーを取得してください。
   - 取得した API キーを環境変数 `GEMINI_API_KEY` に設定してください。

     ```bash
     export GEMINI_API_KEY="YOUR_API_KEY"  # macOS/Linux の場合
     set GEMINI_API_KEY="YOUR_API_KEY"     # Windows の場合
     ```

4. プロンプトファイルの準備:

   - 機能をカスタマイズするために、`prompts` ディレクトリに以下のファイルを作成し、プロンプトを記述してください。
     - `system_prompt.txt`: 要約の際のシステムプロンプト
     - `system_prompt_refine.txt`: 文字起こし修正の際のシステムプロンプト

5. プロジェクトディレクトリ内で以下を実行します。

   ```bash
   python youtube_transcriber.py
   ```

6. プログラムが起動したら、コンソールに表示されるプロンプトに従って YouTube の動画 URL を入力します。

7. プログラムが処理を行い、出力フォルダ内に結果を保存します。保存されるファイルは以下の通りです：
   - `audio.mp3`: ダウンロードした音声ファイル
   - `video_info.json`: 動画の URL とタイトル情報
   - `transcript.txt`: 元の文字起こし結果
   - `transcript_refined.txt`: LLM で修正された文字起こし結果
   - `transcript_{lang}.txt`: 翻訳（必要な場合のみ）
   - `summary.txt`: Gemini API による要約結果

## 注意事項

- 動画の文字起こしや翻訳、要約、修正にはインターネット接続が必要です。
- YouTube 動画のダウンロードには時間がかかる場合があります。
- Gemini API の利用には、API キーの設定が必要です。

## 貢献

バグ報告や改善案などは、GitHub の Issues セクションで受け付けています。
