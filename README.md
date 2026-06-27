# bedrock_agent

Amazon Bedrock（Claude）とLangChainを使った、PDF向けRAGの最小サンプルです。

## 前提

- Python 3.11+
- AWS認証情報は `boto3` の標準認証チェーン（環境変数 / AWS CLI設定 / IAMロールなど）

## ディレクトリ構成

```text
project/
├── data/
│   └── sample.pdf
├── outputs/
├── src/
│   ├── main.py
│   ├── pdf_loader.py
│   ├── rag.py
│   ├── bedrock_client.py
│   └── config.py
├── .env.example
├── requirements.txt
└── README.md
```

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` の値を必要に応じて変更してください。

## 実行方法

`data/` にPDFを配置したうえで実行します。

```bash
python src/main.py --question "このPDFの要点を教えてください"
```

実行後、回答は `outputs/answer.txt` に保存されます。回答には参照したPDFファイル名とページ番号が含まれます。

## 実装概要

- `data/` 配下の全PDFを `pypdf`（`PyPDFLoader`）で読み込み
- テキストをチャンク分割
- Bedrock Embeddingsでベクトル化しFAISSを構築
- 質問に対して関連チャンクを検索
- Claude（Bedrock）で回答生成
- 参照ソース（`ファイル名:ページ番号`）を回答に追記

## エラーハンドリング

- `data/` フォルダがない、またはPDFが1件もない場合は分かりやすいエラーを表示
- `outputs/` がない場合は自動作成
