# MLB選手 WAR予測プロジェクト

## 概要 (Description)
このプロジェクトは，MLB（メジャーリーグベースボール）選手の過去の成績データに基づき，翌シーズンの「WAR (Wins Above Replacement)」を予測する機械学習モデルを構築することを目的としています。

選手の総合的な貢献度を示すWARを予測することで，どの選手が来シーズンにおいてチームの勝利に大きく貢献するかを定量的に評価することを目指します。

## プロジェクトの目的
* 過去のパフォーマンスデータ（打撃，投球，守備など）を用いた翌シーズンWARの予測（回帰モデル）。
* どの指標（特徴量）がWARの予測に最も強く影響するか（特徴量重要度）を分析する。
* 選手の年齢曲線（エイジングカーブ）がパフォーマンスに与える影響を考察する。

## 使用技術 (Tools & Libraries)
* **言語**: Python 3.x
* **データ取得**: `pybaseball`
* **データ処理**: `pandas`
* **機械学習**: `scikit-learn`
* **可視化**: `matplotlib`, `seaborn`
* **開発環境**: Jupyter Notebook / Jupyter Lab （推奨）

## 🚀 セットアップ (Setup)

### 1. 仮想環境の構築
本プロジェクトでは，Pythonの仮想環境 (`.venv`) の使用を推奨します。

```bash
# プロジェクトディレクトリを作成（例: mlb_war_project）
mkdir mlb_war_project
cd mlb_war_project

# ".venv" という名前の仮想環境を作成
python3 -m venv .venv

# 仮想環境をアクティベート（有効化）
# (Mac/Linux)
source .venv/bin/activate
# (Windows)
# .\.venv\Scripts\activate
```

### 2. 必要なライブラリのインストール

仮想環境をアクティベートした状態で，必要なライブラリをインストールします。

#### 方法1: requirements.txtを使用（推奨）

プロジェクトに含まれている `requirements.txt` を使用して，必要なライブラリを一括でインストールします。

```bash
# pipを最新化
pip install --upgrade pip

# requirements.txtに記載されているすべてのライブラリをインストール
pip install -r requirements.txt
```

この方法により，プロジェクトで使用するすべてのライブラリとその依存関係が，正しいバージョンでインストールされます。

#### 方法2: 個別にインストール

`requirements.txt` がない場合や，最小限のパッケージのみをインストールしたい場合は，以下のコマンドを使用します。

```bash
# pipを最新化
pip install --upgrade pip

# 必要なライブラリを個別にインストール
pip install pybaseball pandas scikit-learn matplotlib seaborn jupyter
```

## 📊 分析フロー（予定）

1.  **データ収集**: `pybaseball` を使用し，FanGraphsなどから過去のMLB選手のシーズン成績（WARを含む）を取得する。
2.  **データ前処理**: 「翌シーズンのWAR」を目的変数(Y)とし，「当シーズンまでの成績」を説明変数(X)とする学習データセットを作成する。
3.  **モデル構築**: 線形回帰，ランダムフォレスト回帰などのモデルを試行する。
4.  **モデル評価**: テストデータを用いて，モデルの予測精度（RMSEなど）を評価する。
5.  **考察**: モデルの結果を解釈し，特徴量の重要度などを分析する。
