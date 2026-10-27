# TH-022: 実データに基づく最小評価パッケージ

2026-10-24 / 社内検証用 v0.1 / 科学的プロトコルは未承認

TH-022を実装に進めるため、実データの正解抽出、適格者の棚卸し、被験者単位の分割、提出CSVの検証・採点を実装した。Python標準ライブラリのみで動く。既存サイトの設定や公開コンペは変更しない。

## この施策を選んだ理由

10月19日の依頼は一つのテーマでデータ取得→提出→評価の成立を試すことだった。実際の企画書では予測対象列が未確定で、回帰課題に分類指標の実装を要求する不整合もあった。まず一つの評価経路を実行可能にすれば、エンジニアの着手条件を具体化し、研究協力者がレビューでき、将来のテーマにも入力検証と評価処理を再利用できる。これは確認できた資料に基づく優先順位の判断であり、売上や需要を検証した結果ではない。

## 今回実際に確認したこと

OpenNeuro ds008037 v1.0.0 の公式GitHubミラーを取得し、HEADが `98931f03b602a7685d63f88a85b71dfd35872a11` であることを確認した。

| 項目 | 実測結果 |
|---|---:|
| participants.tsv の人数 | 123 |
| BDS行動ファイルあり | 118 |
| 有効な終了行 b_exit あり | 117 |
| 前頭前野EEG .set の掲載あり | 108 |
| BDS終了値・前頭前野EEG掲載・events.tsv の全条件を満たす | 103 |
| 暫定固定分割 | train 83 / holdout 20 |
| この実行で取得・検証したEEG本体 | 0 |

除外20人の理由は重複ありで、BDS欠損5、BDS終了行なし1、前頭前野EEG未掲載15、events欠損16。EEGはgit-annex参照の存在を確認しただけで、信号品質を確認した人数ではない。条件を変えれば対象人数も変わる。

19件のユニットテストと、SciPy 1.17.0に対する100組のランダム入力による相関値照合を実行済み。提出→採点の通し実行は合成データで実施した。実EEGによる予測性能、第三者再実行、顧客需要、収益はいずれも未検証。

## すぐ動かす

ZIPで受け取った場合は、展開した `th022` フォルダで次の2つを実行する。

```bash
python3 -m unittest discover -s . -v
python3 demo.py
```

以下はこのリポジトリをcloneし、リポジトリのルートで実行する手順。Python 3.10以上。追加インストール不要。

```bash
python3 -m unittest discover -s th022 -v
python3 th022/demo.py
git clone --depth 1 --branch 1.0.0 https://github.com/OpenNeuroDatasets/ds008037.git /tmp/th022-ds008037
git -C /tmp/th022-ds008037 rev-parse HEAD
python3 th022/benchmark.py prepare --dataset /tmp/th022-ds008037 --output /tmp/th022-prepared --freeze-split
```

初回の `--freeze-split` は、その時点の適格者と割り当てを `organizer_only/split_manifest.json` に保存する。再実行時は必ずこのファイルを指定する。

```bash
python3 th022/benchmark.py prepare --dataset /tmp/th022-ds008037 --output /tmp/th022-rerun --split-manifest /tmp/th022-prepared/organizer_only/split_manifest.json
```

固定対象者の欠損・無効化、または新たな適格者の追加がある場合は、対象IDと理由を報告して出力前に停止する。分割を自動で再計算しない。初回に不完全なデータで固定しないよう、適格者数と除外理由を確認し、確定したmanifestを再実行者へ渡す。入力の内容・正解値の同一性まではこのmanifestでは検証しないため、audit.jsonの元データハッシュも照合する。

出力先は未作成のディレクトリを指定する。再実行時は新しい出力先を使う。cloneには通信が必要だが、prepareとscoreはローカルのみ。prepareはDOIを照合するがGitのHEADまでは検証しないため、上記rev-parseの結果を固定SHAと比較する。元ファイルのSHA-256は監査記録に残す。

- `train_labels.csv`: 学習用の被験者IDと正解。
- `holdout_ids.csv`: 評価対象IDだけ。
- `organizer_only/labels.csv`: 主催者用の全正解・分割。
- `organizer_only/split_manifest.json`: 正解値を含まない固定対象者・分割。再実行に使用。
- `organizer_only/audit.json`: 除外理由、元ファイルのハッシュ、EEG実体の有無。

`organizer_only` は参加者配布物に含めない。ただし元データの正解は公開されており、この分割では秘匿テストを実現できない。

## 暫定評価契約

**目的変数:** `sub-*/beh/*_task-bds_beh.tsv` の、唯一かつ最終行の `trialcode=b_exit` にある `mean_span`。途中行のゼロを平均しない。これは逆唱課題の推定スパンであり、「一般的なワーキングメモリ容量」を検証済みの尺度として主張しない。別日測定であることも本監査では確認していない。

**入力候補:** 前頭前野への単発TMS-EEGのみ。安静時EEG・WM課題EEG・行動データ・年齢等は予測器の入力に含めない。今回、EEG読み込み・アーチファクト除去・特徴抽出は実装していない。

**分割:** IDと固定seed `th022-v0.1` のSHA-256順で約20%をholdoutにする。初回に対象者と割り当てを保存し、以後は保存済みの分割を使用する。被験者は重複させない。これは単一holdoutでありleave-one-subject-out交差検証ではない。前処理の学習・モデル選択・符号選択はtrain内で完結させる。

**提出CSV:** 各holdout被験者につき1行、次の列を持つ。

```csv
participant_id,prediction,half_a,half_b
```

`prediction` は学習データだけで方向を決めた予測値。`half_a/half_b` は同じ固定手法で、主催者が事前指定した互いに重ならないTMS試行集合から得た指標。試行の分け方・前処理・対象時間窓は専門家レビュー後に固定するため、この版は本番採点には使わない。

```bash
python3 th022/benchmark.py score --labels /tmp/th022-prepared/organizer_only/labels.csv --submission submission.csv
```

予測Spearman相関と、被験者を観測単位とする半分A/BのPearson相関を別々に返す。後者は各被験者の中で二つのスカラー同士の相関を計算するものではない。重み付き総合点・合格閾値・順位は未定義で、出力は常に `rankable=false`。一定値のベクトルは相関未定義としてnullを返す。負の予測相関は符号を維持し、絶対値で評価しない。

IDはholdoutと完全一致を要求し、欠損・重複・余分なID・trainの混入・非有限値を拒否する。入力ファイルのハッシュも返す。ただしCSVだけではhalf列のコピー、不正な特徴量、公開正解の参照は検出できない。コード提出と主催者側再実行が必要。

## 元企画書からの修正点

| 元の記載・未定義点 | 本パッケージの扱い |
|---|---|
| 123名をそのまま利用できるように読める | 暫定条件の交差集合103名。品質確認は未了 |
| 目的変数の列が未確定 | BDSの終了行mean_spanを検証用候補として実装 |
| 分類精度・macro F1の追加実装が必要 | 回帰・順位相関課題として実装 |
| holdoutとleave-subject-outの混同 | 単一の被験者holdoutと明記 |
| CC0とAllen Instituteの非商用条件が混在 | 公式dataset_descriptionのLicenseはCC0。Allen規約を本データに適用する根拠は本資料にはない |
| 公開ラベルを非公開評価に利用 | 探索的な公開ベンチマークと位置づける。秘匿性を主張しない |

## 次に解除する一つのゲート

**研究担当者による「BDS終了値を目的変数にする妥当性とTMS前処理・分割仕様」の確定。** 確定後、EEG実体を取得し、固定ベースラインを103名候補に適用する。成功条件は第三者が同じ入力・コードから同じ評価結果を再生成すること。仕様が変わる場合はv0.2として再分割・記録し、v0.1と性能を混同しない。103名は科学的・臨床的有用性や十分な検出力の保証ではない。

## 一次資料

- [元企画書 TH-022](https://chatgpt.com/library?file_id=libfile_d04528dba6648191b511ac85e01e7bd1&account_id=d880a24b-78e4-4b4a-9ba5-df1650cc9a04)（所有者のアクセス権が必要）
- [OpenNeuro v1.0.0](https://openneuro.org/datasets/ds008037/versions/1.0.0)
- [公式README](https://github.com/OpenNeuroDatasets/ds008037/blob/98931f03b602a7685d63f88a85b71dfd35872a11/README.md)
- [データセット情報](https://github.com/OpenNeuroDatasets/ds008037/blob/98931f03b602a7685d63f88a85b71dfd35872a11/dataset_description.json)
- [BDS列定義](https://github.com/OpenNeuroDatasets/ds008037/blob/98931f03b602a7685d63f88a85b71dfd35872a11/sub-001/beh/sub-001_task-bds_beh.json)
- [BDSデータ例](https://github.com/OpenNeuroDatasets/ds008037/blob/98931f03b602a7685d63f88a85b71dfd35872a11/sub-001/beh/sub-001_task-bds_beh.tsv)
- [前頭前野TMS-EEG情報](https://github.com/OpenNeuroDatasets/ds008037/blob/98931f03b602a7685d63f88a85b71dfd35872a11/sub-001/eeg/sub-001_task-tmseeg_acq-prefrontal_eeg.json)
