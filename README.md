# study-mlops

# WSL + Node.js + DuckDB + dbt + scikit-learn + MLflow + ONNX + Node.js Serve + GitHub CI/CD 構成ドキュメント

---

## 全体構成図

```
【ログファイル】 logs/*.json
        ↓
【DuckDB】 project.duckdb
        ↓
【dbt】 stg_* モデル経由 → feature_*
        ↓
【scikit-learn】 train.py: RandomForest実行
        ↓
【MLflow】 ログ、model.onnx エクスポート
        ↓
【Node.js REST】 index.js: ONNX 推論 API
        ↓
【GitHub Actions】 dbt run → train.py → model.onnx → npm start
```

---

## ログ形式 (logs/access\_\*.json)

```json
{
  "timestamp": "2024-11-10T14:05:22Z",
  "user_id": 12345,
  "event_type": "view_item",
  "item_id": "ABC123",
  "category": "electronics",
  "device": "mobile",
  "location": "tokyo",
  "referrer": "google"
}
```

---

## DWH: DuckDB + dbt

### stg\_access\_logs

| カラム         | 型         | 内容     |
| ----------- | --------- | ------ |
| user\_id    | INTEGER   | ユーザーID |
| event\_type | TEXT      | イベント種別 |
| timestamp   | TIMESTAMP | 時刻     |

### feature\_user\_behavior.sql (dbt)

```sql
{{ config(materialized='view') }}
SELECT
  user_id,
  COUNT(*) AS total_events,
  COUNTIF(event_type = 'click') AS total_clicks,
  MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS target
FROM {{ ref('stg_access_logs') }}
GROUP BY user_id
```

---

## モデル構築 (train.py)

```python
import duckdb, pandas as pd, mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 特徴量ビュー
df = duckdb.query("SELECT * FROM feature_user_behavior").to_df()
X, y = df.drop(columns=["target"]), df["target"]

with mlflow.start_run():
    model = RandomForestClassifier().fit(X, y)
    mlflow.log_metric("accuracy", accuracy_score(y, model.predict(X)))

    onnx_model = convert_sklearn(model, [("input", FloatTensorType([None, X.shape[1]]))])
    with open("model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
```

---

## Node.js REST API (predict-server/index.js)

```javascript
const express = require('express');
const bodyParser = require('body-parser');
const ort = require('onnxruntime-node');

const app = express();
const PORT = 8080;
app.use(bodyParser.json());

let session = null;
(async () => {
  try {
    session = await ort.InferenceSession.create('./model.onnx');
    console.log('ONNX model loaded');
  } catch (err) {
    console.error('Failed to load model:', err);
    process.exit(1);
  }
})();

app.post('/predict', async (req, res) => {
  if (!session) return res.status(500).json({ error: 'Model not loaded' });

  try {
    const input = req.body.input;
    const tensor = new ort.Tensor('float32', Float32Array.from(input), [1, input.length]);
    const results = await session.run({ input: tensor });
    const output = results[session.outputNames[0]].data;
    res.json({ prediction: output });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => console.log(`ONNX REST server running on port ${PORT}`));
```

---

## GitHub Actions (.github/workflows/train.yml)

```yaml
name: Train and Export Model

on:
  push:
    paths:
      - '**.py'
      - '**.sql'
      - '**/predict-server/**'

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: dbt run
      - run: python train.py
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: cd predict-server && npm install
```

---

## 得られる利点

| 項目    | 内容                                   |
| ----- | ------------------------------------ |
| コスト   | WSL/ローカルで無償開発                        |
| 再現性   | dbt + MLflow で利用調査可                  |
| 実用性   | REST API + ONNX = Go/Node.js/他にも展開自由 |
| CI/CD | GitHub Actions で動的化                  |

---

## オプション

* dbtモデルテンプレート
* MLflow ローカルログ設定
* predict-server/ディレクトリ まとめ
* ZIP/レポジトリ形式での展開 OK
