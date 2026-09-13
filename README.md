# CNN-BiLSTM+Attention for Abnormal Behaviors Detection

A working demonstration of the paper's detector as a live service monitoring a real,
permissioned Ethereum blockchain, with the detection architecture trained on genuine
public benchmark datasets.

## What's real here

- **A real local Ethereum chain.** `chain/` is a small Hardhat-free Node project: a
  Solidity contract (`contracts/HealthcareAccessRegistry.sol`) that logs healthcare
  access events on-chain, compiled with `solc` and deployed to a local Ganache node.
  Transactions are actually submitted, mined, and read back via `web3.py` — this is a
  genuine (single-node) permissioned blockchain, not a simulation.
- **A detection architecture trained on real benchmark data.** `training/` preprocesses
  and trains the CNN + BiLSTM + domain-prior-gated attention model (matching the paper's
  Algorithm 2) on the actual UNSW-NB15, Edge-IIoTset, NSL-KDD, and TON-IoT datasets files.
- **A live backend + dashboard.** The FastAPI backend continuously submits realistic
  healthcare-access transactions to the deployed contract, reads the resulting on-chain
  events, scores each one with a trained instance of the same architecture, and streams
  the results to a browser dashboard in real time.

## Honest scope notes

- **The live on-chain feed uses realistic but synthetic role/action metadata.** There is
  no real hospital system connected, so `chain_client.py` generates plausible healthcare
  actions (role, action type, access risk flags) when submitting each on-chain
  transaction. The live detector (`backend/app/model_service.py`) is trained on this same
  schema; it is a separate model instance from the three benchmark-trained models in
  `models/`, since the benchmark datasets use a different feature schema than an on-chain
  healthcare event.


## Running it

### 1. Start the local chain
```bash
cd chain
npm install
npx ganache --port 8545 --wallet.deterministic --chain.chainId 1337 &
node compile.js
python3 deploy.py
```

### 2. (Optional) Retrain on the real benchmark datasets
The `models/` folder already contains trained weights. To reproduce or retrain:
```bash
cd training
bash fetch_real_datasets.sh
pip install -r ../backend/requirements.txt
python3 train_real_datasets.py
```

### 3. Start the backend (also serves the frontend)
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Open `http://localhost:8001`. On startup the backend trains the live on-chain detector
(roughly 20-30 seconds), then connects to the chain and starts streaming scored
transactions.

## Project layout

```
chain/
  contracts/HealthcareAccessRegistry.sol   on-chain event log
  compile.js                                solc compiler script -> build/*.json
  deploy.py                                 deploys the contract, writes deployment.json
training/
  common.py                                 shared architecture + preprocessing for real datasets
  nslkdd_columns.py                         NSL-KDD's raw column schema
  train_real_datasets.py                    trains/evaluates on datasets
  fetch_real_datasets.sh                    re-downloads the dataset files
models/                                     trained model weights (.keras)
backend/
  app/main.py                               FastAPI app: chain actor+listener, REST, WebSocket
  app/onchain_model.py                      FeatureGate, AdditiveAttention, model architecture
  app/model_service.py                      trains + scores the live on-chain detector
  app/chain_client.py                       web3.py client: submits and reads on-chain events
  app/database.py                           SQLite persistence
frontend/
  index.html, styles.css, app.js, vendor/chart.umd.js
```

## API reference

- `GET /api/events` — recent scored on-chain events (`limit`, `offset`, `only_anomalies`)
- `GET /api/events/{id}` — full detail incl. attention weights and feature-gate values
- `GET /api/alerts` — alerts (`status=open|confirmed|dismissed`)
- `POST /api/feedback` — `{ "alert_id": int, "verdict": "true_positive"|"false_positive" }`
- `GET /api/chain_status` — chain id, contract address, latest block, account count
- `GET /api/stats` — live throughput/anomaly-rate/latency and chain status
- `GET /api/audit/export` — full on-chain event trail as CSV
- `WS /ws/feed` — live event + alert stream
# CNN-BiLSTM-Attention-for-Abnormal-Behaviors-Detection
Developing a Hybrid Neural Network Model with Attention Mechanism for Effectively Abnormal Behaviors Detection in Blockchain based Healthcare System
