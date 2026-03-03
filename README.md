# NeuLoRA 🧠✨

**The Thinking Tutor that Connects to Your Neuron**

LangGraph 기반 멀티턴 RAG 챗봇입니다. PDF/TXT 문서 적재 후, 대화 맥락 + 벡터 검색 + (필요 시) 웹 검색을 조합해 답변하며, **Multi-LoRA**(direct/socratic/scaffolding/feedback) 스타일 라우팅을 지원합니다.

---

## 🚀 주요 기능

- 📚 **문서 RAG**: ChromaDB + LangChain/LangGraph 검색-생성 파이프라인
- 🧭 **질문 라우팅**: 검색 필요 여부 분기, 관련성 평가 후 웹 검색(Tavily) 보강
- 🧠 **대화 기억**: 10턴 단위 저장·정리, policy 기반 답변 방향
- 🎚️ **Multi-LoRA**: 쿼리별 스타일 라우팅 (marimmo/multi-lora 또는 RiverWon/NeuLoRA-*)
- 💬 **웹 UI**: FastAPI + React(Vite), KaTeX 수식 렌더링

---

## 🗂️ 프로젝트 구조

```text
NeuLoRA/
├── LangGraph/
│   ├── LangGraph.py      # 메인 그래프(노드/분기/초기화/답변 정제)
│   ├── api.py            # FastAPI 서버 (chat, upload, status, documents, reset)
│   ├── stream.py         # Streamlit 데모
│   ├── router_model.json # LoRA 스타일 centroid (라우팅용)
│   ├── chroma_db/       # Chroma 로컬 저장소
│   └── frontend/         # React + Vite
│       ├── src/
│       │   ├── App.jsx
│       │   ├── main.jsx
│       │   └── components/   # ChatArea, Sidebar, Toast
│       ├── package.json
│       └── vite.config.js
├── rag/
│   ├── base.py           # 임베딩/RetrievalChain/ANSWER_MODEL
│   ├── chroma.py         # ChromaDB 기반 RAG 체인
│   ├── ingest.py         # PDF/TXT → ChromaDB 적재
│   ├── utils.py          # format_docs 등
│   └── graph_utils.py    # random_uuid, stream_graph 등
├── requirements.txt
├── setup.sh              # 원클릭 환경 셋업 (의존성·패키지 일관 보장)
├── .env.example          # 환경 변수 예시 (복사 후 .env 로 사용)
└── README.md
```

---

## ⚙️ 기술 스택

- **Backend**: FastAPI, LangGraph, LangChain
- **Frontend**: React, Vite, KaTeX (remark-math, rehype-katex)
- **Vector DB**: ChromaDB
- **LLM**: Qwen2.5-14B-Instruct (라우팅·답변), Hugging Face API 또는 로컬(vessel)
- **임베딩**: BAAI/bge-m3 (로컬/API)
- **PEFT**: marimmo/multi-lora (direct/socratic/scaffolding/feedback) 또는 RiverWon/NeuLoRA-*

---

## 🔧 한 번에 셋업 (원격 재연결 시에도 권장)

의존성·라이브러리·패키지를 한 번에 맞추려면 **setup.sh** 한 번 실행하면 됩니다.

```bash
chmod +x setup.sh
./setup.sh
```

이 스크립트는 다음을 수행합니다.

- Python 3.8+ 확인, 가상환경 생성
- pip 업그레이드 후 `requirements.txt` 설치 (캐시 미사용 권장)
- nvm + Node.js 20 LTS, venv 활성화 시 nvm 자동 로드
- 프론트엔드 `npm install` (node_modules 재설치)
- `.env` 없으면 `.env.example` 복사 제안
- Python import 검증 (rag, fastapi, uvicorn)

---

## 🔐 환경 변수 (.env)

프로젝트 루트에 `.env` 파일을 두고 아래 변수를 설정하세요. 없으면 `cp .env.example .env` 후 값만 채우면 됩니다.

| 변수 | 필수 | 설명 |
|------|------|------|
| `HF_API_KEY` | ✅ | Hugging Face 토큰 (API 호출·모델/어댑터 다운로드) |
| `TAVILY_API_KEY` | 선택 | 웹 검색 보강 |
| `EMBEDDING_MODE` | 선택 | `local`(기본) / `api` |
| `LLM_MODE` | 선택 | `api`(기본, 단일 GPU 권장) / `vessel`(라우팅+답변 둘 다 로컬) |
| `LLM_QUANT` | 선택 | `8bit`(기본) / `4bit` |
| `LLM_4BIT` / `LLM_8BIT` | 선택 | vessel 시 4bit/8bit 양자화 |
| `LLM_CPU_OFFLOAD` | 선택 | VRAM 부족 시 `1`로 8bit 일부 CPU 오프로드 |
| `ATTN_IMPLEMENTATION` | 선택 | `sdpa`(기본) / `flash_attention_2` |

**단일 24GB GPU**에서는 `LLM_MODE=api`로 두고, 답변용 14B만 로컬에서 돌리는 구성을 권장합니다.

---

## 🏃 실행 방법

### 1) 셋업 (최초 또는 원격 재연결 후)

```bash
./setup.sh
source venv/bin/activate
```

### 2) 백엔드

```bash
cd LangGraph
uvicorn api:app --reload --port 8800
```

원격에서 접속할 때:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8800
```

### 3) 프론트엔드 (별도 터미널)

```bash
cd LangGraph/frontend
npm run dev
```

원격 접속 시:

```bash
npm run dev -- --host
```

- 프론트: `http://localhost:5173` (또는 `http://<서버IP>:5173`)
- API: `http://localhost:8800` (또는 `http://<서버IP>:8800`)

### 4) Streamlit 데모 (선택)

```bash
cd LangGraph
streamlit run stream.py
```

---

## 🧩 API

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/api/chat` | 질문 전송, 답변 반환 (`message`, 선택 `thread_id`) |
| `POST` | `/api/upload` | PDF/TXT 업로드 및 ChromaDB 적재 |
| `GET` | `/api/status` | 모델/연결 상태 |
| `GET` | `/api/documents` | 컬렉션·문서 개수 |
| `POST` | `/api/reset` | 새 세션 ID 발급 |

---

## 📊 Benchmark Evaluation (NeuLoRA/eval)

아키텍처 비교(B0/B1/B2/P) 및 어댑터 고정 비교를 실행할 수 있습니다.

```bash
cd NeuLoRA
python -m eval.run_benchmark \
  --mode architecture \
  --benchmarks comta,mathdial,mtbench,gsm8k \
  --max-samples 200 \
  --seed 42 \
  --judge-model gemini-2.5-flash
```

어댑터 고정 비교:

```bash
python -m eval.run_benchmark --mode adapters
```

필수 환경변수:
- `HF_API_KEY` (모델 호출)
- `GENAI_API_KEY` (Gemini judge)
- 선택: `BASE_MODEL_NAME`, `EMBEDDING_MODEL_NAME`

출력:
- `eval/outputs/<mode>/<benchmark>__<variant>.json`
- `eval/outputs/<mode>/summary.json`

---

## 🤖 모델 설정

- **라우팅/판단/요약**: `Qwen/Qwen2.5-14B-Instruct` (API 또는 vessel)
- **답변 생성**: 동일 14B + PEFT (marimmo/multi-lora 또는 RiverWon/NeuLoRA-direct 등)
- **임베딩**: `BAAI/bge-m3`

모델/경로 변경은 `LangGraph/LangGraph.py`의 `ROUTER_MODEL`, `CHAIN_MODEL`, `STYLE_MODELS`, `PEFT_REPO` 및 `rag/base.py`의 `ANSWER_MODEL`, `EMBEDDING_MODEL`을 수정하면 됩니다.

---

## 📌 참고

- `.env`는 Git에 포함하지 마세요.
- ChromaDB 데이터는 `LangGraph/chroma_db/`에 저장됩니다.
- 원격 재연결 후에는 `./setup.sh`를 다시 실행해 두면 의존성·패키지 문제를 줄일 수 있습니다.
