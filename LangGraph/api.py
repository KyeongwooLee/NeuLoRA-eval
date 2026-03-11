"""
api.py – FastAPI 백엔드
======================
LangGraph.py 파이프라인을 REST API 로 노출.
React 프론트엔드(frontend/)에서 호출합니다.

실행:
    cd LangGraph
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --reload --port 8800
"""

import os
import sys
import tempfile
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── 프로젝트 루트 path ──
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import LangGraph as lg


# ============================================================
# FastAPI 앱 + 라이프사이클
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 파이프라인 초기화"""
    await asyncio.to_thread(lg.initialize)
    await asyncio.to_thread(lg.build_app)
    # 초기화 로그 버림 (서버 콘솔에 이미 출력됨)
    lg.get_and_clear_logs()
    yield


app = FastAPI(title="RAG Chat API", lifespan=lifespan)

# CORS: React 개발 서버 (Vite 기본 포트)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 요청 / 응답 모델
# ============================================================

class ChatRequest(BaseModel):
    message: str
    #같은 thread_id를 쓰면 이전 대화 내역을 유지하면서 질문 가능
    #{ "message": "그럼 장점은?", "thread_id": "abc-123" }
    thread_id: str | None = None
    forced_style: str | None = None
    variant: str | None = None
    router: str | None = None
    disable_web_search: bool | None = None
    disable_relevance_check: bool | None = None


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    logs: list[str]
    variant: str | None = None
    applied_style: str | None = None
    style_source: str | None = None
    adapter_switched: bool | None = None


class StatusResponse(BaseModel):
    models: dict
    connections: dict


class DocumentCollection(BaseModel):
    name: str
    count: int


class DocumentsResponse(BaseModel):
    collections: list[DocumentCollection]


class ResetResponse(BaseModel):
    thread_id: str


# ============================================================
# 엔드포인트
# ============================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """사용자 메시지를 받아 RAG 파이프라인 실행 후 답변 반환"""
    thread_id = req.thread_id or lg.random_uuid()
    forced_style = (req.forced_style or "").strip().lower() or None
    variant = (req.variant or "").strip().upper() or None
    if forced_style and forced_style not in lg.STYLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unsupported forced_style: {forced_style}",
                "available": list(lg.STYLE_MODELS.keys()),
            },
        )
    if variant and variant not in {"B", "B0", "B1", "B2", "P"}:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unsupported variant: {variant}",
                "available": ["B", "B0", "B1", "B2", "P"],
            },
        )
    router_mode = (req.router or "").strip().lower() or None
    if router_mode and router_mode not in {"cent", "mlp"}:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unsupported router mode: {router_mode}",
                "available": ["cent", "mlp"],
            },
        )

    # LLM 호출은 동기 → 별도 스레드에서 실행
    try:
        result = await asyncio.to_thread(
            lg.query,
            question=req.message,
            thread_id=thread_id,
            forced_style=forced_style,
            variant=variant,
            router=router_mode,
            disable_web_search=req.disable_web_search,
            disable_relevance_check=req.disable_relevance_check,
        )
        logs = lg.get_and_clear_logs()
    except Exception as e:
        logs = lg.get_and_clear_logs()
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(e),
                "logs": logs[-30:],
            },
        ) from e

    return ChatResponse(
        answer=result.get("answer", "답변을 생성하지 못했습니다."),
        thread_id=thread_id,
        logs=logs,
        variant=result.get("variant"),
        applied_style=result.get("applied_style"),
        style_source=result.get("style_source"),
        adapter_switched=result.get("adapter_switched"),
    )


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """PDF 또는 TXT 파일을 ChromaDB 에 적재"""
    suffix = Path(file.filename or "file.txt").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        await asyncio.to_thread(lg.ingest_uploaded_file, tmp_path)
        logs = lg.get_and_clear_logs()
        return {"status": "ok", "filename": file.filename, "logs": logs}
    except Exception as e:
        logs = lg.get_and_clear_logs()
        return {"status": "error", "error": str(e), "logs": logs}
    finally:
        os.unlink(tmp_path)


@app.get("/api/status", response_model=StatusResponse)
async def status():
    """시스템 상태 (모델명, 연결 상태) 반환"""
    return StatusResponse(
        models={
            "answer_llm": lg.CHAIN_MODEL,
            "router_llm": lg.ROUTER_MODEL,
            "embedding": lg.EMBEDDING_MODEL,
        },
        connections={
            "huggingface": bool(os.getenv("HF_API_KEY")),
            "tavily": bool(os.getenv("TAVILY_API_KEY")),
            "chromadb": Path(lg.PERSIST_DIR).exists(),
            "pipeline": lg.is_initialized(),
        },
    )


@app.get("/api/documents", response_model=DocumentsResponse)
async def documents():
    """ChromaDB 컬렉션 목록 + 문서 수 반환"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=lg.PERSIST_DIR)
        cols = client.list_collections()
        return DocumentsResponse(
            collections=[
                DocumentCollection(name=c.name, count=c.count()) for c in cols
            ]
        )
    except Exception:
        return DocumentsResponse(collections=[])


@app.post("/api/reset", response_model=ResetResponse)
async def reset():
    """새 대화 세션 ID 발급"""
    return ResetResponse(thread_id=lg.random_uuid())
