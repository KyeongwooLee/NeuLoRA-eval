#!/usr/bin/env bash
# ============================================================
# NeuLoRA — 원클릭 환경 셋업 (원격 재연결 시에도 의존성/패키지 일관 보장)
# ============================================================
# 새 머신·클라우드 워크스페이스·SSH 재연결 후 이 스크립트만 실행하면
# Python/Node 의존성이 모두 맞춰집니다.
#
# 실행: chmod +x setup.sh && ./setup.sh
#
# 완료 후:
#   source venv/bin/activate
#   cd LangGraph && uvicorn api:app --reload --port 8800
#   (다른 터미널) cd LangGraph/frontend && npm run dev
# ============================================================

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
echo "📁 프로젝트 루트: $PROJECT_ROOT"

# ────────────────────────────────────────────────────────────
# 0. Python 버전 확인 (3.10+ 필수: eval 코드의 타입 문법 호환)
# ────────────────────────────────────────────────────────────
echo ""
echo "🐍 [0/7] Python 버전 확인..."
PYTHON_CMD=""
for p in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$p" &>/dev/null; then
        v=$("$p" -c "import sys; print(sys.version_info.major, sys.version_info.minor)" 2>/dev/null) || true
        if [ -n "$v" ]; then
            maj="${v% *}"
            min="${v#* }"
            if [ "$maj" -gt 3 ] || { [ "$maj" -eq 3 ] && [ "$min" -ge 10 ]; }; then
                PYTHON_CMD="$p"
                echo "  → 사용: $p ($($p --version 2>&1))"
                break
            fi
        fi
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    echo "  ⚠️ Python 3.10+ 필요. python3.10 이상 설치 후 재실행하세요."
    exit 1
fi

# ────────────────────────────────────────────────────────────
# 1. 시스템 패키지 (apt 사용 가능 시)
# ────────────────────────────────────────────────────────────
echo ""
echo "📦 [1/7] 시스템 패키지 (선택)..."
if command -v apt-get &>/dev/null; then
    apt-get update -qq 2>/dev/null || true
else
    echo "  → apt 없음, 건너뜀"
fi

# ────────────────────────────────────────────────────────────
# 2. 가상환경 + pip 업그레이드 + 의존성 설치 (캐시 무시로 재설치 가능)
# ────────────────────────────────────────────────────────────
echo ""
echo "🐍 [2/7] Python 가상환경 및 패키지..."

VENV_DIR="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "  → venv 생성 중 ($PYTHON_CMD)..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

VENV_ACTIVATE=""
VENV_PY=""
if [ -f "$VENV_DIR/bin/activate" ]; then
    VENV_ACTIVATE="$VENV_DIR/bin/activate"
    VENV_PY="$VENV_DIR/bin/python"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    VENV_ACTIVATE="$VENV_DIR/Scripts/activate"
    VENV_PY="$VENV_DIR/Scripts/python.exe"
else
    echo "  ⚠️ 가상환경 활성화 스크립트를 찾지 못했습니다: $VENV_DIR"
    exit 1
fi

source "$VENV_ACTIVATE"

echo "  → pip/setuptools/wheel 업그레이드..."
"$VENV_PY" -m pip install --upgrade pip setuptools wheel -q

echo "  → requirements.txt 설치 (캐시 미사용으로 깨끗 설치)..."
"$VENV_PY" -m pip install --no-cache-dir -r requirements.txt -q

LPITUTOR_REQUIREMENTS=""
if [ -f "$PROJECT_ROOT/LPITutor/requirements.txt" ]; then
    LPITUTOR_REQUIREMENTS="$PROJECT_ROOT/LPITutor/requirements.txt"
elif [ -f "$PROJECT_ROOT/../LPITutor/requirements.txt" ]; then
    LPITUTOR_REQUIREMENTS="$PROJECT_ROOT/../LPITutor/requirements.txt"
fi

if [ -n "$LPITUTOR_REQUIREMENTS" ]; then
    echo "  → LPITutor requirements 설치 (B2 비교용)..."
    "$VENV_PY" -m pip install --no-cache-dir -r "$LPITUTOR_REQUIREMENTS" -q
fi

echo "  ✅ Python 패키지 설치 완료"

# ────────────────────────────────────────────────────────────
# 3. nvm + Node.js (프론트엔드 빌드)
# ────────────────────────────────────────────────────────────
echo ""
echo "🟢 [3/7] nvm + Node.js..."

export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [ ! -s "$NVM_DIR/nvm.sh" ]; then
    echo "  → nvm 설치 중..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash 2>/dev/null || true
fi
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

if ! command -v node &>/dev/null || [ "$(node -v 2>/dev/null | sed 's/v//;s/\..*//')" -lt 18 ] 2>/dev/null; then
    echo "  → Node.js 20 LTS 설치..."
    nvm install 20 2>/dev/null || true
    nvm use 20 2>/dev/null || true
    nvm alias default 20 2>/dev/null || true
fi
echo "  → Node $(node -v 2>/dev/null || echo 'n/a'), npm $(npm -v 2>/dev/null || echo 'n/a')"

# ────────────────────────────────────────────────────────────
# 4. venv activate 시 nvm 로드
# ────────────────────────────────────────────────────────────
echo ""
echo "🔗 [4/7] venv activate에 nvm 로드 연결..."
ACTIVATE_FILE="$VENV_ACTIVATE"
NVM_MARKER="# >>> nvm auto-load >>>"
if ! grep -q "$NVM_MARKER" "$ACTIVATE_FILE" 2>/dev/null; then
    cat >> "$ACTIVATE_FILE" << 'NVMEOF'

# >>> nvm auto-load >>>
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
# <<< nvm auto-load <<<
NVMEOF
    echo "  ✅ nvm 자동 로드 추가됨"
else
    echo "  → 이미 설정됨"
fi

# ────────────────────────────────────────────────────────────
# 5. 프론트엔드 npm (node_modules 재설치로 플랫폼 일치)
# ────────────────────────────────────────────────────────────
echo ""
echo "🌐 [5/7] 프론트엔드 npm 패키지..."
FRONTEND_DIR="$PROJECT_ROOT/LangGraph/frontend"
if [ -d "$FRONTEND_DIR" ]; then
    ( cd "$FRONTEND_DIR" && rm -rf node_modules package-lock.json 2>/dev/null; npm install --silent )
    echo "  ✅ npm 설치 완료"
else
    echo "  ⚠️ $FRONTEND_DIR 없음, 건너뜀"
fi

# ────────────────────────────────────────────────────────────
# 6. .env 없으면 예시에서 복사 제안
# ────────────────────────────────────────────────────────────
echo ""
echo "🔑 [6/7] .env 확인..."
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"
if [ -f "$ENV_FILE" ]; then
    echo "  ✅ .env 존재"
    for key in HF_API_KEY GENAI_API_KEY TAVILY_API_KEY LLM_MODE EMBEDDING_MODE; do
        grep -q "^${key}=" "$ENV_FILE" 2>/dev/null && echo "    ✓ $key" || echo "    ✗ $key 미설정"
    done
else
    if [ -f "$ENV_EXAMPLE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        echo "  ✅ .env.example → .env 복사됨. 값 수정 후 사용하세요."
    else
        echo "  ⚠️ .env 없음. 아래 내용으로 프로젝트 루트에 .env 파일을 만드세요:"
        echo "    HF_API_KEY=your_hf_token"
        echo "    TAVILY_API_KEY=your_tavily_key"
        echo "    LLM_MODE=api"
        echo "    EMBEDDING_MODE=local"
    fi
fi

# ────────────────────────────────────────────────────────────
# 7. import 검증 (의존성 깨짐 조기 발견)
# ────────────────────────────────────────────────────────────
echo ""
echo "✔️ [7/7] Python 의존성 검증..."
if ( cd "$PROJECT_ROOT" && source "$VENV_ACTIVATE" && "$VENV_PY" -c "
import sys
sys.path.insert(0, '.')
try:
    import rag.base
    import fastapi
    import uvicorn
    import chromadb
    import datasets
    import sentence_transformers
    import huggingface_hub
    from google import genai
    import eval.run_benchmark
    # LangGraph는 api.py 실행 시 LangGraph 디렉터리 기준으로 로드됨
    print('  → backend + eval imports OK')
except Exception as e:
    print('  ✗ import 실패:', e)
    sys.exit(1)
" ); then
    echo "  ✅ 의존성 검증 통과"
else
    echo "  ⚠️ 일부 패키지 import 실패. 위 에러 확인 후 pip install 재실행 권장."
fi

# ────────────────────────────────────────────────────────────
# 완료
# ────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "🎉 셋업 완료 (원격 재연결 후에도 이 스크립트 재실행 가능)"
echo "=============================================="
echo ""
echo "사용법:"
echo "  1. 가상환경: source $VENV_ACTIVATE"
echo "  2. 백엔드:   cd $PROJECT_ROOT/LangGraph && uvicorn api:app --reload --port 8800"
echo "  3. 프론트:   cd $PROJECT_ROOT/LangGraph/frontend && npm run dev"
echo ""
