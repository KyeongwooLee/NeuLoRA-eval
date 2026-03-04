from abc import ABC, abstractmethod
from operator import itemgetter
import os

# LangChain Core 및 Community
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_classic import hub

# Hugging Face 관련 라이브러리 (OpenAI 대체)
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── 임베딩 설정 (ingest.py, chroma.py 등에서도 이 설정을 공유) ────────
# ⚠️ 적재(ingest)와 검색(retrieve) 시 반드시 동일한 모델을 사용해야 합니다!
EMBEDDING_MODEL = "BAAI/bge-m3"  # 임베딩 모델명
EMBEDDING_DEVICE = "auto"                         # "cpu" / "cuda" / "auto"
ANSWER_MODEL = "Qwen/Qwen2.5-14B-Instruct"


def _resolve_embedding_device():
    """PyTorch/SentenceTransformer는 'auto'를 받지 않으므로, 'auto'일 때 cuda/cpu로 변환."""
    dev = EMBEDDING_DEVICE.strip().lower()
    if dev == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return dev if dev in ("cpu", "cuda") else "cpu"


def create_embedding_vessel():
    """
    [로컬 방식] 모델을 다운로드하여 로컬에서 실행
    - 최초 실행 시 ~/.cache/huggingface/hub/ 에 모델 다운로드
    - 이후 캐시에서 로드 (오프라인 가능)
    - GPU 활용 가능 → 빠른 임베딩
    """
    device = _resolve_embedding_device()
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_embedding_api():
    """
    [API 방식] HuggingFace Inference API를 호출하여 임베딩 생성
    - 모델을 로컬에 다운로드하지 않음
    - HF_API_KEY 환경변수 필요
    - 네트워크 필요, 무료 티어는 속도 제한 있음
    """
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    api_token = os.environ.get("HF_API_KEY")
    if not api_token:
        raise ValueError("HF_API_KEY 환경 변수가 설정되지 않았습니다.")

    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=api_token,
    )


# ─── 임베딩 방식 선택 ─────────────────────────────────────────────
# 환경변수 EMBEDDING_MODE로 제어: "local" (기본) 또는 "api"
def create_embedding_auto():
    """환경변수 EMBEDDING_MODE에 따라 로컬/API 방식 자동 선택"""
    mode = os.environ.get("EMBEDDING_MODE", "local").lower()
    if mode == "api":
        print(f"🌐 임베딩: API 방식 ({EMBEDDING_MODEL})")
        return create_embedding_api()
    else:
        print(f"💻 임베딩: 로컬 방식 ({EMBEDDING_MODEL}, device={EMBEDDING_DEVICE})")
        return create_embedding_vessel()


class RetrievalChain(ABC):
    def __init__(self):
        self.source_uri = None
        self.k = 5

    @abstractmethod
    def load_documents(self, source_uris):
        """loader를 사용하여 문서를 로드합니다."""
        pass

    @abstractmethod
    def create_text_splitter(self):
        """text splitter를 생성합니다."""
        pass

    def split_documents(self, docs, text_splitter):
        """text splitter를 사용하여 문서를 분할합니다."""
        return text_splitter.split_documents(docs)

    def create_embedding(self):
        """
        Embeddings 생성 (로컬/API 자동 선택)
        - 환경변수 EMBEDDING_MODE="api" → HF Inference API 호출 (다운로드 없음)
        - 환경변수 EMBEDDING_MODE="local" 또는 미설정 → 로컬 다운로드 후 실행
        """
        return create_embedding_auto()

    def create_vectorstore(self, split_docs):
        """VectorStore 생성 (FAISS)"""
        return FAISS.from_documents(
            documents=split_docs, embedding=self.create_embedding()
        )

    def create_retriever(self, vectorstore):
        """Retriever 생성 (MMR 방식 등)"""
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        return dense_retriever

    def create_model(self, model_name: str | None = None, external_model=None):
        """
        LLM 모델 생성
        - HuggingFaceEndpoint 기반 Qwen/Qwen2.5-14B-Instruct 사용
        """
        if external_model is not None:
            return external_model

        target_model = model_name or ANSWER_MODEL

        mode = os.getenv("LLM_MODE", "").lower()
        if mode == "vessel":

            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
            from langchain_huggingface import HuggingFacePipeline

            model_kwargs = {"device_map": "auto", "dtype": "auto"}
            q4 = os.getenv("LLM_4BIT", "").lower() in ("1", "true", "yes")
            q8 = os.getenv("LLM_8BIT", "").lower() in ("1", "true", "yes")
            if q4:
                print("⚖️  4bit 양자화 사용")
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif q8:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True) # 3090 24GB 등에서 VRAM 절약
            model = AutoModelForCausalLM.from_pretrained(target_model, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(target_model)

            pipe = pipeline(
                    model=model,
                    tokenizer=tokenizer,
                    task="text-generation",
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    )
            llm = HuggingFacePipeline(pipeline=pipe)

            return ChatHuggingFace(llm=llm)

        api_token = os.environ.get("HF_API_KEY")
        if not api_token:
            raise ValueError("HF_API_KEY 환경 변수가 설정되지 않았습니다.")

        llm = HuggingFaceEndpoint(
            repo_id=target_model,
            task="text-generation",
            max_new_tokens=1024,
            temperature=0.7,
            huggingfacehub_api_token=api_token
        )
        return ChatHuggingFace(llm=llm)

    
    def create_prompt(self):
        """Prompt Template 로드"""
        # 기존 프롬프트 사용 (필요시 모델에 맞춰 변경 가능)
        return hub.pull("teddynote/rag-prompt-chat-history")

    def create_prompt_new(self):
        system_template = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Be kind and friendly when user says life question. "
            "And answer the question based on the policy.\n\n"
            "Policy: {policy}\n\n"
            "keep answer concise. And answer in Korean if user asks in Korean.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        return prompt

    @staticmethod
    def format_docs(docs):
        return "\n".join([doc.page_content for doc in docs])

    def create_chain(self):
        """RAG 체인 구성"""
        # 1. 문서 로드
        docs = self.load_documents(self.source_uri)
        
        # 2. 텍스트 분할
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)
        
        # 3. 벡터 저장소 및 리트리버 생성
        self.vectorstore = self.create_vectorstore(split_docs)
        self.retriever = self.create_retriever(self.vectorstore)
        
        # 4. 모델 및 프롬프트 준비
        model = self.create_model()
        prompt = self.create_prompt_new()
        
        # 5. 체인 연결
        self.chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("context"),
                "chat_history": itemgetter("chat_history"),
                "policy": itemgetter("policy"),
            }
            | prompt
            | model
            | StrOutputParser()
        )
        return self
