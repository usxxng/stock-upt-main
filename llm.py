import os
import re
import time
import json
import random
import math
import html
import hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple

from bs4 import BeautifulSoup

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Ollama (우선 시도)
try:
    from langchain_ollama import ChatOllama
except Exception:
    # 구버전 호환
    from langchain_community.chat_models import ChatOllama  # type: ignore

# Embeddings: Upstage 우선, 실패 시 HuggingFace
def _get_embeddings(provider: str):
    if provider.lower() == "upstage":
        try:
            from langchain_upstage import UpstageEmbeddings
            return UpstageEmbeddings()
        except Exception:
            pass
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import (
    OLLAMA_MODEL,
    EMBEDDING_PROVIDER,
    UPSTAGE_API_KEY,
    USER_AGENTS,
    MOBILE_USER_AGENTS,
    CSV_PATH,
    NEWS_CSV_PATH,
    NEWS_LOOKBACK_DAYS,
    NEWS_MAX_PAGES,
    NEWS_MAX_ARTICLES,
    REQUEST_TIMEOUT,
    SYSTEM_PROMPT,
    answer_examples,
    naver_date_range,
)

# -----------------------------
# Session History (기존 구조 유지)
# -----------------------------
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# -----------------------------
# LLM
# -----------------------------
def get_llm():
    # Ollama 서버는 로컬에서 `ollama serve` 실행 및 모델 pull 필요
    return ChatOllama(model=OLLAMA_MODEL, temperature=0.3, streaming=True)

# -----------------------------
# Utilities
# -----------------------------
def _headers(mobile: bool = False):
    if mobile:
        return {"User-Agent": random.choice(MOBILE_USER_AGENTS)}
    return {"User-Agent": random.choice(USER_AGENTS)}

# def _get(url: str, params=None) -> requests.Response:
#     resp = requests.get(url, params=params, headers=_headers(), timeout=REQUEST_TIMEOUT)
#     resp.raise_for_status()
#     return resp

def _get(url: str, params=None, max_retry: int = 3, backoff: float = 0.8, mobile: bool = False) -> requests.Response:
    last_err = None
    for i in range(max_retry):
        resp = None
        try:
            resp = requests.get(url, params=params, headers=_headers(mobile=mobile), timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            if resp is not None and 500 <= resp.status_code < 600:
                last_err = e
                time.sleep(backoff * (i + 1) + random.random() * 0.3)
                continue
            raise
        except Exception as e:
            last_err = e
            time.sleep(backoff * (i + 1) + random.random() * 0.3)
    if last_err:
        raise last_err

def _clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sanitize_company_name(q: str) -> str:
    """
    사용자 문장형 입력에서 '기업명'만 뽑아 종목코드 검색에 적합한 질의로 정제.
    - 질문형 어미/분석 요청어 제거
    - 특수문자 제거(한/영/숫자/공백만)
    - 가장 긴 토큰(기업명일 확률↑) 선택, 과도하게 길면 20자 제한
    """
    q = _clean_text(q)

    # 자주 등장하는 불용어(질문형 표현/분석 키워드) 제거
    stop_words = [
        "전망", "알려줘", "알려", "분석", "뉴스", "예상", "예측",
        "호재", "악재", "궁금", "가격", "주가", "리포트", "보고서",
        "해주세요", "해줘", "어때", "어떤가요", "어떄", "에 대해", "분석해줘",
        "투자", "사야", "팔아", "올라", "떨어져", "상승", "하락"
    ]
    for w in stop_words:
        # 단어 경계 기준으로 깔끔히 제거
        q = re.sub(rf"\b{re.escape(w)}\b", " ", q)

    # 한/영/숫자/공백만 남기기
    q = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", q)
    q = _clean_text(q)

    if not q:
        return q

    tokens = q.split()
    if not tokens:
        return q

    # 가장 긴 토큰을 우선(기업명일 확률이 높음)
    base = max(tokens, key=len)[:20]
    return base

# -----------------------------
# Naver News Search & Article Crawling
# -----------------------------

# 교체: fetch_article_body()
def fetch_article_body(url: str) -> Tuple[str, str]:
    """
    모바일 상세(/news/...) 또는 news.naver.com 기사 본문을 파싱.
    - 모바일 상세일 경우: 페이지 내 '원문' 혹은 news.naver.com 링크를 찾아 재요청
    - 그 외: 일반 본문 후보/메타 태그로 파싱
    """
    def _parse_generic(soup):
        candidates = [
            ("article", None),
            ("div", {"id": "newsct_article"}),
            ("div", {"id": "articeBody"}),
            ("div", {"itemprop": "articleBody"}),
            ("div", {"class": "article-body"}),
        ]
        text = ""
        for tag, attrs in candidates:
            el = soup.find(tag, attrs=attrs) if attrs else soup.find(tag)
            if el:
                text = _clean_text(el.get_text(" "))
                if len(text) > 200:
                    break
        if len(text) < 200:
            ps = [p.get_text(" ") for p in soup.find_all("p")]
            text = _clean_text(" ".join(ps))
        published = ""
        meta_time = soup.find("meta", {"property": "article:published_time"}) or soup.find("meta", {"name": "ptime"})
        if meta_time and meta_time.get("content"):
            published = meta_time["content"]
        return text, published

    # 1차 요청
    try:
        r = _get(url, mobile=True)
    except Exception:
        return "", ""

    soup = BeautifulSoup(r.text, "html.parser")

    # 모바일 상세에서 원문(news.naver.com) 링크를 찾는다
    origin_link = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "news.naver.com" in href:
            origin_link = href
            break

    # 원문이 있으면 원문으로 재요청
    if origin_link:
        try:
            r2 = _get(origin_link)
            soup2 = BeautifulSoup(r2.text, "html.parser")
            return _parse_generic(soup2)
        except Exception:
            pass

    # 원문이 없으면 현재 페이지에서 직접 파싱
    return _parse_generic(soup)

# -----------------------------
# Simple LLM-based Classification (호재/악재/중립)
# -----------------------------
def classify_with_llm(llm, text: str) -> Dict[str, Any]:
    """로컬 LLM으로 간단 분류(라벨/신뢰도/근거 1문장)"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 한국 증권 애널리스트다. 다음 뉴스 전문을 읽고 '호재' 또는 '악재' 또는 '중립' 중 하나로 분류하라."
                   "출력은 JSON 한 줄로만: {\"label\": \"호재|악재|중립\", \"confidence\": 0-100, \"reason\": \"...\"}"),
        ("human", "{text}")
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        raw = chain.invoke({"text": text})
        raw = raw.strip()
        # JSON 파싱 복원
        js = json.loads(re.findall(r"\{.*\}", raw, re.S)[0])
        label = js.get("label", "중립")
        conf = int(js.get("confidence", 50))
        reason = _clean_text(js.get("reason", ""))
    except Exception:
        # 실패 시 휴리스틱 폴백
        label, conf, reason = "중립", 50, "충분한 신뢰도로 분류하지 못했습니다."
    conf = max(0, min(100, conf))
    if label not in ("호재", "악재", "중립"):
        label = "중립"
    return {"label": label, "confidence": conf, "reason": reason}

# -----------------------------
# Naver Finance (종목코드/현재가)
# -----------------------------
def search_stock_code(query: str) -> Tuple[str, str]:
    q = str(query).strip()
    kospi_list = pd.read_csv(CSV_PATH, encoding='euc-kr')
    kospi_list["종목코드"] = kospi_list["종목코드"].str.zfill(6)
    kospi_list["종목명"] = kospi_list["종목명"].astype(str).str.strip()

    s = kospi_list.loc[kospi_list["종목명"].eq(q), "종목코드"]

    if s.empty:
        # 못 찾은 경우: 빈 코드 반환(이후 로직에서 방어)
        return "", q

    # 여러 건이면 첫 번째를 선택(필요하면 우선순위 규칙 추가 가능)
    return s.iloc[0], q

    # code = kospi_list['종목코드'].loc[kospi_list['종목명'] == q].item()
    # name = q
    # return code, name

def search_news_title(name: str) -> Tuple[str, str]:
    q = str(name).strip()
    news_list = pd.read_csv(NEWS_CSV_PATH)

    s = news_list.loc[news_list['종목명'].eq(q), "기사제목"]

    if s.empty:
        # 못 찾은 경우: 빈 코드 반환(이후 로직에서 방어)
        return ""

    # 여러 건이면 첫 번째를 선택(필요하면 우선순위 규칙 추가 가능)
    return s.iloc[0]

    # code = kospi_list['종목코드'].loc[kospi_list['종목명'] == q].item()
    # name = q
    # return code, name

def fetch_price_summary_by_keyword(code):
    """
    입력: '삼성전자' 같은 키워드
    출력: ("현재 주가 67,600원", "전일대비 2,100 하락")
    """
    if not code:
        return ("현재 주가 -원", "전일대비 -", "방향 -")

    url = f"https://finance.naver.com/item/main.naver?code={code}"

    try:
        resp = _get(url)  # 기존 유틸 사용
    except NameError:
        import requests, random
        _UAS = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36",
        ]
        resp = requests.get(url, headers={"User-Agent": random.choice(_UAS), "Referer": "https://finance.naver.com/"}, timeout=12)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # 현재가: <p class="no_today"><span class="blind">67,600</span> ...
    price = ""
    el_price = soup.select_one("p.no_today span.blind")
    if el_price:
        price = el_price.get_text(strip=True)
    else:
        # 폴백: 전체 소스에서 숫자 패턴 탐색
        m = re.search(r'<p class="no_today">.*?<span class="blind">([\d,\.]+)</span>', resp.text, re.S)
        if m:
            price = m.group(1)

    # 전일대비/방향: <p class="no_exday"> ... <span class="ico up|down">상승|하락</span> ... <span class="blind">2,100</span>
    delta = ""
    direction = ""
    ex = soup.select_one("p.no_exday")
    if ex:
        # 방향
        ico = ex.select_one(".ico")
        if ico:
            cls = " ".join(ico.get("class", []))
            if "up" in cls:
                direction = "상승"
            elif "down" in cls:
                direction = "하락"
            else:
                t = ico.get_text(strip=True)
                if "상승" in t: direction = "상승"
                elif "하락" in t: direction = "하락"

        # 변화액(숫자, % 제외)
        for s in ex.select("span.blind"):
            text = s.get_text(strip=True)
            if "%" in text:
                continue
            m = re.search(r"[\d,]+(?:\.\d+)?", text)
            if m:
                delta = m.group(0)
                break
    now_str = price
    chg_str = delta
    dir_str = direction
    #now_str = f"현재 주가 {price}원" if price else "현재 주가 -원"
    #chg_str = f"전일대비 {delta} {direction}".strip() if delta or direction else "전일대비 -"
    return now_str, chg_str, dir_str


# -----------------------------
# Build Retriever from articles
# -----------------------------
def build_retriever_from_articles(articles: List[Dict[str, Any]]):
    embs = _get_embeddings(EMBEDDING_PROVIDER)
    docs = []
    for a in articles:
        body = a.get("body", "")
        meta = {
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "press": a.get("press", ""),
            "published": a.get("published", ""),
        }
        if body:
            docs.append(Document(page_content=body[:3000], metadata=meta))
    if not docs:
        # 빈 리트리버 방어: 더미 문서
        docs = [Document(page_content="관련 기사가 충분하지 않습니다.", metadata={"title": "N/A"})]
    vs = FAISS.from_documents(docs, embs)
    return vs.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# RAG Chain 생성
# -----------------------------
def get_rag_chain(system_prompt: str, few_shots):
    llm = get_llm()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=few_shots,
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\n{context}"),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    def _chain_with_retriever(retriever):
        question_answer_chain = create_stuff_documents_chain(get_llm(), qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda sid: get_session_history(sid),
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        ).pick("answer")
        return conversational_rag_chain

    return _chain_with_retriever

# -----------------------------
# Public Entry (Streamlit에서 사용)
# -----------------------------
def get_ai_response(user_message: str):
    """
    1) 뉴스 수집·본문 크롤링
    2) LLM 분류(호재/악재/중립)
    3) 종목코드/현재가
    4) 기사로 벡터스토어 구성 → RAG 스트리밍 답변
    """
    llm = get_llm()

    query = user_message.strip()

    # 종목코드/이름 식별 (기존 sanitize + 검색 유지)
    base_q = sanitize_company_name(query)
    code, name = search_stock_code(base_q)

    # news = search_naver_news(query)
    # for n in news:
    #     body, published = fetch_article_body(n["url"])
    #     n["body"], n["published"] = body, published

    # 코드 못 찾으면 바로 종료(또는 키워드 뉴스 폴백을 원하면 기존 함수 호출)
    if not name:
        # 필요 시: 키워드 뉴스 폴백을 원하면 여길 교체
        news = []
    else:
        # 2) 코드 기반 모바일 뉴스 수집
        #news = fetch_stock_news_by_code(code, limit=NEWS_MAX_ARTICLES)
        news = search_news_title(name)


    # 3) 본문 수집
    # for n in news:
    #     body, published = fetch_article_body(n["url"])
    #     n["body"], n["published"] = body, published
    
    news = news.split('/')
    # 분류
    labeled = []
    news_res = dict(label="", confidence=0.0, reason="")
    pos, neg, neu = 0.0, 0.0, 0.0
    for n in news:
        if not n:
            continue
        res = classify_with_llm(llm, n[:3500])
        news_res["label"] = res["label"]
        news_res["confidence"] = res["confidence"]
        news_res["reason"] = res["reason"]
        labeled.append(news_res)
        weight = res["confidence"] / 100.0
        if res["label"] == "호재":
            pos += weight
        elif res["label"] == "악재":
            neg += weight
        else:
            neu += weight

    # 간단 시그널
    score = pos - neg
    if abs(score) < 0.4:
        signal = "중립"
    elif score > 0:
        signal = "상승 가능성"
    else:
        signal = "하락 가능성"
    conf_pct = int(min(90, max(10, abs(score) / (pos + neg + 1e-6) * 100))) if (pos + neg) > 0 else 50

    # 종목 코드/가격
    #price_info = fetch_stock_price_by_code(code) if code else {}
    now_price, chg_price, dir_str = fetch_price_summary_by_keyword(code)

    # 분석 요약 텍스트
    # def _top_reasons(items, want="호재", k=3):
    #     arr = [f"- {i.get('title','')} ({i.get('press','')}) — {i.get('reason','')}" for i in items if i.get("label")==want][:k]
    #     return "\n".join(arr) if arr else "- 관련 근거가 충분하지 않습니다."

    analysis = (
        f"[기업] {name or query}\n"
        f"[집계] 호재{pos:.1f} / 악재{neg:.1f} / 중립{neu:.1f}\n"
        #f"[현재가] {price_info.get('price','-')} (전일대비 {price_info.get('change','-')} / {price_info.get('rate','-')})\n"
        f"[현재가] {now_price}원 (전일대비 {chg_price} {dir_str})\n"
        f"[시그널] {signal} · 신뢰도 {conf_pct}%\n"
        #f"[호재 근거]\n{_top_reasons(labeled, '호재')}\n"
        #f"[악재 근거]\n{_top_reasons(labeled, '악재')}\n"
    )

    # 리트리버 구성 & 체인
    retriever = build_retriever_from_articles(labeled if labeled else news)
    chain_builder = get_rag_chain(SYSTEM_PROMPT.replace("{analysis}", analysis), answer_examples)
    chain = chain_builder(retriever)

    # 스트리밍 응답
    return chain.stream(
        {"input": user_message},
        config={"configurable": {"session_id": "abc123"}},
    )
