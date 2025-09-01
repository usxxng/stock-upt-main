import os
from datetime import datetime, timedelta

# --------------------------------------
# Models & Embeddings (LLM)
# --------------------------------------
# Ollama Local Model Name
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Upstage Embedding (권장) / 없으면 HuggingFace 로컬 임베딩으로 자동 Fallback
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")  # "upstage" | "huggingface"
UPSTAGE_EMBEDDING = os.getenv("UPSTAGE_EMBEDDING", "upstage-embedding")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY", "up_qPxMRSVUJxnuDb1Jd5FPxr6Y8eMHM")

# --------------------------------------
# Naver News Crawling
# --------------------------------------
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", 7))  # 크롤링할 뉴스의 최대 일수 (기본: 7일)
NEWS_MAX_PAGES = int(os.getenv("NEWS_MAX_PAGES", 2))  # 크롤링할 뉴스의 최대 페이지 수 (기본: 2페이지)
NEWS_MAX_ARTICLES = int(os.getenv("NEWS_MAX_ARTICLES", 20))  # 크롤링할 뉴스의 최대 기사 수 (기본: 20개)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 10))  # HTTP 요청 타임아웃 (기본: 10초)

# 가벼운 UA 로테이션
USER_AGENTS = [
    # 필요한 만큼 추가 가능
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36",
]

MOBILE_USER_AGENTS = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 13; SM-S918N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Mobile Safari/537.36",
]

CSV_PATH = os.getenv("KOSPI_CSV_PATH", os.path.join(os.path.dirname(__file__), "kospi_250901.csv"))
NEWS_CSV_PATH = os.getenv("NEWS_CSV_PATH", os.path.join(os.path.dirname(__file__), "news_titles_top10_sample.csv"))


# -----------------------------
# Few-shot (간단 예시) & System Prompt
# -----------------------------
answer_examples = [  # 기존 인터페이스 유지용 이름
    {
        "input": "삼성전자 최근 뉴스 기반으로 호재/악재 요약해줘.",
        "answer": "최근 3일 기사 다수에서 AI/HBM 수요 증가가 언급되어 호재 비중이 높습니다. "
                  "다만 공급망 변수도 있어 단기 과열 가능성은 체크가 필요합니다."
    },
    {
        "input": "하이브 전망 알려줘.",
        "answer": "컴백/투어 모멘텀과 IP 파이프라인 확대로 호재 기사 비중이 큽니다. 환율·소비 둔화는 리스크로 언급됩니다."
    },
]

SYSTEM_PROMPT = """너는 한국 주식 애널리스트다. 제공된 기사 컨텍스트와 분석요약(analysis)을 바탕으로
- (1) 최근 이슈 요약(불릿), (2) 호재/악재 집계와 핵심 근거, (3) 현재가/전일대비/등락률, 
- (4) 단기 전망[상승 가능성/하락 가능성/중립] + 신뢰도(%)를 간결히 제시한다.
투자조언이 아님을 마지막에 명시한다.
답변은 한국어로만, 과장 없이 사실 중심으로.
{analysis}
"""

# -----------------------------
# Helper
# -----------------------------
def naver_date_range(days: int):
    end = datetime.now()
    start = end - timedelta(days=days)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


# answer_examples = [
#     {
#         "input": "소득은 어떻게 구분되나요?", 
#         "answer": """소득세법 제 4조(소득의 구분)에 따르면 소득은 아래와 같이 구분됩니다.
# 1. 종합소득
#     - 이 법에 따라 과세되는 모든 소득에서 제2호 및 제3호에 따른 소득을 제외한 소득으로서 다음 각 목의 소득을 합산한 것
#     - 가. 이자소득
#     - 나. 배당소득
#     - 다. 사업소득
#     - 라. 근로소득
#     - 마. 연금소득
#     - 바. 기타소득
# 2. 퇴직소득
# 3. 양도소득
# """
#     },
#     {
#         "input": "소득세의 과세 기간은 어떻게 되나요?", 
#         "answer": """소득세법 제5조(과세기간)에 따르면, 
# 일반적인 소득세의 과세기간은 1월 1일부터 12월 31일까지 1년입니다
# 하지만 거주자가 사망한 경우는 1월 1일부터 사망일까지, 
# 거주자가 해외로 이주한 경우 1월 1일부터 출국한 날까지 입니다"""
#     },
#     {
#         "input": "원천징수 영수증은 언제 발급받을 수 있나요?", 
#         "answer": """소득세법 제143조(근로소득에 대한 원천징수영수증의 발급)에 따르면, 
# 근로소득을 지급하는 원천징수의무자는 해당 과세기간의 다음 연도 2월 말일까지 원천징수영수증을 근로소득자에게 발급해야하고. 
# 다만, 해당 과세기간 중도에 퇴직한 사람에게는 퇴직한 한 날의 다음 달 말일까지 발급하여야 하며, 
# 일용근로자에 대하여는 근로소득의 지급일이 속하는 달의 다음 달 말일까지 발급하여야 합니다.
# 만약 퇴사자가 원청징수영수증을 요청한다면 지체없이 바로 발급해야 합니다"""
#     },
# ]