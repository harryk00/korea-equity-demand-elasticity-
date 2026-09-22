from pathlib import Path

from korea_sd.data.kis_client import KisOpenApiClient


class Resp:
    headers = {"tr_cont": ""}

    def __init__(self, body, status_code=200, text=None):
        self._body = body
        self.status_code = status_code
        self.text = text if text is not None else str(body)

    def json(self):
        return self._body


class Session:
    def __init__(self, get_responses=None):
        self.posts = []
        self.gets = []
        self.get_responses = list(get_responses or [])

    def post(self, url, json, timeout):
        self.posts.append((url, json))
        return Resp({"access_token": "TOKEN", "expires_in": 3600})

    def get(self, url, headers, params, timeout):
        self.gets.append((url, headers, params))
        if self.get_responses:
            return self.get_responses.pop(0)
        return Resp({"rt_cd": "0", "output": [{"x": "1"}]})


def test_kis_client_auth_and_get(tmp_path):
    s = Session()
    c = KisOpenApiClient(
        "KEY",
        "SECRET",
        pause_seconds=0,
        token_cache_path=str(tmp_path / "token.json"),
        session=s,
    )
    body, headers = c.get("/example", "TR123", {"A": 1})
    assert body["output"][0]["x"] == "1"
    assert s.gets[0][1]["authorization"] == "Bearer TOKEN"
    assert s.gets[0][1]["tr_id"] == "TR123"


def test_kis_client_retries_egw00201(tmp_path):
    rate = Resp(
        {
            "rt_cd": "1",
            "msg_cd": "EGW00201",
            "msg1": "초당 거래건수를 초과하였습니다.",
        },
        status_code=500,
        text='{"rt_cd":"1","msg_cd":"EGW00201"}',
    )
    ok = Resp({"rt_cd": "0", "output": [{"x": "2"}]})
    s = Session([rate, ok])
    c = KisOpenApiClient(
        "KEY2",
        "SECRET",
        pause_seconds=0,
        rate_limit_wait_seconds=0,
        retry_jitter_seconds=0,
        rate_limit_max_retries=3,
        token_cache_path=str(tmp_path / "token.json"),
        session=s,
    )
    body, _ = c.get("/example", "TR123", {"A": 1})
    assert body["output"][0]["x"] == "2"
    assert len(s.gets) == 2


def test_kis_token_cache_reused_across_clients(tmp_path):
    cache = tmp_path / "token.json"
    first_session = Session()
    first = KisOpenApiClient(
        "CACHEKEY",
        "SECRET",
        pause_seconds=0,
        token_cache_path=str(cache),
        session=first_session,
    )
    first.get("/example", "TR123", {})
    assert len(first_session.posts) == 1
    assert cache.exists()

    second_session = Session()
    second = KisOpenApiClient(
        "CACHEKEY",
        "SECRET",
        pause_seconds=0,
        token_cache_path=str(cache),
        session=second_session,
    )
    second.get("/example", "TR123", {})
    assert len(second_session.posts) == 0
    assert second_session.gets[0][1]["authorization"] == "Bearer TOKEN"
