"""自治体の例規集 (条例・規則の全文) を収集し、1 行 1 文 (条・項・号の段落単位) のコーパスを作る。

e-Gov 法令と Wikipedia だけでは自治体の条例に特有の言い回し (「個人の市民税」「納税通知書」等) が
足りないため、ぎょうせい「Super Reiki-Base」形式と第一法規「d1w_reiki」形式で公開されている
例規集から本文ページを取得して学習コーパスに加える。

使い方:
  python3 scripts/fetch_reiki.py                 # 収集 → 抽出 (data/corpus/reiki.txt を書く)
  python3 scripts/fetch_reiki.py --probe         # 各サイトの目次だけ辿って本文ページ数を表示
  python3 scripts/fetch_reiki.py --extract-only  # ネットワークに出ず、キャッシュから抽出だけやり直す
  python3 scripts/fetch_reiki.py --list          # 対象自治体の一覧
  python3 scripts/fetch_reiki.py --sites 小樽市,堺市 --time-limit 600

取得のルール (礼儀正しいクローリング):
  - robots.txt を守る (Crawl-delay があればそれに従う。robots.txt が 5xx/取得不能なら全面禁止とみなす)
  - 同一ホストへのリクエストは 1 秒に 1 回以下 (ホスト単位のロックで直列化)。異なるホストは並行
  - User-Agent に研究目的であることを明記する
  - 失敗 (404 を含む) が連続したらその自治体は諦める
  - 取得した HTML は ~/celso-data/reiki_html/<自治体>/ にキャッシュし、再実行時は再取得しない

評価データ (横浜市市税条例) の漏洩を防ぐため、横浜市の例規は絶対に取得・出力しない
(サイト表への登録拒否 + 全リクエストのホスト検査 + 出力時の題名検査の三重のガード)。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(__file__))
from textnorm import kanji_numbers_to_arabic, norm  # noqa: E402

USER_AGENT = (
    "celso-reiki-research/0.1 (non-commercial research crawler for a Japanese typo checker; "
    "+https://github.com/bokuweb/celso; 1 req/s per host)"
)
UA_TOKEN = "celso-reiki-research"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.expanduser("~/celso-data/reiki_html")
DEFAULT_OUT = os.path.join(REPO, "data", "corpus", "reiki.txt")

# 評価データ漏洩防止。名前・ホスト・題名のどれかに当たったら取得も出力もしない。
BANNED_NAME = re.compile(r"横浜")
BANNED_HOST = re.compile(r"yokohama", re.I)


@dataclass
class Site:
    pref: str
    name: str
    kind: str  # "srb" (Super Reiki-Base) / "d1w" (第一法規)
    url: str  # srb: reiki_menu.html / d1w: reiki.html
    category: str  # 政令市 / 中核市 / 一般市 / 特別区 / 町村
    enabled: bool = True  # False は予備 (--all で対象に含める)

    @property
    def sid(self) -> str:
        return f"{self.pref}_{self.name}"


# 全国自治体例規集リンク集 (https://www.rilg.or.jp/htdocs/main/zenkoku_reiki/zenkoku_Link.html) から、
# 例規集を自前ホストで公開している自治体を地域・規模がばらけるように選んだ。
# g-reiki.net / d1-law.com のような共用ホストは 1 ホスト 1 req/s の制約で並行できないので、
# 小樽市 (g-reiki.net) を除き自前ホストの自治体を優先している。
# 末尾が False の行は予備 (既定では取得しない。--all で含める)。
# 2026-10 時点で除外したもの: 旭川市・十和田市 (robots.txt で例規集が Disallow)、
# 鳴門市 (ドメイン移転先の robots.txt を取得できず)、千曲市・鳥取市 (リンク集の URL が 404)、
# 北杜市 (目次から本文ページへのリンクを抽出できない形式)。横浜市は評価データなので絶対に入れない。
SITES: list[Site] = [
    # 北海道・東北
    Site("北海道", "札幌市", "d1w", "https://www.city.sapporo.jp/ncms/reiki/d1w_reiki/reiki.html", "政令市"),
    Site("北海道", "小樽市", "srb", "https://www1.g-reiki.net/city.otaru/reiki_menu.html", "一般市"),
    Site("北海道", "夕張市", "srb", "https://www.city.yubari.lg.jp/contents/illustrative/reiki_int/reiki_menu.html", "一般市", False),
    Site("北海道", "美幌町", "srb", "http://bousai.town.bihoro.hokkaido.jp/reiki/reiki_menu.html", "町村"),
    Site("青森県", "青森市", "srb", "https://www.city.aomori.aomori.jp/area/reiki/reiki_menu.html", "中核市"),
    Site("岩手県", "奥州市", "srb", "https://www.city.oshu.iwate.jp/section/reiki_int/reiki_menu.html", "一般市"),
    Site("岩手県", "矢巾町", "srb", "https://www.town.yahaba.iwate.jp/contents/18reiki/reiki_menu.html", "町村", False),
    Site("宮城県", "大崎市", "srb", "https://www.city.osaki.miyagi.jp/section/reiki/reiki_menu.html", "一般市", False),
    Site("宮城県", "利府町", "srb", "https://www.town.rifu.miyagi.jp/section/reiki/reiki_int/reiki_menu.html", "町村"),
    Site("秋田県", "秋田市", "srb", "https://www.city.akita.akita.jp/city/gn/dc/reiki/reiki_menu.html", "中核市", False),
    Site("福島県", "会津若松市", "srb", "https://www.city.aizuwakamatsu.fukushima.jp/j/reiki_int/reiki_menu.html", "一般市", False),
    # 関東
    Site("茨城県", "取手市", "srb", "http://reiki.city.toride.ibaraki.jp/reiki_menu.html", "一般市"),
    Site("群馬県", "高崎市", "srb", "https://www.city.takasaki.gunma.jp/reiki/reiki_menu.html", "中核市"),
    Site("群馬県", "草津町", "srb", "http://www.town.kusatsu.gunma.jp/reiki/reiki_menu.html", "町村", False),
    Site("埼玉県", "川越市", "srb", "https://www.city.kawagoe.saitama.jp/reiki_int/reiki_menu.html", "中核市", False),
    Site("東京都", "品川区", "d1w", "https://www.city.shinagawa.tokyo.jp/reiki/reiki_menu.html", "特別区"),
    Site("東京都", "小平市", "srb", "http://www.city.kodaira.tokyo.jp/reiki/reiki_menu.html", "一般市"),
    Site("東京都", "小笠原村", "srb", "https://www.vill.ogasawara.tokyo.jp/reiki_int/reiki_menu.html", "町村", False),
    # 中部
    Site("新潟県", "佐渡市", "srb", "https://www.city.sado.niigata.jp/reiki_int/reiki_menu.html", "一般市"),
    Site("長野県", "原村", "srb", "https://www.vill.hara.lg.jp/reiki/reiki_int/reiki_menu.html", "町村"),
    Site("岐阜県", "大垣市", "srb", "http://www2.city.ogaki.lg.jp/reiki_int/reiki_menu.html", "一般市", False),
    Site("静岡県", "藤枝市", "srb", "https://www.city.fujieda.shizuoka.jp/static/reiki_int/reiki_menu.html", "一般市", False),
    Site("愛知県", "一宮市", "srb", "https://www2.city.ichinomiya.aichi.jp/reiki/reiki_menu.html", "中核市", False),
    Site("愛知県", "豊田市", "srb", "https://www2.city.toyota.aichi.jp/reiki_int/reiki_menu.html", "中核市"),
    # 近畿
    Site("三重県", "伊勢市", "srb", "http://reikisyu.city.ise.mie.jp/ise/reiki_menu.html", "一般市"),
    Site("滋賀県", "守山市", "srb", "https://www2.city.moriyama.lg.jp/reiki_int/reiki_menu.html", "一般市", False),
    Site("京都府", "舞鶴市", "srb", "https://www.city.maizuru.kyoto.jp/html/reiki_int/reiki_menu.html", "一般市"),
    Site("大阪府", "堺市", "srb", "https://www.city.sakai.lg.jp/reiki/reiki_menu.html", "政令市"),
    Site("大阪府", "吹田市", "d1w", "https://www2.city.suita.osaka.jp/reiki/d1w_reiki/reiki.html", "中核市"),
    Site("大阪府", "高槻市", "srb", "https://www.city.takatsuki.osaka.jp/bunsyo/reiki_int/reiki_menu.html", "中核市", False),
    Site("大阪府", "寝屋川市", "srb", "http://www2.city.neyagawa.osaka.jp/reiki/reiki_menu.html", "中核市", False),
    Site("兵庫県", "加古川市", "srb", "https://www.city.kakogawa.lg.jp/section/reiki_int/reiki_menu.html", "一般市", False),
    Site("奈良県", "生駒市", "srb", "https://www.city.ikoma.lg.jp/reiki/reiki_menu.html", "一般市", False),
    Site("奈良県", "斑鳩町", "srb", "https://www.town.ikaruga.nara.jp/reiki_int/reiki_menu.html", "町村"),
    Site("和歌山県", "橋本市", "srb", "https://www.city.hashimoto.lg.jp/section/reiki_menu.html", "一般市", False),
    Site("和歌山県", "串本町", "srb", "https://www.town.kushimoto.wakayama.jp/reiki_int/reiki_menu.html", "町村", False),
    # 中国・四国
    Site("鳥取県", "倉吉市", "d1w", "https://www.city.kurayoshi.lg.jp/d1w_reiki/reiki.html", "一般市"),
    Site("鳥取県", "智頭町", "srb", "https://www.chizutown.jp/contents/reiki_202602/reiki_menu.html", "町村"),
    Site("岡山県", "笠岡市", "srb", "https://www.city.kasaoka.okayama.jp/reiki_int/reiki_menu.html", "一般市", False),
    Site("広島県", "福山市", "srb", "https://www.city.fukuyama.hiroshima.jp/soumu/reiki_int/reiki_menu.html", "中核市"),
    Site("山口県", "萩市", "srb", "https://www.city.hagi.lg.jp/reiki/reiki_menu.html", "一般市"),
    Site("香川県", "観音寺市", "d1w", "https://www.city.kanonji.kagawa.jp/d1w_reiki/reiki.html", "一般市"),
    Site("愛媛県", "新居浜市", "srb", "https://www.city.niihama.lg.jp/kouhou/reiki_int/reiki_menu.html", "一般市", False),
    Site("高知県", "南国市", "srb", "https://www.city.nankoku.lg.jp/reiki/reiki_menu.html", "一般市", False),
    # 九州・沖縄
    Site("福岡県", "福岡市", "srb", "http://www.city.fukuoka.lg.jp/d1w_reiki/reiki.html", "政令市"),
    Site("福岡県", "久留米市", "srb", "https://www1.city.kurume.fukuoka.jp/reiki_int/reiki_menu.html", "中核市"),
    Site("佐賀県", "武雄市", "srb", "https://www.city.takeo.lg.jp/reiki/reiki_menu.html", "一般市", False),
    Site("長崎県", "対馬市", "srb", "https://www.city.tsushima.nagasaki.jp/section/reiki_int/reiki_menu.html", "一般市"),
    Site("熊本県", "天草市", "srb", "http://www2.city.amakusa.kumamoto.jp/reiki/reiki_menu.html", "一般市", False),
    Site("熊本県", "甲佐町", "srb", "https://www.town.kosa.lg.jp/reiki_int/reiki_menu.html", "町村", False),
    Site("大分県", "中津市", "d1w", "https://www.city-nakatsu.jp/d1w_reiki/reiki.html", "一般市"),
    Site("大分県", "佐伯市", "srb", "https://www.city.saiki.oita.jp/reiki/reiki_menu.html", "一般市", False),
    Site("宮崎県", "高千穂町", "srb", "https://www.town-takachiho.jp/section/reiki_int/reiki_menu.html", "町村"),
    Site("鹿児島県", "鹿児島市", "srb", "http://g-reiki.city.kagoshima.lg.jp/kagoshima2/reiki_menu.html", "中核市"),
    Site("沖縄県", "北谷町", "srb", "https://www.chatan.jp/reiki/reiki_menu.html", "町村"),
]

for _s in SITES:
    # サイト表に横浜市が紛れ込んだら起動時点で止める
    assert not BANNED_NAME.search(_s.name) and not BANNED_HOST.search(_s.url), _s


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S ") + msg
    print(line, file=sys.stderr, flush=True)
    if LOG_FILE:
        with LOG_LOCK:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")


# 共用ホストは他自治体の利用者とも帯域を分け合うので、最初から間隔を広げておく
# (www1.g-reiki.net は 1.1 秒間隔でも 429 を返した)
HOST_INTERVAL = {"www1.g-reiki.net": 4.0}

LOG_FILE: str | None = None
LOG_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 取得 (rate limit / robots.txt / キャッシュ)
# ---------------------------------------------------------------------------


class FetchError(Exception):
    def __init__(self, status: int | None, msg: str):
        super().__init__(msg)
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # リダイレクト先も robots.txt とホスト間隔の対象にしたいので、自動追従はしない
    def redirect_request(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


class Fetcher:
    """ホストごとに直列化し、前回リクエスト完了から interval 秒以上空けて取得する。"""

    def __init__(self, interval: float, offline: bool):
        self.interval = interval
        self.offline = offline
        self._locks: dict[str, threading.Lock] = {}
        self._next: dict[str, float] = {}
        self._host_interval: dict[str, float] = dict(HOST_INTERVAL)
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._meta = threading.Lock()
        self.requests = 0

    def _lock(self, host: str) -> threading.Lock:
        with self._meta:
            if host not in self._locks:
                self._locks[host] = threading.Lock()
            return self._locks[host]

    def _raw_get(self, url: str) -> tuple[int, bytes, dict[str, str]]:
        host = urllib.parse.urlsplit(url).netloc.lower()
        if BANNED_HOST.search(host):
            raise FetchError(None, f"banned host {host}")
        if self.offline:
            raise FetchError(None, "offline")
        with self._lock(host):
            wait = self._next.get(host, 0.0) - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "ja"})
            try:
                with self._meta:
                    self.requests += 1
                try:
                    with _OPENER.open(req, timeout=30) as r:
                        return r.status, r.read(), {k.lower(): v for k, v in r.headers.items()}
                except urllib.error.HTTPError as e:
                    body = e.read() if e.fp else b""
                    return e.code, body, {k.lower(): v for k, v in (e.headers or {}).items()}
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                    raise FetchError(None, f"{type(e).__name__}: {e}") from e
            finally:
                iv = max(self.interval, self._host_interval.get(host, 0.0))
                self._next[host] = time.monotonic() + iv

    def robots(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        """None は「全面禁止」(robots.txt が 401/403/5xx/取得不能)。"""
        sp = urllib.parse.urlsplit(url)
        key = f"{sp.scheme}://{sp.netloc.lower()}"
        with self._meta:
            if key in self._robots:
                return self._robots[key]
        rp = urllib.robotparser.RobotFileParser()
        robots_url = key + "/robots.txt"
        result: urllib.robotparser.RobotFileParser | None
        try:
            status, body, headers = self._raw_get(robots_url)
            # robots.txt 自体のリダイレクト (http→https、ドメイン移転等) は RFC 9309 に従い 5 段まで追う
            loc = robots_url
            for _ in range(5):
                if status not in (301, 302, 303, 307, 308) or not headers.get("location"):
                    break
                loc = urllib.parse.urljoin(loc, headers["location"])
                status, body, headers = self._raw_get(loc)
            if status == 200:
                text = body.decode("utf-8", errors="ignore")
                # HTML のエラーページを robots.txt として返すサーバがあるので、その場合は「なし」扱い
                if "<html" in text[:500].lower():
                    rp.parse([])
                else:
                    rp.parse(text.splitlines())
                result = rp
            elif status in (401, 403):
                result = None
            elif 400 <= status < 500:
                rp.parse([])  # robots.txt なし → 全面許可
                result = rp
            else:
                result = None
        except FetchError as e:
            if self.offline:
                rp.parse([])
                result = rp
            else:
                log(f"robots.txt 取得失敗 {robots_url}: {e}")
                result = None
        if result is not None:
            delay = result.crawl_delay(UA_TOKEN)
            if delay:
                h = sp.netloc.lower()
                self._host_interval[h] = max(self._host_interval.get(h, 0.0), min(float(delay), 30.0))
        with self._meta:
            self._robots[key] = result
        return result

    def allowed(self, url: str) -> bool:
        if self.offline:
            return True
        rp = self.robots(url)
        return rp is not None and rp.can_fetch(UA_TOKEN, url)

    def get(self, url: str, max_redirects: int = 5) -> tuple[str, bytes]:
        """(最終 URL, 本文) を返す。リダイレクトは各段で robots.txt を確認しつつ追う。"""
        for _ in range(max_redirects + 1):
            if not self.allowed(url):
                raise FetchError(None, f"robots.txt disallow: {url}")
            attempt = 0
            while True:
                status, body, headers = self._raw_get(url)
                if status in (429, 503):
                    # 混雑・レート制限を返されたら、そのホストの間隔を倍にして (上限 30 秒) 以後も遅くする
                    host = urllib.parse.urlsplit(url).netloc.lower()
                    with self._meta:
                        cur = max(self.interval, self._host_interval.get(host, 0.0))
                        self._host_interval[host] = min(cur * 2, 30.0)
                    log(f"HTTP {status}: {host} の間隔を {self._host_interval[host]:.1f}s に広げる")
                    if attempt < 2:
                        ra = headers.get("retry-after", "")
                        time.sleep(min(int(ra), 120) if ra.isdigit() else 30 * (attempt + 1))
                        attempt += 1
                        continue
                if status >= 500 and attempt < 1:
                    time.sleep(10)
                    attempt += 1
                    continue
                break
            if status in (301, 302, 303, 307, 308) and headers.get("location"):
                url = urllib.parse.urljoin(url, headers["location"])
                continue
            if status != 200:
                raise FetchError(status, f"HTTP {status}: {url}")
            return url, body
        raise FetchError(None, f"too many redirects: {url}")


class SiteCache:
    """~/celso-data/reiki_html/<自治体>/ 以下に、サイト基点からの相対パスで HTML を保存する。"""

    def __init__(self, root: str, site: Site):
        self.dir = os.path.join(root, site.sid)
        os.makedirs(self.dir, exist_ok=True)
        self.base = site.url.rsplit("/", 1)[0] + "/"

    def path_for(self, url: str) -> str:
        u = url.split("#", 1)[0]
        if u.startswith(self.base):
            rel = u[len(self.base):]
        else:
            sp = urllib.parse.urlsplit(u)
            rel = "_ext/" + sp.netloc + sp.path
        rel = rel.replace("?", "_q_")
        rel = re.sub(r"[^\w./\-]", "_", rel).lstrip("/")
        if not rel or rel.endswith("/"):
            rel += "index.html"
        parts = [p for p in rel.split("/") if p not in ("", ".", "..")]
        return os.path.join(self.dir, *parts)

    def read(self, url: str) -> bytes | None:
        p = self.path_for(url)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
        return None

    def failed(self, url: str) -> bool:
        return os.path.exists(self.path_for(url) + ".failed")

    def write(self, url: str, data: bytes) -> None:
        p = self.path_for(url)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)

    def mark_failed(self, url: str, status: int | None) -> None:
        # 404 等の恒久エラーだけ記録して、再実行時に同じ URL を叩き直さない
        p = self.path_for(url) + ".failed"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(str(status))


def decode_html(data: bytes) -> str:
    head = data[:2048].decode("ascii", errors="ignore").lower()
    m = re.search(r"charset\s*=\s*[\"']?([\w\-]+)", head)
    encs = []
    if m:
        cs = m.group(1)
        encs.append({"shift_jis": "cp932", "sjis": "cp932", "x-sjis": "cp932", "windows-31j": "cp932",
                     "euc-jp": "euc_jp", "x-euc-jp": "euc_jp"}.get(cs, cs))
    encs += ["utf-8", "cp932", "euc_jp"]
    for enc in encs:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 本文抽出
# ---------------------------------------------------------------------------

_VOID = {"br", "img", "hr", "input", "meta", "link", "wbr", "area", "col", "source"}


class _ChunkText(HTMLParser):
    """1 段落ぶんの HTML 断片からテキストを集める。skip(tag, classes) が真の要素の中身は捨てる。"""

    def __init__(self, skip):
        super().__init__(convert_charrefs=True)
        self.skip = skip
        self.stack: list[tuple[str, bool]] = []
        self.first_class: str | None = None
        self.out: list[str] = []

    def handle_starttag(self, tag, attrs):
        cls = ""
        for k, v in attrs:
            if k == "class" and v:
                cls = v
        if self.first_class is None and tag in ("div", "p"):
            self.first_class = cls
        if tag in _VOID:
            return
        parent_skip = bool(self.stack) and self.stack[-1][1]
        self.stack.append((tag, parent_skip or self.skip(tag, cls.split())))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        if self.stack and self.stack[-1][1]:
            return
        self.out.append(data)

    def text(self) -> str:
        return "".join(self.out)


_ALWAYS_SKIP = {"table", "script", "style", "rt", "rp", "select", "button", "noscript"}


def _srb_skip(tag: str, classes: list[str]) -> bool:
    if tag in _ALWAYS_SKIP:
        return True
    if tag == "span" and "num" in classes:  # 「第3条」「(1)」「2」などの番号
        return True
    if tag == "p" and "title" in classes:  # 「(定義)」などの条見出し
        return True
    return any(c in ("revise_record", "table_frame", "table-wrapper", "note") for c in classes)


# Super Reiki-Base の段落種別 (eline 直下の要素の class)。条・項・号・号の細分だけを本文として使う。
_SRB_KEEP = re.compile(r"^(article|clause|item|li\d+|enactment|announcement)$")


def extract_srb(doc: str) -> tuple[str, list[str], dict[str, int]]:
    """Super Reiki-Base の本文ページ。<div class="eline"> が 1 段落。"""
    m = re.search(r"<title>(.*?)</title>", doc, re.S | re.I)
    title = html.unescape(m.group(1)).strip() if m else ""
    end = len(doc)
    # 大きなページは属性の引用符を省いた形 (<div id=l000000001 class=eline>) で出力されることがある
    m_end = re.search(r'<div id="?secondary"?|<!-- /本文 -->|<!-- secondary -->', doc)
    if m_end:
        end = m_end.start()
    body = doc[:end]
    starts = [m.start() for m in re.finditer(r'<div id="?l\d+"? class="?eline"?>', body)]
    paras: list[str] = []
    kinds: dict[str, int] = {}
    seen_body = False  # 本則の条・項が始まったか
    in_appendix = False  # 別表・様式の中か (附則見出しで抜ける)
    for a, b in zip(starts, starts[1:] + [len(body)]):
        chunk = body[a:b]
        # eline 自身の開きタグを外して、中の最初の要素の class を段落種別にする
        chunk = chunk[chunk.index(">") + 1:]
        p = _ChunkText(_srb_skip)
        p.feed(chunk)
        p.close()
        kind = (p.first_class or "").split()[0] if p.first_class else ""
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind in ("table_section", "form_section", "xref_frame", "figure_frame"):
            in_appendix = True
        elif kind == "s-head":
            in_appendix = False
        if in_appendix:
            continue
        if _SRB_KEEP.match(kind):
            seen_body = True
            paras.append(p.text())
        elif kind == "p" and not seen_body:
            # 本則より前の class="p" 段落は前文 (別表の中の class="p" は表の注記なので使わない)
            paras.append(p.text())
    return title, paras, kinds


def _d1w_skip(tag: str, classes: list[str]) -> bool:
    return tag in _ALWAYS_SKIP


# d1w の段落 id は "h:<部><種><番号>..."。部: h=本則 s=附則 k=改正附則 b=別表 z=冒頭 d=題名。
# 種 L の 20/30 が条・項・号の本文、L10 は条見出し、W10 は改正注記なので除く。dF10 は制定文。
# id 属性を引用符なしで出力するサイト (札幌市の長い条例など) もある。
_D1W_PARA = re.compile(r'<div id="?h:([a-z])([A-Z])(\d\d)[^\s">]*"?')


def extract_d1w(doc: str) -> tuple[str, list[str], dict[str, int]]:
    m = re.search(r"<title>(.*?)</title>", doc, re.S | re.I)
    title = html.unescape(m.group(1)).strip() if m else ""
    ms = list(_D1W_PARA.finditer(doc))
    paras: list[str] = []
    kinds: dict[str, int] = {}
    for i, mm in enumerate(ms):
        part, typ, num = mm.groups()
        key = part + typ + num
        kinds[key] = kinds.get(key, 0) + 1
        if not (part in "hsk" and typ == "L" and num in ("20", "30") or key == "dF10"):
            continue
        end = ms[i + 1].start() if i + 1 < len(ms) else len(doc)
        chunk = doc[mm.end():end]
        j = chunk.find("</div>")  # d1w の段落 div は入れ子にならない
        chunk = chunk[chunk.index(">") + 1: j if j >= 0 else None]
        p = _ChunkText(_d1w_skip)
        p.feed(chunk)
        p.close()
        paras.append(p.text())
    return title, paras, kinds


_KANA_LABEL = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
_IROHA = "イロハニホヘトチリヌルヲワカヨタレソツネナラムウヰノオクヤマケフコエテアサキユメミシヱヒモセス"
_LABEL = re.compile(
    r"^(?:第\d+条(?:の\d+)*|第\d+項|\(\d+\)|\d+|[一二三四五六七八九十]+|"
    rf"[{_KANA_LABEL}{_IROHA}]|\([{_KANA_LABEL}{_IROHA}]\)|[a-zA-Z]|\([a-zA-Z]\)|[ⅰ-ⅿi-x]+|\([ⅰ-ⅿi-x]+\))\s+"
)
# (平17条例95・一部改正) / (令4規則17・追加) のような改正履歴注記
_REVISE_NOTE = re.compile(r"^[\(〔][^()〔〕]*(?:改正|追加|全改|繰下|繰上|旧第|削除|新設|全部改正)[^()〔〕]*[\)〕]$")


def clean_line(text: str) -> str | None:
    t = norm(kanji_numbers_to_arabic(text))
    t = re.sub(r"\s+", " ", t).strip()
    t = _LABEL.sub("", t, count=1).strip()
    if len(t) < 2 or t in ("削除", "削除。", "略", "(略)"):
        return None
    # 全体が括弧書きの行は、附則の見出し「(施行期日)」、省略表示「(以下省略)」、改正履歴注記などで文ではない
    if _REVISE_NOTE.match(t) or re.match(r"^\([^()]*\)$", t):
        return None
    return t


_SRB_MARK = re.compile(r'class="?eline"?>')


def extract_page(data: bytes) -> tuple[str, list[str], dict[str, int]]:
    doc = decode_html(data)
    # 形式はページの中身で判定する (サイト表の kind は目安にすぎない)
    title, paras, kinds = (extract_srb if _SRB_MARK.search(doc) else extract_d1w)(doc)
    lines = []
    for para in paras:
        c = clean_line(para)
        if c:
            lines.append(c)
    return title, lines, kinds


# ---------------------------------------------------------------------------
# 目次の探索
# ---------------------------------------------------------------------------

_HREF = re.compile(r"""(?:href|src)\s*=\s*["']?([^"' >]+)""", re.I)


def _links(base_url: str, doc: str) -> list[str]:
    out = []
    for h in _HREF.findall(doc):
        h = html.unescape(h)
        if h.startswith(("javascript:", "mailto:", "#")):
            continue
        out.append(urllib.parse.urljoin(base_url, h).split("#", 1)[0])
    # onclick="window.open('../reiki_honbun/xxx.html')" 形式も拾う
    for h in re.findall(r"window\.open\(\s*'([^']+\.html?)'", doc):
        out.append(urllib.parse.urljoin(base_url, h).split("#", 1)[0])
    return out


@dataclass
class SiteState:
    site: Site
    status: str = "pending"
    reason: str = ""
    index_pages: int = 0
    honbun_urls: list[str] = field(default_factory=list)
    fetched: int = 0  # 今回ネットワークから取得した本文ページ
    cached: int = 0  # キャッシュ済みの本文ページ
    failed: int = 0
    chars: int = 0
    pages_ok: int = 0


class GiveUp(Exception):
    pass


class Crawler:
    def __init__(self, fetcher: Fetcher, cache_root: str, max_consecutive_errors: int, deadline: float):
        self.fetcher = fetcher
        self.cache_root = cache_root
        self.max_errors = max_consecutive_errors
        self.deadline = deadline
        self.stop = threading.Event()
        self.total_chars = 0
        self._lock = threading.Lock()

    def _get(self, st: SiteState, cache: SiteCache, url: str, errors: list[int]) -> bytes | None:
        """キャッシュ優先で取得。連続失敗が上限を超えたら GiveUp。"""
        data = cache.read(url)
        if data is not None:
            return data
        if cache.failed(url):
            return None
        if self.stop.is_set() or time.time() > self.deadline:
            raise GiveUp("time limit")
        try:
            final, data = self.fetcher.get(url)
        except FetchError as e:
            if self.fetcher.offline:
                return None
            errors[0] += 1
            st.failed += 1
            # 404 等の恒久エラーと robots.txt 禁止だけ記録する (429 等の一時エラーは次回やり直す)
            if e.status in (401, 403, 404, 410) or "disallow" in str(e):
                cache.mark_failed(url, e.status)
            log(f"[{st.site.name}] 失敗 ({errors[0]} 連続): {e}")
            if errors[0] >= self.max_errors:
                raise GiveUp(f"{errors[0]} 回連続で失敗 (最後: {e})")
            return None
        if BANNED_HOST.search(urllib.parse.urlsplit(final).netloc):
            raise GiveUp(f"banned redirect {final}")
        if final != url:
            log(f"[{st.site.name}] リダイレクト: {url} -> {final}")
        errors[0] = 0
        cache.write(url, data)
        return data

    def discover(self, st: SiteState, cache: SiteCache) -> None:
        site = st.site
        errors = [0]
        manifest = os.path.join(cache.dir, "_honbun_urls.json")
        if os.path.exists(manifest):
            with open(manifest, encoding="utf-8") as f:
                st.honbun_urls = json.load(f)
            return
        if not self.fetcher.offline and not self.fetcher.allowed(site.url):
            raise GiveUp("robots.txt で禁止 (または robots.txt 取得不能)")
        top = self._get(st, cache, site.url, errors)
        if top is None:
            raise GiveUp("トップページを取得できない")
        doc = decode_html(top)
        base = cache.base
        honbun: set[str] = set()
        # リンク集の URL 形式と実体が違うことがある (福岡市は d1w_reiki/ 配下が Super Reiki-Base、
        # 品川区は reiki_menu.html が d1w 形式) ので、トップページの中身で形式を判定する
        if re.search(r"reiki_(?:kana|taikei)/", doc):
            kind = "srb"
        elif "mokuji_index" in doc or "mokuji_bunya" in doc:
            kind = "d1w"
        else:
            raise GiveUp("例規集の形式を判定できない (Super Reiki-Base / d1w のどちらでもない)")
        if kind == "srb":
            # 五十音順目次 (reiki_kana/) と体系目次 (reiki_taikei/) を辿り、reiki_honbun/*.html を集める。
            # 未施行の例規 (reiki_miseko/) は現行版とほぼ重複するので辿らない。
            index_re = re.compile(re.escape(base) + r"(reiki_kana|reiki_taikei)/[^?]*\.html?$")
            honbun_re = re.compile(re.escape(base) + r"reiki_honbun/[^/?]+\.html?$")
            queue = [u for u in _links(site.url, doc) if index_re.match(u)]
            seen = set(queue)
            while queue and len(seen) < 600:
                u = queue.pop(0)
                data = self._get(st, cache, u, errors)
                if data is None:
                    continue
                st.index_pages += 1
                for v in _links(u, decode_html(data)):
                    v = v.split("?", 1)[0]
                    if honbun_re.match(v):
                        honbun.add(v)
                    elif index_re.match(v) and v not in seen:
                        seen.add(v)
                        queue.append(v)
        else:
            # 五十音検索の左フレーム (mokuji_index_index.html) → index_NNN.html → OpenResDataWin('H...')
            idx_url = base + "mokuji_index_index.html"
            data = self._get(st, cache, idx_url, errors)
            if data is None:
                raise GiveUp("五十音目次 (mokuji_index_index.html) を取得できない")
            pages = sorted({u for u in _links(idx_url, decode_html(data)) if re.search(r"/index_\d+\.html?$", u)})
            for u in pages:
                d = self._get(st, cache, u, errors)
                if d is None:
                    continue
                st.index_pages += 1
                for hno in re.findall(r"OpenResDataWin\('([^'/]+)'\)", decode_html(d)):
                    honbun.add(f"{base}{hno}/{hno}_j.html")
        if not honbun:
            raise GiveUp(f"本文ページへのリンクが見つからない (目次 {st.index_pages} ページ)")
        st.honbun_urls = sorted(honbun)
        if not self.fetcher.offline:
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump(st.honbun_urls, f, ensure_ascii=False, indent=0)

    def run_site(self, st: SiteState, probe: bool, max_pages: int, target_chars: int) -> None:
        site = st.site
        cache = SiteCache(self.cache_root, site)
        try:
            st.status = "discovering"
            self.discover(st, cache)
            log(f"[{site.name}] 本文 {len(st.honbun_urls)} ページ (目次 {st.index_pages} ページ)")
            if probe:
                # 抽出結果の確認用に先頭の数ページだけ取得する
                urls = st.honbun_urls[:: max(1, len(st.honbun_urls) // 3)][:3]
            else:
                urls = st.honbun_urls[:max_pages] if max_pages else st.honbun_urls
            st.status = "crawling"
            errors = [0]
            for u in urls:
                if self.stop.is_set():
                    raise GiveUp("target reached")
                was_cached = cache.read(u) is not None
                data = self._get(st, cache, u, errors)
                if data is None:
                    continue
                if was_cached:
                    st.cached += 1
                else:
                    st.fetched += 1
                title, lines, _ = extract_page(data)
                n = sum(len(x) for x in lines)
                st.chars += n
                st.pages_ok += 1
                with self._lock:
                    self.total_chars += n
                    if target_chars and self.total_chars >= target_chars:
                        self.stop.set()
                if st.fetched and st.fetched % 200 == 0:
                    log(f"[{site.name}] {st.pages_ok}/{len(urls)} ページ, {st.chars:,} 字 (全体 {self.total_chars:,} 字)")
            st.status = "done"
        except GiveUp as e:
            st.status = "partial" if st.pages_ok else "failed"
            st.reason = str(e)
            log(f"[{site.name}] 中断: {e}")
        except Exception as e:  # noqa: BLE001  1 サイトの想定外エラーで全体を止めない
            st.status = "partial" if st.pages_ok else "failed"
            st.reason = f"{type(e).__name__}: {e}"
            log(f"[{site.name}] 例外: {st.reason}")


# ---------------------------------------------------------------------------
# コーパス書き出し
# ---------------------------------------------------------------------------


def _extract_site(args: tuple[str, str]) -> list[tuple[str, str, list[str]]]:
    """1 自治体ぶんのキャッシュを抽出する (ProcessPoolExecutor 用)。[(URL, 題名, 行)]"""
    cache_root, sid = args
    site = next(s for s in SITES if s.sid == sid)
    cache = SiteCache(cache_root, site)
    manifest = os.path.join(cache.dir, "_honbun_urls.json")
    if not os.path.exists(manifest):
        return []
    with open(manifest, encoding="utf-8") as f:
        urls = json.load(f)
    out = []
    for u in urls:
        data = cache.read(u)
        if data is None:
            continue
        title, lines, _ = extract_page(data)
        if lines:
            out.append((u, title, lines))
    return out


def write_corpus(states: list[SiteState], cache_root: str, out_path: str) -> dict:
    """キャッシュ済み本文ページから reiki.txt を書く。同一内容のページ (別 URL・別自治体) は 1 回だけ出す。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    seen_pages: set[bytes] = set()
    stats = {}
    total_lines = total_chars = dup_pages = 0
    with open(tmp, "w", encoding="utf-8") as out, ProcessPoolExecutor() as ex:
        results = ex.map(_extract_site, [(cache_root, st.site.sid) for st in states])
        for st, pages_data in zip(states, results):
            site = st.site
            if not pages_data:
                continue
            pages = chars = lines_n = 0
            for u, title, lines in pages_data:
                if BANNED_NAME.search(title):
                    log(f"[{site.name}] 題名に横浜を含むページを除外: {u}")
                    continue
                h = hashlib.sha1("\n".join(lines).encode()).digest()
                if h in seen_pages:
                    dup_pages += 1
                    continue
                seen_pages.add(h)
                for line in lines:
                    out.write(line + "\n")
                pages += 1
                lines_n += len(lines)
                chars += sum(len(x) for x in lines)
            stats[site.sid] = {"pref": site.pref, "name": site.name, "category": site.category,
                               "kind": site.kind, "url": site.url, "pages": pages,
                               "lines": lines_n, "chars": chars, "status": st.status, "reason": st.reason}
            total_lines += lines_n
            total_chars += chars
    os.replace(tmp, out_path)
    summary = {"lines": total_lines, "chars": total_chars, "duplicate_pages": dup_pages, "sites": stats}
    with open(os.path.join(cache_root, "_stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", help="対象自治体名をカンマ区切りで (既定: 全件)")
    ap.add_argument("--exclude", help="除外する自治体名をカンマ区切りで")
    ap.add_argument("--all", action="store_true", help="予備の自治体 (サイト表で False のもの) も含める")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help=f"HTML キャッシュ (既定: {DEFAULT_CACHE})")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"出力 (既定: {DEFAULT_OUT})")
    ap.add_argument("--interval", type=float, default=1.1, help="同一ホストへのリクエスト間隔 [秒] (1.0 未満は不可)")
    ap.add_argument("--time-limit", type=float, default=7200, help="収集の打ち切り時間 [秒]")
    ap.add_argument("--max-pages", type=int, default=0, help="1 自治体あたりの本文ページ上限 (0 = 無制限)")
    ap.add_argument("--target-chars", type=int, default=100_000_000, help="この字数に達したら収集を止める")
    ap.add_argument("--max-errors", type=int, default=8, help="この回数連続で失敗したら自治体を諦める")
    ap.add_argument("--probe", action="store_true", help="目次の探索と数ページの抽出確認だけ行う")
    ap.add_argument("--extract-only", action="store_true", help="ネットワークに出ずキャッシュから抽出だけ行う")
    ap.add_argument("--list", action="store_true", help="対象自治体の一覧を表示して終わる")
    args = ap.parse_args()
    if args.interval < 1.0:
        ap.error("--interval は 1.0 秒以上にすること")

    sites = SITES if args.all or args.sites else [s for s in SITES if s.enabled]
    if args.sites:
        names = set(args.sites.split(","))
        if any(BANNED_NAME.search(n) for n in names):
            ap.error("横浜市は評価データなので対象にできない")
        sites = [s for s in sites if s.name in names or s.sid in names]
    if args.exclude:
        ex = set(args.exclude.split(","))
        sites = [s for s in sites if s.name not in ex and s.sid not in ex]
    if args.list:
        for s in sites:
            print(f"{s.pref}\t{s.name}\t{s.category}\t{s.kind}\t{s.url}")
        return

    global LOG_FILE
    os.makedirs(args.cache, exist_ok=True)
    LOG_FILE = os.path.join(args.cache, "_crawl.log")
    states = [SiteState(s) for s in sites]

    # 前回の状態 (中断理由など) を読む。今回取得しない自治体の行はこれを引き継ぐ
    prev = {}
    stats_path = os.path.join(args.cache, "_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, encoding="utf-8") as f:
            prev = json.load(f).get("sites", {})

    if not args.extract_only:
        fetcher = Fetcher(args.interval, offline=False)
        crawler = Crawler(fetcher, args.cache, args.max_errors, time.time() + args.time_limit)
        log(f"開始: {len(sites)} 自治体, 間隔 {args.interval}s/ホスト, 制限 {args.time_limit:.0f}s")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(states)) as ex:
            for f in [ex.submit(crawler.run_site, st, args.probe, args.max_pages, args.target_chars) for st in states]:
                f.result()
        log(f"収集終了: {time.time() - t0:.0f}s, リクエスト {fetcher.requests} 件")
        for st in states:
            print(f"{st.site.pref}\t{st.site.name}\t{st.status}\t本文URL {len(st.honbun_urls)}\t"
                  f"取得 {st.fetched}\tキャッシュ {st.cached}\t失敗 {st.failed}\t{st.chars:,} 字\t{st.reason}",
                  file=sys.stderr)
        if args.probe:
            return

    # reiki.txt は常に「既定の対象 (予備を除く) + 今回指定した自治体」の全キャッシュから作り直す。
    # --sites で一部だけ再取得しても、他の自治体の分が出力から消えないようにするため。
    crawled = {st.site.sid: st for st in states} if not args.extract_only else {}
    out_states = []
    for s in SITES:
        if not (s.enabled or s.sid in {x.site.sid for x in states}):
            continue
        st = crawled.get(s.sid) or SiteState(s)
        if s.sid not in crawled and s.sid in prev:
            st.status = prev[s.sid].get("status", "")
            st.reason = prev[s.sid].get("reason", "")
        out_states.append(st)
    states = out_states

    summary = write_corpus(states, args.cache, args.out)
    print(f"{args.out}: {summary['lines']:,} 行, {summary['chars']:,} 字 (重複ページ除外 {summary['duplicate_pages']})",
          file=sys.stderr)
    for s in summary["sites"].values():
        print(f"{s['pref']}\t{s['name']}\t{s['category']}\t{s['pages']} ページ\t{s['chars']:,} 字\t{s['status']}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
