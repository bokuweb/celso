"""契約書コーパス用の丁寧な取得ツール (celso リポジトリ外に置く)。

  python3 fetch.py links URL [PATTERN]   # ページを取得し、リンク (テキスト\thref) を列挙
  python3 fetch.py get URL NAME [TITLE]  # ファイルを files/NAME に保存し SOURCES.tsv に追記

ルール: robots.txt を守る / 同一ホスト 1.1 秒以上の間隔 (プロセスをまたいで state ファイルで管理) /
研究目的の User-Agent。JEITA と IPA アジャイル開発モデル契約は評価用なので取得自体を拒否する。
"""
import fcntl, html, json, os, re, sys, time, urllib.error, urllib.parse, urllib.request, urllib.robotparser

# 取得物・中間ファイルはリポジトリの外 (既定 data/contracts_raw/、gitignore 済み) に置く
ROOT = os.environ.get("CONTRACTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "contracts_raw")
os.makedirs(ROOT, exist_ok=True)
FILES = os.path.join(ROOT, "files")
STATE = os.path.join(ROOT, ".host_state.json")
SOURCES = os.path.join(ROOT, "SOURCES.tsv")
UA = ("celso-contracts-research/0.1 (non-commercial research crawler for a Japanese typo checker; "
      "+https://github.com/bokuweb/celso; 1 req/s per host)")
UA_TOKEN = "celso-contracts-research"
INTERVAL = 1.1
# 評価データの漏洩防止
BANNED = re.compile(r"jeita|agile|アジャイル", re.I)


def wait_host(host):
    with open(STATE + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = json.load(open(STATE)) if os.path.exists(STATE) else {}
        wait = st.get(host, 0) + INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        st[host] = time.time()
        json.dump(st, open(STATE, "w"))


def raw_get(url):
    host = urllib.parse.urlsplit(url).netloc.lower()
    wait_host(host)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ja"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
            final = r.geturl()
    except urllib.error.HTTPError as e:
        return e.code, b"", url
    finally:
        wait_host_done(host)
    return 200, body, final


def wait_host_done(host):
    with open(STATE + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = json.load(open(STATE)) if os.path.exists(STATE) else {}
        st[host] = time.time()
        json.dump(st, open(STATE, "w"))


_robots = {}


def allowed(url):
    sp = urllib.parse.urlsplit(url)
    key = f"{sp.scheme}://{sp.netloc}"
    if key not in _robots:
        cache = os.path.join(ROOT, ".robots", sp.netloc + ".txt")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        if os.path.exists(cache):
            text = open(cache, encoding="utf-8").read()
        else:
            try:
                st, body, _ = raw_get(key + "/robots.txt")
            except Exception as e:  # noqa: BLE001
                print(f"robots.txt 取得失敗 {key}: {e}", file=sys.stderr)
                st, body = 599, b""
            if st == 200:
                text = body.decode("utf-8", "ignore")
                if "<html" in text[:500].lower():
                    text = ""
            elif 400 <= st < 500:
                # RFC 9309: 4xx は「robots.txt なし」。ただし本文ページ自体が 403 のサイト (WAF) は get() が失敗するので取得されない
                text = ""
            else:
                text = "User-agent: *\nDisallow: /\n"
            open(cache, "w", encoding="utf-8").write(text)
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(text.splitlines())
        _robots[key] = rp
    return _robots[key].can_fetch(UA_TOKEN, url)


def get(url):
    if BANNED.search(urllib.parse.unquote(url)):
        raise SystemExit(f"BANNED (評価データ): {url}")
    if not allowed(url):
        raise SystemExit(f"robots.txt disallow: {url}")
    st, body, final = raw_get(url)
    if st != 200:
        raise SystemExit(f"HTTP {st}: {url}")
    if final != url and not allowed(final):
        raise SystemExit(f"robots.txt disallow (redirect): {final}")
    return body, final


def get_page(url):
    import hashlib
    d = os.path.join(ROOT, "pages"); os.makedirs(d, exist_ok=True)
    f = os.path.join(d, hashlib.md5(url.encode()).hexdigest() + ".html")
    if os.path.exists(f):
        b = open(f, "rb").read()
        nl = b.index(b"\n")
        return b[nl + 1:], b[:nl].decode()
    body, final = get(url)
    open(f, "wb").write(final.encode() + b"\n" + body)
    return body, final


def links(url, pat=None):
    body, final = get_page(url)
    m = re.search(rb'charset=["\']?([\w-]+)', body[:3000], re.I)
    enc = m.group(1).decode() if m else "utf-8"
    try:
        text = body.decode(enc, "ignore")
    except LookupError:
        text = body.decode("utf-8", "ignore")
    out = []
    for m in re.finditer(r'<a\s[^>]*?href\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</a>', text, re.S | re.I):
        href = urllib.parse.urljoin(final, html.unescape(m.group(1)))
        t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", m.group(2)))).strip()
        if pat and not re.search(pat, href + " " + t, re.I):
            continue
        out.append((t, href))
    return out


def save(url, name, title=""):
    path = os.path.join(FILES, name)
    if os.path.exists(path):
        print(f"skip (exists) {name}")
        return
    body, final = get(url)
    open(path, "wb").write(body)
    new = not os.path.exists(SOURCES)
    with open(SOURCES, "a", encoding="utf-8") as f:
        if new:
            f.write("file\turl\ttitle\tbytes\tfetched_at\n")
        f.write(f"{name}\t{final}\t{title}\t{len(body)}\t{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    print(f"saved {name} {len(body)} bytes")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "links":
        for t, h in links(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None):
            print(f"{t}\t{h}")
    elif cmd == "text":
        body, final = get_page(sys.argv[2])
        t = body.decode("utf-8", "ignore")
        t = re.sub(r"(?is)<(script|style).*?</\\1>", "", t)
        t = re.sub(r'(?is)<a\s[^>]*href="([^"]+)"[^>]*>', lambda m: " [" + m.group(1) + "] ", t)
        t = re.sub(r"<[^>]+>", " ", t)
        print(re.sub(r"[ \t\r]*\n\s*", "\n", html.unescape(t)))
    elif cmd == "get":
        save(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    elif cmd == "batch":
        # TSV: url\tname\ttitle
        for line in open(sys.argv[2], encoding="utf-8"):
            if not line.strip() or line.startswith("#"):
                continue
            u, n, *t = line.rstrip("\n").split("\t")
            try:
                save(u, n, t[0] if t else "")
            except SystemExit as e:
                print(f"FAIL {n}: {e}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {n}: {type(e).__name__} {e}", file=sys.stderr)
