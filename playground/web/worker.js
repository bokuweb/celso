// 検査器の読み込みと検査を担う Web Worker。
// 辞書 (IPADIC 約 6 万字種の語彙) の解析に数秒かかるため、画面のスレッドから外している。
import init, { Playground } from './pkg/celso_playground.js';

// 読み込むファイル (gzip 済み)。size は進捗表示の目安 (実際の値は Content-Length を優先する)
const ASSETS = [
  ['model', 'assets/model.bin.gz'],
  ['cooc', 'assets/cooc.bin.gz'],
  ['inflections', 'assets/inflections.tsv.gz'],
  ['readings', 'assets/readings.tsv.gz'],
  ['lex', 'assets/lex.csv.gz'],
  ['matrix', 'assets/matrix.def.gz'],
  ['char', 'assets/char.def.gz'],
  ['unk', 'assets/unk.def.gz'],
];

let playground = null;

const post = (type, payload) => self.postMessage({ type, ...payload });

// gzip のまま取得して進捗を通知し、DecompressionStream で展開したバイト列を返す
async function fetchGzip(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const blob = new Blob(chunks);
  // 配信側が Content-Encoding: gzip を付けるとブラウザが展開済みで渡してくるので、
  // gzip の先頭 2 バイト (1f 8b) を見て、展開が要るときだけ DecompressionStream を通す
  const head = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
  if (head[0] !== 0x1f || head[1] !== 0x8b) {
    return new Uint8Array(await blob.arrayBuffer());
  }
  const stream = blob.stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function load() {
  const started = performance.now();
  await init();
  const progress = new Map();
  const report = () => {
    let received = 0;
    let total = 0;
    for (const [r, t] of progress.values()) {
      received += r;
      total += t;
    }
    post('progress', { phase: 'download', received, total });
  };
  const files = await Promise.all(
    ASSETS.map(async ([key, url]) => {
      progress.set(key, [0, 0]);
      const bytes = await fetchGzip(url, (r, t) => {
        progress.set(key, [r, t]);
        report();
      });
      return [key, bytes];
    }),
  );
  const f = Object.fromEntries(files);
  post('progress', { phase: 'build' });
  playground = new Playground(f.model, f.cooc, f.inflections, f.readings, f.lex, f.matrix, f.char, f.unk);
  post('ready', { ms: Math.round(performance.now() - started) });
}

self.onmessage = (e) => {
  const { type, id, text } = e.data;
  if (type !== 'check' || playground == null) return;
  const started = performance.now();
  const result = JSON.parse(playground.check(text));
  // 検査した本文も返す (結果が届く前に入力が変わっても、強調表示がずれないように)
  post('result', { id, text, result, ms: performance.now() - started });
};

load().catch((err) => post('error', { message: String(err && err.message ? err.message : err) }));
