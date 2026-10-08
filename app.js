// celso playground の画面。検査は worker.js (WebAssembly) で行い、ここでは入力・強調表示・修正の適用だけを扱う。

// サンプル文は samples.json に置く (tests/regression.rs が同じファイルで「期待どおりに直るか」を確かめる)
let samples = [];

const KIND_LABELS = {
  delete: '余計な文字',
  substitute: '助詞の誤り',
  inflection: '活用の誤り',
  insert: '文字の抜け',
  homophone: '変換ミス',
  char: '誤字',
  pattern: '誤字',
};

const DOMAIN_LABELS = { legal: '法令文', contract: '契約書', general: '一般文' };

const $ = (id) => document.getElementById(id);
const input = $('input');
const preview = $('preview');
const findingsEl = $('findings');
const statusEl = $('status');
const statusText = $('status-text');
const barFill = $('bar-fill');
const resultMeta = $('result-meta');
const charCount = $('char-count');

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let ready = false;
let requestId = 0;
let latestShownId = 0;
let timer = 0;
let current = { text: '', findings: [] };

const formatMB = (bytes) => `${(bytes / 1024 / 1024).toFixed(1)}MB`;

worker.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case 'progress':
      if (msg.phase === 'download') {
        const ratio = msg.total ? msg.received / msg.total : 0;
        barFill.style.width = `${Math.round(ratio * 90)}%`;
        statusText.textContent = `モデルと辞書を読み込んでいます… ${formatMB(msg.received)}${msg.total ? ` / ${formatMB(msg.total)}` : ''}`;
      } else {
        barFill.style.width = '95%';
        statusText.textContent = '辞書を組み立てています…';
      }
      break;
    case 'ready':
      ready = true;
      statusEl.classList.add('ready');
      statusText.textContent = `準備ができました (${(msg.ms / 1000).toFixed(1)} 秒)。入力すると自動で検査します。`;
      requestCheck(0);
      break;
    case 'char-ready':
      // 文字モデルが付いたので検査し直す (語の中の 1 字の誤りも出るようになる)
      statusText.textContent += ` 文字モデルも読み込みました (${(msg.ms / 1000).toFixed(1)} 秒)。`;
      requestCheck(0);
      break;
    case 'result':
      if (msg.id < latestShownId) return;
      latestShownId = msg.id;
      render(msg.text, msg.result, msg.ms);
      break;
    case 'error':
      statusEl.classList.add('error');
      statusText.textContent = `読み込みに失敗しました: ${msg.message}`;
      break;
  }
};

// 入力が落ち着いてから検査する (文単位のキャッシュがあるので、2 回目以降は変わった文だけを計算する)
function requestCheck(delay = 300) {
  clearTimeout(timer);
  timer = setTimeout(() => {
    if (!ready) return;
    const text = input.value;
    requestId += 1;
    worker.postMessage({ type: 'check', id: requestId, text });
  }, delay);
}

const replacementLabel = (r) => (r === '' ? '（削除）' : r);

function render(text, result, ms) {
  current = { text, findings: result.findings };
  const chars = Array.from(text);
  resultMeta.textContent = `${DOMAIN_LABELS[result.domain] ?? result.domain}として検査 ・ ${result.findings.length} 件 ・ ${ms.toFixed(0)}ms`;

  // 強調表示: 指摘の範囲を <mark> で囲む (オフセットは Unicode スカラー値 = Array.from の添字)
  preview.textContent = '';
  let pos = 0;
  result.findings.forEach((f, i) => {
    if (f.start < pos) return;
    preview.append(chars.slice(pos, f.start).join(''));
    const mark = document.createElement('mark');
    // 挿入 (start == end) は前後が見えるよう、次の 1 文字を囲む
    const end = f.end > f.start ? f.end : Math.min(f.start + 1, chars.length);
    mark.textContent = chars.slice(f.start, end).join('') || '␣';
    mark.dataset.index = String(i);
    mark.addEventListener('click', () => activate(i, true));
    preview.append(mark);
    pos = end;
  });
  preview.append(chars.slice(pos).join(''));

  findingsEl.textContent = '';
  if (result.findings.length === 0) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = text.trim() ? '誤字脱字の可能性がある箇所は見つかりませんでした' : '';
    findingsEl.append(li);
    return;
  }
  result.findings.forEach((f, i) => findingsEl.append(findingItem(f, i)));
}

function change(original, replacement, isInsert) {
  return isInsert ? `「□」→「${replacement}」` : `「${original}」→「${replacementLabel(replacement)}」`;
}

function findingItem(f, i) {
  const li = document.createElement('li');
  li.dataset.index = String(i);
  li.addEventListener('mouseenter', () => activate(i, false));

  const head = document.createElement('div');
  head.className = 'finding-head';
  const kind = document.createElement('span');
  kind.className = 'kind';
  kind.textContent = KIND_LABELS[f.kind] ?? f.kind;
  const ch = document.createElement('span');
  ch.className = 'change';
  ch.textContent = change(f.original, f.replacement, f.start === f.end);
  const score = document.createElement('span');
  score.className = 'score';
  // 判定器の対数オッズ・言語モデルの改善幅など、種類によって尺度が違うので目安として出す
  score.title = '確からしさの目安 (大きいほど誤りの可能性が高い)';
  score.textContent = f.score.toFixed(1);
  head.append(kind, ch, score);

  const choices = document.createElement('div');
  choices.className = 'choices';
  const primary = document.createElement('button');
  primary.type = 'button';
  primary.className = 'primary';
  primary.textContent = f.replacement === '' ? '削除する' : `「${f.replacement}」に直す`;
  primary.addEventListener('click', () => apply(f));
  choices.append(primary);
  for (const alt of f.alternatives) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = `別案: ${change(alt.original, alt.replacement, alt.start === alt.end)}`;
    b.addEventListener('click', () => apply(alt));
    choices.append(b);
  }
  li.append(head, choices);
  return li;
}

function activate(i, scroll) {
  for (const el of document.querySelectorAll('.active')) el.classList.remove('active');
  const mark = preview.querySelector(`mark[data-index="${i}"]`);
  const li = findingsEl.querySelector(`li[data-index="${i}"]`);
  mark?.classList.add('active');
  li?.classList.add('active');
  if (scroll) li?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

// 修正案を本文に反映して検査し直す。検査に出した本文と入力欄が一致するときだけ適用する
function apply(s) {
  if (input.value !== current.text) return;
  const chars = Array.from(current.text);
  chars.splice(s.start, s.end - s.start, ...Array.from(s.replacement));
  input.value = chars.join('');
  updateCount();
  requestCheck(0);
}

function updateCount() {
  charCount.textContent = `${Array.from(input.value).length.toLocaleString()} 文字`;
}

input.addEventListener('input', () => {
  updateCount();
  requestCheck();
});

function setText(text) {
  input.value = text;
  updateCount();
  requestCheck(0);
}

// サンプルのボタンを samples.json から作る (最後に「クリア」)
async function loadSamples() {
  const box = $('samples');
  try {
    const res = await fetch('samples.json');
    samples = await res.json();
  } catch {
    samples = [];
  }
  for (const s of samples) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = s.label;
    b.addEventListener('click', () => setText(s.text));
    box.append(b);
  }
  const clear = document.createElement('button');
  clear.type = 'button';
  clear.textContent = 'クリア';
  clear.addEventListener('click', () => setText(''));
  box.append(clear);
  if (input.value === '' && samples.length > 0) setText(samples[0].text);
}

updateCount();
loadSamples();
