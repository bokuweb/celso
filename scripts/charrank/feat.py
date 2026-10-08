"""文字単位の直しの判定器の特徴量 (src/charcheck.rs の CharRanker::features と同じ名前・同じ値にする)。

celso check の調査用の出力 (CELSO_TRACE) の [char] 行を読む。
"""
import re,math,unicodedata,collections
nf=lambda s: unicodedata.normalize('NFKC',s).replace(' ','')
isk=lambda c: '一'<=c<='鿿' or c=='々'
ishira=lambda c: 'ぁ'<=c<='ゖ' or c=='ー'
iskata=lambda c: 'ァ'<=c<='ヺ'
def cls(c):
    if not c: return 'none'
    c=c[0]
    return 'k' if isk(c) else 'h' if ishira(c) else 'K' if iskata(c) else 'o'
PRIOR={}
for l in open(S+'/prior_ex.tsv'):
    a,b,v=l.rstrip('\n').split('\t'); PRIOR[(a,b)]=int(v)
RX=re.compile(r'「(.*?)」→「(.*?)」 文字=(\S+) 単語=(\S+) 位置=(\d+) logp=(\S+) 差=(\S+) 怪しい=(\d+) 未知語=(-?\d+) 語数=(-?\d+) 文法=(\S+) 共起=(\S+)\t(.*)')
def parse(fn):
    out=[]
    for l in open(fn):
        m=RX.search(l)
        if not m: continue
        o,r,cd,wd,pos,lp,mg,ns,unk,nt,ad,co,fx=m.groups()
        out.append(dict(o=o,r=r,cd=float(cd),wd=float(wd),pos=int(pos),lp=float(lp),mg=float(mg),ns=int(ns),unk=int(unk),nt=int(nt),ad=float(ad),co=float(co),fx=fx))
    return out
def typ(o,r):
    if not o: return 'ins'
    if not r: return 'del'
    if len(o)==2 and len(r)==2 and o[::-1]==r: return 'swap'
    if isk(o[0]): return 'subk'
    return 'subn'
PART=set('のにがをはでともへやか')
def feats(c):
    o,r=c['o'],c['r']; t=typ(o,r)
    f={}
    def add(k,v=1.0): f[k]=f.get(k,0)+v
    p=PRIOR.get((o,r),0); lp_=math.log1p(p)
    # 直す位置の前後の文字の種類 (直した文から)
    fx=c['fx']; i=c['pos']
    left=fx[i-1] if i>0 else ''
    right_i=i+len(r)
    right=fx[right_i] if right_i<len(fx) else ''
    oc=cls(o); rc=cls(r)
    for pre in ['', t+':']:
        add(pre+'bias')
        add(pre+'cd',c['cd']); add(pre+'wd',c['wd']); add(pre+'min',min(c['cd'],c['wd']))
        add(pre+'lp',c['lp']); add(pre+'mg',min(c['mg'],10)); add(pre+'prior',lp_); add(pre+'prior0',1.0 if p==0 else 0.0)
        add(pre+'unk',c['unk']); add(pre+'nt',c['nt']); add(pre+'ns',math.log1p(c['ns']))
        add(pre+'ad',c['ad']); add(pre+'co',c['co'])
    add(f'{t}:oc={oc}'); add(f'{t}:rc={rc}'); add(f'{t}:lc={cls(left)}'); add(f'{t}:Rc={cls(right)}')
    add(f'{t}:lc={cls(left)}:rc={cls(right)}')
    if len(o)<=1 and len(r)<=1 and not (isk(o or 'x') ):
        add(f'{t}:o={o}'); add(f'{t}:r={r}')
    if o in PART and r in PART and t=='subn': add('partpair')
    add(t+':cdwd',c['cd']*c['wd']/10)
    # 値の区間 (線形だけでは表せない形を補う)
    b=lambda v,step,lo,hi: int(max(lo,min(hi,v))//step)
    add(f"{t}:cdb={b(c['cd'],1,0,12)}"); add(f"{t}:wdb={b(c['wd'],1,-3,10)}")
    add(f"{t}:cdb={b(c['cd'],2,0,12)}:wdb={b(c['wd'],2,-2,10)}")
    add(f"{t}:lpb={b(c['lp'],1,-8,-3)}"); add(f"{t}:mgb={b(c['mg'],1,0,8)}")
    add(f"{t}:pb={b(lp_,1,0,8)}")
    add(f"{t}:nsb={b(c['ns'],2,0,12)}")
    add(f"{t}:adb={b(c['ad'],1,-4,6)}"); add(f"{t}:cob={b(c['co'],0.5,-2,2)}")
    return f
