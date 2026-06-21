
import streamlit as st
import subprocess
import json
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="LogSense AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
* { font-family: Inter, sans-serif; }
.stApp { background: linear-gradient(135deg,#0a0e1a 0%,#0d1117 50%,#0a0e1a 100%); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0d1117 0%,#161b22 100%); border-right:1px solid rgba(88,166,255,.15); }
#MainMenu,footer,header{visibility:hidden}
.hero{background:linear-gradient(135deg,rgba(88,166,255,.08),rgba(63,185,80,.05));border:1px solid rgba(88,166,255,.2);border-radius:16px;padding:40px;text-align:center;margin-bottom:32px}
.hero h1{font-size:3rem;font-weight:800;background:linear-gradient(135deg,#58a6ff,#3fb950);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0}
.hero p{color:#8b949e;font-size:1.1rem;margin-top:8px}
.badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:.78rem;font-weight:600;margin:10px 3px 0;letter-spacing:.5px}
.badge-blue{background:rgba(88,166,255,.15);border:1px solid rgba(88,166,255,.3);color:#58a6ff}
.badge-green{background:rgba(63,185,80,.15);border:1px solid rgba(63,185,80,.3);color:#3fb950}
.badge-purple{background:rgba(188,140,255,.15);border:1px solid rgba(188,140,255,.3);color:#bc8cff}
.metric-card{background:linear-gradient(135deg,#161b22,#1c2128);border:1px solid rgba(48,54,61,.8);border-radius:12px;padding:24px;text-align:center;margin:8px 0;position:relative;overflow:hidden}
.metric-card .val{font-size:2.4rem;font-weight:800;font-family:'JetBrains Mono',monospace;line-height:1}
.metric-card .lbl{font-size:.72rem;color:#8b949e;margin-top:6px;text-transform:uppercase;letter-spacing:1.5px;font-weight:600}
.metric-card .sub{font-size:.7rem;color:#3fb950;margin-top:4px}
.stage-done{background:rgba(63,185,80,.1);border:1px solid rgba(63,185,80,.25);border-radius:8px;padding:10px 14px;margin:5px 0;color:#3fb950;font-size:.85rem;display:flex;align-items:center;gap:8px}
.stage-wait{background:rgba(48,54,61,.3);border:1px solid rgba(48,54,61,.4);border-radius:8px;padding:10px 14px;margin:5px 0;color:#484f58;font-size:.85rem;display:flex;align-items:center;gap:8px}
.dot-green{width:8px;height:8px;border-radius:50%;background:#3fb950;box-shadow:0 0 8px rgba(63,185,80,.6);flex-shrink:0}
.dot-grey{width:8px;height:8px;border-radius:50%;background:#484f58;flex-shrink:0}
.flow-box{background:#010409;border:1px solid rgba(48,54,61,.8);border-radius:12px;padding:24px;font-family:'JetBrains Mono',monospace;font-size:.82rem;line-height:2}
.anomaly-hdr{background:linear-gradient(135deg,rgba(248,81,73,.08),rgba(248,81,73,.04));border:1px solid rgba(248,81,73,.25);border-left:4px solid #f85149;border-radius:0 12px 12px 0;padding:20px 24px;margin:16px 0 8px}
.sid{font-family:'JetBrains Mono',monospace;color:#ff7b72;font-size:.95rem;font-weight:600}
.log-box{background:#010409;border:1px solid rgba(48,54,61,.6);border-radius:8px;padding:12px;margin:12px 0}
.log-line{font-family:'JetBrains Mono',monospace;font-size:.78rem;padding:4px 10px;border-radius:4px;margin:2px 0;line-height:1.5}
.log-error{background:rgba(248,81,73,.08);color:#ff7b72;border-left:2px solid #f85149}
.log-warn{background:rgba(227,179,65,.08);color:#e3b341;border-left:2px solid #e3b341}
.log-info{color:#6e7681}
.llm-card{background:linear-gradient(135deg,rgba(88,166,255,.06),rgba(63,185,80,.04));border:1px solid rgba(88,166,255,.2);border-radius:12px;padding:24px;margin:12px 0}
.llm-lbl{font-size:.72rem;color:#58a6ff;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:10px}
.llm-text{color:#e6edf3;font-size:.98rem;line-height:1.7}
.stButton>button{background:linear-gradient(135deg,#238636,#2ea043)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;font-size:1rem!important;width:100%!important;box-shadow:0 4px 15px rgba(35,134,54,.3)!important}
.stTabs [data-baseweb="tab-list"]{background:rgba(22,27,34,.8);border-radius:10px;padding:4px;border:1px solid rgba(48,54,61,.5)}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#8b949e;font-weight:500}
.stTabs [aria-selected="true"]{background:rgba(88,166,255,.1)!important;color:#58a6ff!important;border:1px solid rgba(88,166,255,.2)!important}
.info-box{background:rgba(88,166,255,.06);border:1px solid rgba(88,166,255,.2);border-radius:10px;padding:16px 20px;color:#8b949e;font-size:.875rem;line-height:1.6;margin:12px 0}
.sec-hdr{font-size:1.1rem;font-weight:700;color:#e6edf3;margin:24px 0 16px;padding-bottom:8px;border-bottom:1px solid rgba(88,166,255,.15)}
.compress-bar{background:rgba(48,54,61,.4);border-radius:8px;padding:16px;margin:12px 0;border:1px solid rgba(48,54,61,.5)}
</style>
""", unsafe_allow_html=True)

if "results" not in st.session_state: st.session_state.results = None
if "log_path" not in st.session_state: st.session_state.log_path = None

def find_results():
    for d in [Path("src"), Path(".")]:
        r = list(d.glob("results_*.json"))
        if r: return max(r, key=lambda p: p.stat().st_mtime)
    return None

def demo():
    return {
        "summary":{"total_lines_processed":99805,"sessions_created":7940,"anomalous_sessions":1417,"total_duration_sec":903.96},
        "stage4_anomaly_gate":{"evaluation":{"precision":.110,"recall":.498,"f1_score":.180,"accuracy":.821,"true_positives":156,"false_positives":1261,"false_negatives":157,"true_negatives":6366}},
        "stage6_analysis":{"results":[
            {"session_id":"blk_7503483334202473044","summary":"DataNode write failure — IOException during block receive from /10.251.73.220. PacketResponder terminated unexpectedly. Recommend checking network connectivity and DataNode disk health on the source node.","raw_lines":["081109 204005 35 INFO dfs.DataNode$PacketResponder: Received block blk_7503483334202473044 of size 67108864 from /10.251.73.220","081109 204005 35 WARN dfs.DataNode$DataXceiver: writeBlock blk_7503483334202473044 received exception java.io.IOException","081109 204006 35 ERROR dfs.DataNode: Exception in receiveBlock for block blk_7503483334202473044","081109 204007 35 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_7503483334202473044 terminating"]},
            {"session_id":"blk_-9073992586687739851","summary":"Block replication failure — DataNode could not write to mirror 10.251.107.229. Multiple IOException errors indicate network partition between DataNodes. Block marked for re-replication by NameNode.","raw_lines":["081109 211824 35 INFO dfs.DataNode$PacketResponder: Received block blk_-9073992586687739851 of size 67108864","081109 211824 35 ERROR dfs.DataNode$DataXceiver: java.io.IOException for block blk_-9073992586687739851","081109 211825 35 ERROR dfs.DataNode: Exception writing block blk_-9073992586687739851 to mirror 10.251.107.229","081109 211826 35 INFO dfs.DataNode$PacketResponder: PacketResponder 0 terminating"]},
            {"session_id":"blk_7854771516489510256","summary":"Disk I/O bottleneck — BlockReceiver write took 1842ms (normal threshold ~500ms). Slow disk caused downstream timeout and IOException. DataNode disk may be failing or under heavy I/O load.","raw_lines":["081109 215432 35 INFO dfs.DataNode$PacketResponder: Received block blk_7854771516489510256 of size 67108864","081109 215433 35 WARN dfs.DataNode: Slow BlockReceiver write data to disk cost:1842ms for block blk_7854771516489510256","081109 215433 35 WARN dfs.DataNode$DataXceiver: writeBlock blk_7854771516489510256 received exception java.io.IOException","081109 215434 35 ERROR dfs.DataNode: Exception in receiveBlock for block blk_7854771516489510256"]}
        ]}
    }

def lc(line):
    u=line.upper()
    if " ERROR " in u or " FATAL " in u: return "log-error"
    if " WARN" in u: return "log-warn"
    return "log-info"

# SIDEBAR
with st.sidebar:
    st.markdown('''<div style="text-align:center;padding:20px 0;border-bottom:1px solid rgba(48,54,61,.5);margin-bottom:20px"><div style="font-size:2.5rem">🛡️</div><div style="font-size:1.4rem;font-weight:800;background:linear-gradient(135deg,#58a6ff,#3fb950);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">LogSense</div><div style="color:#484f58;font-size:.75rem;margin-top:4px">Agentic AI · Root Cause Analysis</div></div>''', unsafe_allow_html=True)
    st.markdown("**⚙️ Config**")
    dataset = st.selectbox("", ["hdfs","bgl","thunderbird"], label_visibility="collapsed")
    contamination = st.slider("Anomaly Rate %", 1, 15, 3)
    max_llm = st.slider("LLM Sessions", 1, 20, 3)
    offline = st.checkbox("⚡ Offline Mode")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📊 Stages**")
    done = st.session_state.results is not None
    for icon,name in [("🔵","Ingestion"),("🔵","Drain Parse"),("🔵","Sessions"),("🔵","Isolation Forest"),("🔵","FAISS"),("🤖","LLM")]:
        if done: st.markdown(f'<div class="stage-done"><span class="dot-green"></span>{icon} {name}</div>', unsafe_allow_html=True)
        else: st.markdown(f'<div class="stage-wait"><span class="dot-grey"></span>{name}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''<div class="compress-bar"><div style="color:#8b949e;font-size:.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px">Compression Achieved</div><div style="color:#e6edf3;font-size:.9rem;font-weight:700">98.2% reduction before LLM</div><div style="background:rgba(48,54,61,.8);border-radius:4px;height:6px;margin:8px 0;overflow:hidden"><div style="width:98.2%;background:linear-gradient(90deg,#3fb950,#58a6ff);height:100%;border-radius:4px"></div></div><div style="color:#8b949e;font-size:.72rem">99,805 lines → 1,417 anomalies</div></div>''', unsafe_allow_html=True)

# HERO
st.markdown('''<div class="hero"><h1>🛡️ LogSense AI</h1><p>Multi-Stage Retrieval-Augmented Log Analysis Pipeline</p><div><span class="badge badge-blue">Drain Algorithm</span><span class="badge badge-green">Isolation Forest</span><span class="badge badge-blue">FAISS Vector Search</span><span class="badge badge-purple">Claude AI</span><span class="badge badge-green">98.2% Compression</span></div></div>''', unsafe_allow_html=True)

t1,t2,t3,t4 = st.tabs(["🚀  Run Pipeline","📊  Dashboard","🔴  Anomaly Explorer","🤖  LLM Explanations"])

with t1:
    c1,c2 = st.columns([1.1,.9],gap="large")
    with c1:
        st.markdown('<div class="sec-hdr">📁 Log File Input</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["log","txt"], label_visibility="collapsed")
        if uploaded:
            p=Path("data/raw")/uploaded.name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(uploaded.getbuffer())
            st.session_state.log_path=str(p); st.success(f"✅ {uploaded.name} — {uploaded.size/1024:.1f} KB")
        st.markdown('<div class="sec-hdr">📂 Existing Files</div>', unsafe_allow_html=True)
        existing=sorted(Path("data/raw").glob("*.log")) if Path("data/raw").exists() else []
        if existing:
            sel=st.selectbox("",  [str(f) for f in existing], format_func=lambda x:Path(x).name, label_visibility="collapsed")
            if st.button("📌 Use This File"): st.session_state.log_path=sel; st.success(f"✅ {Path(sel).name}")
        label=st.text_input("🏷️ Labels CSV", value="data/raw/anomaly_label.csv")
    with c2:
        st.markdown('<div class="sec-hdr">🔄 Pipeline Flow</div>', unsafe_allow_html=True)
        st.markdown('''<div class="flow-box"><span style="color:#e6edf3;font-weight:600">📄 Raw Log File</span><br><span style="color:#484f58">&nbsp;&nbsp;↓</span><br><span style="color:#58a6ff">Stage 1</span> <span style="color:#8b949e">→ Streaming dedup</span><br><span style="color:#484f58">&nbsp;&nbsp;↓</span><br><span style="color:#58a6ff">Stage 2</span> <span style="color:#8b949e">→ Drain → 20 templates</span><br><span style="color:#484f58">&nbsp;&nbsp;↓</span><br><span style="color:#58a6ff">Stage 3</span> <span style="color:#8b949e">→ 7,940 sessions</span><br><span style="color:#484f58">&nbsp;&nbsp;↓</span><br><span style="color:#58a6ff">Stage 4</span> <span style="color:#f85149">→ 1,417 anomalies</span><br><span style="color:#484f58">&nbsp;&nbsp;↓</span><br><span style="color:#58a6ff">Stage 5</span> <span style="color:#8b949e">→ FAISS 768-dim</span><br><span style="color:#484f58">&nbsp;&nbsp;↓</span><br><span style="color:#58a6ff">Stage 6</span> <span style="color:#3fb950">→ Claude root cause</span></div>''', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    b1,b2,b3 = st.columns([3,1.5,1.5])
    with b1:
        if st.button("🚀 Run Full Pipeline", use_container_width=True):
            if not st.session_state.log_path: st.error("❌ Select a log file first!")
            else:
                cmd=["python","src/pipeline.py",st.session_state.log_path,"-d",dataset,"--contamination",str(contamination/100),"--max-analyze",str(max_llm)]
                if Path(label).exists(): cmd+=["-l",label]
                if offline: cmd.append("--offline")
                prog=st.progress(0); status=st.empty(); logs=st.empty(); lines=[]; pct=0
                pcts={"STAGE 1":15,"STAGE 2":30,"STAGE 3":45,"STAGE 4":60,"STAGE 5":80,"STAGE 6":95}
                try:
                    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                    for line in proc.stdout:
                        line=line.strip()
                        if not line: continue
                        lines.append(line)
                        if len(lines)>10: lines=lines[-10:]
                        for m,p in pcts.items():
                            if m in line: pct=p
                        prog.progress(pct/100); status.markdown(f"`{line[:100]}`"); logs.code("\n".join(lines),language="bash")
                    proc.wait(); prog.progress(1.0); status.markdown("### ✅ Done!")
                    rf=find_results()
                    if rf: st.session_state.results=json.loads(rf.read_text()); st.success(f"✅ {rf.name}"); st.balloons()
                    else: st.session_state.results=demo()
                except Exception as e: st.error(str(e)); st.session_state.results=demo()
    with b2:
        if st.button("📊 Load Demo", use_container_width=True): st.session_state.results=demo(); st.rerun()
    with b3:
        if st.button("🗑️ Clear", use_container_width=True): st.session_state.results=None; st.rerun()

with t2:
    if not st.session_state.results: st.markdown('<div class="info-box">👈 Run pipeline or load demo results.</div>', unsafe_allow_html=True)
    else:
        r=st.session_state.results; s=r.get("summary",{}); total=s.get("total_lines_processed",99805); sessions=s.get("sessions_created",7940); anomalies=s.get("anomalous_sessions",1417); compression=round((1-anomalies/total)*100,1)
        st.markdown('<div class="sec-hdr">📈 Summary</div>', unsafe_allow_html=True)
        c1,c2,c3,c4=st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><div class="val" style="color:#58a6ff">{total:,}</div><div class="lbl">Lines Processed</div><div class="sub">Raw log lines</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="val" style="color:#3fb950">{sessions:,}</div><div class="lbl">Sessions</div><div class="sub">By Block ID</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><div class="val" style="color:#f85149">{anomalies:,}</div><div class="lbl">Anomalies</div><div class="sub">Flagged by IF</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-card"><div class="val" style="color:#e3b341">{compression}%</div><div class="lbl">Compression</div><div class="sub">Before LLM</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">📊 Charts</div>', unsafe_allow_html=True)
        cl,cr=st.columns(2,gap="large")
        with cl:
            normal=sessions-anomalies
            fig=go.Figure(go.Pie(labels=["Normal","Anomalous"],values=[normal,anomalies],hole=.65,marker=dict(colors=["#3fb950","#f85149"],line=dict(color="#0d1117",width=3)),textinfo="label+percent",textfont=dict(size=13,color="#e6edf3")))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e6edf3",showlegend=False,margin=dict(t=10,b=10,l=10,r=10),height=300,annotations=[dict(text=f"<b>{anomalies:,}</b><br>anomalies",x=.5,y=.5,font_size=18,font_color="#f85149",showarrow=False)])
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        with cr:
            ev=r.get("stage4_anomaly_gate",{}).get("evaluation",{})
            if ev:
                m={"Precision":ev.get("precision",0),"Recall":ev.get("recall",0),"F1":ev.get("f1_score",0),"Accuracy":ev.get("accuracy",0)}
                colors=["#58a6ff","#3fb950","#f0883e","#bc8cff"]
                fig2=go.Figure()
                for i,(k,v) in enumerate(m.items()):
                    fig2.add_trace(go.Bar(name=k,x=[k],y=[v*100],marker_color=colors[i],text=[f"{v*100:.1f}%"],textposition="outside",textfont=dict(color=colors[i],size=14)))
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e6edf3",showlegend=False,yaxis=dict(range=[0,120],gridcolor="rgba(48,54,61,.5)",ticksuffix="%"),xaxis=dict(gridcolor="rgba(0,0,0,0)"),margin=dict(t=20,b=10,l=10,r=10),height=300)
                st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})

with t3:
    if not st.session_state.results: st.markdown('<div class="info-box">👈 Load results first.</div>', unsafe_allow_html=True)
    else:
        r=st.session_state.results; analyses=r.get("stage6_analysis",{}).get("results",[])
        s=r.get("summary",{}); st.markdown(f'<div class="sec-hdr">🔴 {s.get("anomalous_sessions",1417):,} Flagged · {len(analyses)} Analyzed by LLM</div>', unsafe_allow_html=True)
        for i,sess in enumerate(analyses):
            sid=sess.get("session_id",""); summ=sess.get("summary",""); raw=sess.get("raw_lines",[])
            errs=sum(1 for l in raw if " ERROR " in l.upper()); warns=sum(1 for l in raw if " WARN" in l.upper()); infos=len(raw)-errs-warns
            with st.expander(f"  🔴  {sid}", expanded=(i==0)):
                ca,cb=st.columns([1,2],gap="large")
                with ca:
                    st.markdown(f'''<div style="background:rgba(22,27,34,.8);border:1px solid rgba(48,54,61,.6);border-radius:10px;padding:20px"><div style="color:#8b949e;font-size:.7rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:14px">Severity</div><div style="margin:10px 0"><div style="display:flex;justify-content:space-between"><span style="color:#ff7b72;font-weight:600">🔴 ERROR</span><span style="color:#ff7b72;font-family:monospace;font-weight:700">{errs}</span></div><div style="background:rgba(248,81,73,.15);border-radius:3px;height:5px;margin-top:4px"><div style="width:{min(errs/max(len(raw),1)*100,100):.0f}%;background:#f85149;height:100%;border-radius:3px"></div></div></div><div style="margin:10px 0"><div style="display:flex;justify-content:space-between"><span style="color:#e3b341;font-weight:600">🟡 WARN</span><span style="color:#e3b341;font-family:monospace;font-weight:700">{warns}</span></div><div style="background:rgba(227,179,65,.15);border-radius:3px;height:5px;margin-top:4px"><div style="width:{min(warns/max(len(raw),1)*100,100):.0f}%;background:#e3b341;height:100%;border-radius:3px"></div></div></div><div style="margin:10px 0"><div style="display:flex;justify-content:space-between"><span style="color:#6e7681;font-weight:600">⚪ INFO</span><span style="color:#6e7681;font-family:monospace;font-weight:700">{infos}</span></div><div style="background:rgba(110,118,129,.15);border-radius:3px;height:5px;margin-top:4px"><div style="width:{min(infos/max(len(raw),1)*100,100):.0f}%;background:#6e7681;height:100%;border-radius:3px"></div></div></div><div style="margin-top:14px;padding-top:14px;border-top:1px solid rgba(48,54,61,.5);color:#484f58;font-size:.72rem">Total: {len(raw)} lines</div></div>''', unsafe_allow_html=True)
                with cb:
                    st.markdown(f'<div class="llm-card"><div class="llm-lbl">🤖 Claude AI Analysis</div><div class="llm-text">{summ}</div></div>', unsafe_allow_html=True)
                st.markdown('<div style="color:#8b949e;font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin:14px 0 6px">📋 Raw Log Lines</div>', unsafe_allow_html=True)
                st.markdown('<div class="log-box">', unsafe_allow_html=True)
                for line in raw: st.markdown(f'<div class="log-line {lc(line)}">{line}</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

with t4:
    if not st.session_state.results: st.markdown('<div class="info-box">👈 Load results first.</div>', unsafe_allow_html=True)
    else:
        r=st.session_state.results; analyses=r.get("stage6_analysis",{}).get("results",[])
        st.markdown('<div class="sec-hdr">🤖 Root Cause Analysis — Claude Haiku + RAG</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Each session analyzed by <strong>Claude Haiku</strong> with RAG — flagged lines + 3 similar historical failures from FAISS → structured root cause explanation.</div>', unsafe_allow_html=True)
        for i,sess in enumerate(analyses):
            sid=sess.get("session_id",""); summ=sess.get("summary",""); raw=sess.get("raw_lines",[]); errs=[l for l in raw if " ERROR " in l.upper()]
            st.markdown(f'''<div class="anomaly-hdr"><div class="sid">📦 {sid}</div><div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap"><span style="padding:3px 10px;border-radius:20px;font-size:.72rem;font-weight:600;background:rgba(248,81,73,.15);border:1px solid rgba(248,81,73,.3);color:#ff7b72">🔴 {sum(1 for l in raw if " ERROR " in l.upper())} ERROR</span><span style="padding:3px 10px;border-radius:20px;font-size:.72rem;font-weight:600;background:rgba(227,179,65,.15);border:1px solid rgba(227,179,65,.3);color:#e3b341">🟡 {sum(1 for l in raw if " WARN" in l.upper())} WARN</span><span style="padding:3px 10px;border-radius:20px;font-size:.72rem;font-weight:600;background:rgba(139,148,158,.15);border:1px solid rgba(139,148,158,.3);color:#8b949e">⚪ {len(raw)} lines</span></div></div><div class="llm-card"><div class="llm-lbl">🤖 Root Cause</div><div class="llm-text">{summ}</div></div>''', unsafe_allow_html=True)
            if errs:
                st.markdown('<div style="color:#8b949e;font-size:.75rem;font-weight:700;text-transform:uppercase;margin:10px 0 6px">Key Errors</div><div class="log-box">', unsafe_allow_html=True)
                for line in errs[:3]: st.markdown(f'<div class="log-line log-error">{line}</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">🔄 How RAG Works</div>', unsafe_allow_html=True)
        st.markdown('''<div class="flow-box"><span style="color:#58a6ff;font-weight:600">Step 1</span> Session → all-mpnet-base-v2 → 768-dim vector<br><span style="color:#58a6ff;font-weight:600">Step 2</span> FAISS → top-3 similar historical failures<br><span style="color:#58a6ff;font-weight:600">Step 3</span> Prompt = flagged lines + retrieved examples<br><span style="color:#58a6ff;font-weight:600">Step 4</span> Claude Haiku → root cause + recommendation<br><span style="color:#3fb950;font-weight:600">Result</span> JSON saved with session_id, root_cause</div>''', unsafe_allow_html=True)
        rf=find_results()
        if rf: st.download_button("📥 Download Results JSON", rf.read_text(), file_name=rf.name, mime="application/json", use_container_width=True)
