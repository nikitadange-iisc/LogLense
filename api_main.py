from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import subprocess, json, glob, os

app = FastAPI(title="LogSense API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

state = {"running":False,"stage":0,"stage_name":"","log":[],"error":None,"done":False}

class RunRequest(BaseModel):
    log_file: str
    dataset: str = "hdfs"
    contamination: float = 0.03
    max_sessions: int = 5
    label_path: str = "data/raw/anomaly_label.csv"
    offline: bool = False

def run_stage(cmd, name):
    state["stage_name"]=name
    state["log"].append(f"▶ {name}")
    r=subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout+r.stderr).split("\n"):
        if line.strip(): state["log"].append(line)
    if r.returncode!=0: raise Exception(f"{name} failed: {r.stderr[-300:]}")
    state["log"].append(f"✅ {name} complete")
    state["stage"]+=1

def pipeline(req):
    try:
        state.update({"running":True,"stage":0,"log":[],"error":None,"done":False})
        stem=Path(req.log_file).stem
        csv_f=f"data/processed/{stem}_structured.csv"
        json_f=f"data/processed/{stem}_anomalies.json"

        run_stage(["python","src/module1_ingest_parse.py",req.log_file,
                   "--dataset",req.dataset,"--max-lines","100000"],
                  "Module 1: Ingestion & Parsing")

        cmd2=["python","src/module2_session_anomaly.py",csv_f,
              "--dataset",req.dataset,"--contamination",str(req.contamination)]
        if Path(req.label_path).exists():
            cmd2+=["--label-path",req.label_path]
        run_stage(cmd2,"Module 2: Anomaly Detection")

        run_stage(["python","src/module3_embed_index.py",json_f,
                   "--dataset",req.dataset,"--model","all-MiniLM-L6-v2"],
                  "Module 3: FAISS Embedding")

        if not req.offline:
            run_stage(["python","src/module4_rag_analysis.py",json_f,
                       "--dataset",req.dataset,"--max-sessions",str(req.max_sessions)],
                      "Module 4: LLM Analysis")
        else:
            run_stage(["python","src/module4_rag_analysis.py",json_f,
                       "--dataset",req.dataset,"--max-sessions",str(req.max_sessions),"--offline"],
                      "Module 4: Offline (no LLM)")

        state["log"].append("🎉 Pipeline complete!")
        state["done"]=True
    except Exception as e:
        state["error"]=str(e)
        state["log"].append(f"❌ {e}")
    finally:
        state["running"]=False

@app.get("/api/files")
def files():
    f=list(Path("data/raw").glob("*.log")) if Path("data/raw").exists() else []
    return {"files":[str(x) for x in sorted(f)]}

@app.post("/api/run")
def run(req:RunRequest, bg:BackgroundTasks):
    if state["running"]: raise HTTPException(400,"Pipeline already running")
    bg.add_task(pipeline,req)
    return {"status":"started"}

@app.get("/api/status")
def status():
    return {**state,"log_tail":state["log"][-30:]}

@app.get("/api/results")
def results():
    files=glob.glob("data/processed/*rag_results*.json")+glob.glob("src/results_*.json")
    if not files: return {"results":None}
    latest=max(files,key=os.path.getmtime)
    return {"results":json.loads(Path(latest).read_text()),"file":Path(latest).name}

@app.get("/api/history")
def history():
    files=glob.glob("data/processed/*rag_results*.json")+glob.glob("src/results_*.json")
    out=[]
    for f in sorted(files,key=os.path.getmtime,reverse=True)[:8]:
        try:
            d=json.loads(Path(f).read_text())
            out.append({"file":Path(f).name,
                        "provider":d.get("llm_provider",d.get("provider","?")),
                        "model":d.get("llm_model",d.get("model","?")),
                        "analyzed":d.get("sessions_analysed",len(d.get("results",[]))),
                        "mtime":os.path.getmtime(f)})
        except: pass
    return {"history":out}

# Serve React UI — must be LAST
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/")
def index():
    return FileResponse("ui/index.html")
