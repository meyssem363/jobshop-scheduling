"""
Job Shop Scheduling Dashboard — Semaine 3
Données réelles : 10 jobs industriels × 5 machines
Contraintes : Pannes machines / Due Dates / Setup Time
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
import time
import collections
import math
import pandas as pd
from ortools.sat.python import cp_model

st.set_page_config(
    page_title="Job Shop Scheduling",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:500; }
    .badge-good { background:#E1F5EE; color:#0F6E56; }
    .badge-mid  { background:#FAEEDA; color:#854F0B; }
    .badge-bad  { background:#FCEBEB; color:#A32D2D; }
    .stTabs [data-baseweb="tab"] { font-size:14px; }
</style>
""", unsafe_allow_html=True)


# ─── Données fixes ────────────────────────────────────────────────────────────
MACHINES = ["Tour CNC", "Fraiseuse", "Rectifieuse", "Perceuse", "Montage"]
JOBS = ["Vilebrequin", "Soupape", "Piston", "Bielle", "Culasse",
        "Arbre cames", "Segment", "Palier", "Joint culasse", "Turbo"]
NUM_MACHINES = len(MACHINES)
NUM_JOBS     = len(JOBS)

ROUTES = [
    [0,2,1,4,3],[1,0,3,2,4],[2,1,0,4,3],[0,3,2,1,4],[1,2,3,0,4],
    [3,0,1,2,4],[0,1,2,3,4],[4,2,0,1,3],[2,3,1,0,4],[1,3,0,2,4],
]
DURATIONS = np.random.default_rng(42).integers(5, 25, size=(NUM_JOBS, NUM_MACHINES)).tolist()

DUE_DATES = [150, 220, 180, 120, 200, 160, 250, 210, 140, 230]

BREAKDOWNS = [
    (1, 40, 55),
    (3, 60, 75),
]

SETUP_TIME = 5

COLORS_JOBS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
               "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]


# ─── Algorithme aléatoire ─────────────────────────────────────────────────────
def solve_random(seed=7):
    random.seed(seed)
    order = list(range(NUM_JOBS))
    random.shuffle(order)
    machine_free = [0] * NUM_MACHINES
    job_free     = [0] * NUM_JOBS
    schedule     = {j: [] for j in range(NUM_JOBS)}
    op_index     = [0] * NUM_JOBS
    for _ in range(NUM_JOBS * NUM_MACHINES):
        for j in order:
            op = op_index[j]
            if op >= NUM_MACHINES: continue
            m, dur = ROUTES[j][op], DURATIONS[j][op]
            start = max(job_free[j], machine_free[m])
            end   = start + dur
            schedule[j].append({"op":op,"machine":m,"start":start,"end":end,"duration":dur})
            job_free[j] = end; machine_free[m] = end; op_index[j] += 1
            break
    return {"method":"Aléatoire","makespan":int(max(machine_free)),"schedule":schedule}


# ─── SPT ──────────────────────────────────────────────────────────────────────
def solve_spt():
    order = sorted(range(NUM_JOBS), key=lambda j: sum(DURATIONS[j]))
    machine_free = [0] * NUM_MACHINES
    job_free     = [0] * NUM_JOBS
    schedule     = {j: [] for j in range(NUM_JOBS)}
    op_index     = [0] * NUM_JOBS
    for _ in range(NUM_JOBS * NUM_MACHINES):
        for j in order:
            op = op_index[j]
            if op >= NUM_MACHINES: continue
            m, dur = ROUTES[j][op], DURATIONS[j][op]
            start = max(job_free[j], machine_free[m])
            end   = start + dur
            schedule[j].append({"op":op,"machine":m,"start":start,"end":end,"duration":dur})
            job_free[j] = end; machine_free[m] = end; op_index[j] += 1
            break
    return {"method":"SPT","makespan":int(max(machine_free)),"schedule":schedule}


# ─── OR-Tools base ────────────────────────────────────────────────────────────
def solve_ortools_base(time_limit=30):
    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()
    horizon = sum(DURATIONS[j][o] for j in range(NUM_JOBS) for o in range(NUM_MACHINES)) + 100
    Task = collections.namedtuple("Task","start end interval")
    all_tasks, machine_intervals = {}, collections.defaultdict(list)

    for j in range(NUM_JOBS):
        for op in range(NUM_MACHINES):
            m, dur = ROUTES[j][op], DURATIONS[j][op]
            s = model.NewIntVar(0, horizon, f"s{j}_{op}")
            e = model.NewIntVar(0, horizon, f"e{j}_{op}")
            i = model.NewIntervalVar(s, dur, e, f"i{j}_{op}")
            all_tasks[(j,op)] = Task(s,e,i)
            machine_intervals[m].append(i)

    for m in range(NUM_MACHINES):
        model.AddNoOverlap(machine_intervals[m])
    for j in range(NUM_JOBS):
        for op in range(NUM_MACHINES-1):
            model.Add(all_tasks[(j,op+1)].start >= all_tasks[(j,op)].end)

    ms_var = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(ms_var, [all_tasks[(j,NUM_MACHINES-1)].end for j in range(NUM_JOBS)])
    model.Minimize(ms_var)
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = False
    solver.Solve(model)

    schedule = {j:[] for j in range(NUM_JOBS)}
    for j in range(NUM_JOBS):
        for op in range(NUM_MACHINES):
            m, dur = ROUTES[j][op], DURATIONS[j][op]
            s = solver.Value(all_tasks[(j,op)].start)
            schedule[j].append({"op":op,"machine":m,"start":s,"end":s+dur,"duration":dur})
    return {"method":"OR-Tools","makespan":int(solver.ObjectiveValue()),
            "schedule":schedule,"tardiness":None,"breakdowns":[]}


# ─── OR-Tools avec contraintes ────────────────────────────────────────────────
def solve_ortools_constraints(use_breakdowns, use_due_dates, use_setup, time_limit=30):
    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()
    horizon = sum(DURATIONS[j][o] for j in range(NUM_JOBS) for o in range(NUM_MACHINES)) + 300
    Task = collections.namedtuple("Task","start end interval")
    all_tasks, machine_intervals = {}, collections.defaultdict(list)

    for j in range(NUM_JOBS):
        for op in range(NUM_MACHINES):
            m, dur = ROUTES[j][op], DURATIONS[j][op]
            s = model.NewIntVar(0, horizon, f"s{j}_{op}")
            e = model.NewIntVar(0, horizon, f"e{j}_{op}")
            i = model.NewIntervalVar(s, dur, e, f"i{j}_{op}")
            all_tasks[(j,op)] = Task(s,e,i)
            machine_intervals[m].append(i)

    # Pannes machines
    active_breakdowns = []
    if use_breakdowns:
        for m_id, bd_s, bd_e in BREAKDOWNS:
            dur_bd = bd_e - bd_s
            s_bd = model.NewConstant(bd_s)
            e_bd = model.NewConstant(bd_e)
            i_bd = model.NewIntervalVar(s_bd, dur_bd, e_bd, f"breakdown_m{m_id}_{bd_s}")
            machine_intervals[m_id].append(i_bd)
            active_breakdowns.append((m_id, bd_s, bd_e))

    for m in range(NUM_MACHINES):
        model.AddNoOverlap(machine_intervals[m])
    for j in range(NUM_JOBS):
        for op in range(NUM_MACHINES-1):
            model.Add(all_tasks[(j,op+1)].start >= all_tasks[(j,op)].end)

    # Setup time
    if use_setup:
        for machine in range(NUM_MACHINES):
            machine_tasks = [(j,op) for j in range(NUM_JOBS)
                             for op in range(NUM_MACHINES) if ROUTES[j][op] == machine]
            for i in range(len(machine_tasks)):
                for k in range(i+1, len(machine_tasks)):
                    j1,op1 = machine_tasks[i]
                    j2,op2 = machine_tasks[k]
                    t1 = all_tasks[(j1,op1)]; t2 = all_tasks[(j2,op2)]
                    before = model.NewBoolVar(f"before_m{machine}_{j1}_{j2}")
                    model.Add(t2.start >= t1.end + SETUP_TIME).OnlyEnforceIf(before)
                    model.Add(t1.start >= t2.end + SETUP_TIME).OnlyEnforceIf(before.Not())

    # Due dates
    tardiness = []
    if use_due_dates:
        for j in range(NUM_JOBS):
            last_end = all_tasks[(j, NUM_MACHINES-1)].end
            tard = model.NewIntVar(0, horizon, f"tard_{j}")
            model.AddMaxEquality(tard, [model.NewConstant(0), last_end - DUE_DATES[j]])
            tardiness.append(tard)

    ms_var = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(ms_var, [all_tasks[(j,NUM_MACHINES-1)].end for j in range(NUM_JOBS)])

    if use_due_dates and tardiness:
        model.Minimize(ms_var + 10 * sum(tardiness))
    else:
        model.Minimize(ms_var)

    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = False
    solver.Solve(model)

    schedule = {j:[] for j in range(NUM_JOBS)}
    for j in range(NUM_JOBS):
        for op in range(NUM_MACHINES):
            m, dur = ROUTES[j][op], DURATIONS[j][op]
            s = solver.Value(all_tasks[(j,op)].start)
            schedule[j].append({"op":op,"machine":m,"start":s,"end":s+dur,"duration":dur})

    tard_vals = None
    if use_due_dates and tardiness:
        tard_vals = {j: max(0, solver.Value(all_tasks[(j,NUM_MACHINES-1)].end) - DUE_DATES[j])
                     for j in range(NUM_JOBS)}

    return {"method":"OR-Tools","makespan":int(solver.Value(ms_var)),
            "schedule":schedule,"tardiness":tard_vals,"breakdowns":active_breakdowns}


# ─── Gantt ────────────────────────────────────────────────────────────────────
def plot_gantt(result, title="Gantt"):
    schedule  = result["schedule"]
    makespan  = result["makespan"]
    breakdowns = result.get("breakdowns", [])
    tard_vals  = result.get("tardiness")

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    for j in range(NUM_JOBS):
        for op in schedule[j]:
            ax.barh(op["machine"], op["duration"], left=op["start"],
                    height=0.6, color=COLORS_JOBS[j], edgecolor="white",
                    linewidth=0.8, alpha=0.88)
            if op["duration"] > makespan * 0.04:
                ax.text(op["start"] + op["duration"]/2, op["machine"],
                        JOBS[j][:4], ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white")

    for m_id, bd_s, bd_e in breakdowns:
        ax.barh(m_id, bd_e-bd_s, left=bd_s, height=0.7,
                color="#ff000055", edgecolor="red", linewidth=1.5, hatch="////")
        ax.text((bd_s+bd_e)/2, m_id+0.42, f"PANNE\n{bd_s}→{bd_e}",
                ha="center", va="bottom", fontsize=7, color="red", fontweight="bold")
        ax.axvspan(bd_s, bd_e, alpha=0.05, color="red")

    if tard_vals:
        for j in range(NUM_JOBS):
            fin = schedule[j][-1]["end"]
            due = DUE_DATES[j]
            color = "#E24B4A" if tard_vals[j] > 0 else "#1D9E75"
            ax.axvline(due, color=color, linewidth=1, alpha=0.7, linestyle="--")

    ax.set_yticks(range(NUM_MACHINES))
    ax.set_yticklabels(MACHINES, fontsize=10)
    ax.set_xlabel("Temps (minutes)", fontsize=10)
    ax.set_title(f"{title} — Makespan : {makespan} min", fontsize=12, fontweight="bold")
    ax.axvline(makespan, color="#E24B4A", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlim(0, makespan * 1.05)
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    patches = [mpatches.Patch(color=COLORS_JOBS[j], label=JOBS[j]) for j in range(NUM_JOBS)]
    if breakdowns:
        patches.append(mpatches.Patch(facecolor="#ff000055", edgecolor="red",
                                       hatch="////", label="Panne machine"))
    ax.legend(handles=patches, loc="upper right", ncol=2, fontsize=8, framealpha=0.7)
    plt.tight_layout()
    return fig


# ─── Comparaison ──────────────────────────────────────────────────────────────
def plot_comparison(results_list):
    labels = [r["method"] for r in results_list]
    values = [r["makespan"] for r in results_list]
    colors = ["#d62728","#ff7f0e","#2ca02c"][:len(results_list)]
    ref    = results_list[0]["makespan"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("white")

    bars = ax1.bar(labels, values, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+ref*0.01,
                 f"{val} min", ha="center", va="bottom", fontweight="bold", fontsize=11)
    for i in range(1, len(results_list)):
        reduc = (ref - values[i]) / ref * 100
        sign  = f"−{reduc:.1f}%" if reduc > 0 else f"+{abs(reduc):.1f}%"
        ax1.annotate(sign, xy=(i, values[i]), xytext=(i, ref*1.12),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.3),
                     ha="center", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Makespan (minutes)", fontsize=10)
    ax1.set_title("Makespan par méthode", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, ref*1.4)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines[["top","right"]].set_visible(False)

    reductions = [0]+[(ref-v)/ref*100 for v in values[1:]]
    bars2 = ax2.bar(labels, reductions, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars2, reductions):
        if val > 0:
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f"−{val:.1f}%", ha="center", va="bottom",
                     fontweight="bold", color="#0F6E56", fontsize=11)
    ax2.set_ylabel("Réduction vs aléatoire (%)", fontsize=10)
    ax2.set_title("Gain par rapport à l'aléatoire", fontsize=11, fontweight="bold")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Job Shop Scheduling")
st.sidebar.markdown("---")

st.sidebar.subheader("Méthodes")
show_rand = st.sidebar.checkbox("Aléatoire", value=True)
show_spt  = st.sidebar.checkbox("SPT", value=True)
show_or   = st.sidebar.checkbox("OR-Tools (CP-SAT)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Contraintes réelles")
use_breakdowns = st.sidebar.toggle("🔧 Pannes machines", value=False)
use_due_dates  = st.sidebar.toggle("📅 Due dates (délais)", value=False)
use_setup      = st.sidebar.toggle("⏱️ Setup time (5 min)", value=False)

if use_breakdowns:
    st.sidebar.caption(f"Fraiseuse : indisponible t=40→55")
    st.sidebar.caption(f"Perceuse  : indisponible t=60→75")
if use_due_dates:
    st.sidebar.caption("Pénalité retard × 10 dans l'objectif")
if use_setup:
    st.sidebar.caption(f"Délai de {SETUP_TIME} min entre 2 jobs sur même machine")

st.sidebar.markdown("---")
or_time = st.sidebar.slider("Limite temps OR-Tools (s)", 5, 60, 30)

with st.sidebar.expander("📋 Jobs"):
    for j, name in enumerate(JOBS):
        dd = f" — due: {DUE_DATES[j]} min" if use_due_dates else ""
        st.write(f"J{j} {name}{dd}")
with st.sidebar.expander("🔧 Machines"):
    for m, name in enumerate(MACHINES):
        st.write(f"M{m} — {name}")

run_btn = st.sidebar.button("▶ Lancer l'optimisation", type="primary", use_container_width=True)
st.sidebar.caption("Job Shop Scheduling — Atelier Mécanique")


# ─── Titre ────────────────────────────────────────────────────────────────────
contraintes_actives = []
if use_breakdowns: contraintes_actives.append("Pannes")
if use_due_dates:  contraintes_actives.append("Due Dates")
if use_setup:      contraintes_actives.append("Setup Time")
contraintes_str = " · ".join(contraintes_actives) if contraintes_actives else "Modèle de base"

st.title("📊 Job Shop Scheduling — Atelier Mécanique")
st.markdown(
    f"**{NUM_JOBS} pièces** × **{NUM_MACHINES} machines** · "
    f"Contraintes actives : **{contraintes_str}**"
)

# ─── Session state ────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

needs_constraints = use_breakdowns or use_due_dates or use_setup

if run_btn or st.session_state.results is None:
    res = {}
    with st.spinner("Calcul en cours…"):
        if show_rand:
            t0 = time.time()
            res["random"] = solve_random()
            res["random"]["time"] = round(time.time()-t0, 4)
        if show_spt:
            t0 = time.time()
            res["spt"] = solve_spt()
            res["spt"]["time"] = round(time.time()-t0, 4)
        if show_or:
            t0 = time.time()
            if needs_constraints:
                res["ortools"] = solve_ortools_constraints(
                    use_breakdowns, use_due_dates, use_setup, or_time)
            else:
                res["ortools"] = solve_ortools_base(or_time)
            res["ortools"]["time"] = round(time.time()-t0, 2)
    st.session_state.results = res

results = st.session_state.results
if not results:
    st.warning("Sélectionnez au moins une méthode et cliquez sur Lancer.")
    st.stop()

# ─── Métriques ────────────────────────────────────────────────────────────────
ref  = results.get("random", {}).get("makespan")
cols = st.columns(3)
for i, (key, label) in enumerate([("random","Aléatoire"),("spt","SPT"),("ortools","OR-Tools")]):
    with cols[i]:
        if key in results:
            ms = results[key]["makespan"]
            if ref and key != "random":
                delta = f"{round((1-ms/ref)*100)}%"
                st.metric(label=f"{label} (min)", value=ms, delta=delta, delta_color="inverse")
            else:
                st.metric(label=f"{label} (min)", value=ms)
        else:
            st.metric(label=label, value="—")

# ─── Onglets ──────────────────────────────────────────────────────────────────
tabs = st.tabs(["📈 Gantt", "📊 Comparaison", "📋 Tableau", "ℹ️ Statistiques"])

# ── Gantt ─────────────────────────────────────────────────────────────────────
with tabs[0]:
    method_options = {k:v for k,v in
                      [("random","Aléatoire"),("spt","SPT"),("ortools","OR-Tools")]
                      if k in results}
    gantt_key = st.selectbox("Méthode", list(method_options.keys()),
                              format_func=lambda k: method_options[k])
    fig = plot_gantt(results[gantt_key], title=method_options[gantt_key])
    st.pyplot(fig); plt.close(fig)

# ── Comparaison ───────────────────────────────────────────────────────────────
with tabs[1]:
    compare_list = []
    for key, label in [("random","Aléatoire"),("spt","SPT"),("ortools","OR-Tools")]:
        if key in results:
            compare_list.append({**results[key], "method": label})
    if len(compare_list) >= 2:
        fig2 = plot_comparison(compare_list)
        st.pyplot(fig2); plt.close(fig2)
    else:
        st.info("Sélectionnez au moins 2 méthodes.")

# ── Tableau ───────────────────────────────────────────────────────────────────
with tabs[2]:
    rows = []
    for key, label in [("random","Aléatoire"),("spt","SPT"),("ortools","OR-Tools")]:
        if key not in results: continue
        ms = results[key]["makespan"]
        t  = results[key].get("time","—")
        delta = f"−{round((1-ms/ref)*100)}%" if ref and key != "random" else "référence"
        rows.append({"Méthode":label,"Makespan (min)":ms,"Réduction":delta,"Temps (s)":t})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Méthode"), use_container_width=True)

    # Tableau due dates si actif
    if use_due_dates and "ortools" in results and results["ortools"].get("tardiness"):
        st.markdown("---")
        st.subheader("📅 Suivi des délais de livraison")
        tard_vals = results["ortools"]["tardiness"]
        sched     = results["ortools"]["schedule"]
        dd_rows = []
        for j in range(NUM_JOBS):
            fin   = sched[j][-1]["end"]
            tard  = tard_vals[j]
            status = f"✗ +{tard} min" if tard > 0 else "✓ À l'heure"
            dd_rows.append({"Job":JOBS[j],"Fin réelle":f"{fin} min",
                            "Due Date":f"{DUE_DATES[j]} min",
                            "Retard":f"{tard} min","Statut":status})
        df_dd = pd.DataFrame(dd_rows)
        st.dataframe(df_dd, use_container_width=True, height=300)
        total_tard = sum(tard_vals.values())
        jobs_retard = sum(1 for t in tard_vals.values() if t > 0)
        col1, col2 = st.columns(2)
        col1.metric("Jobs en retard", f"{jobs_retard} / {NUM_JOBS}")
        col2.metric("Retard total", f"{total_tard} min")

    # Détail opérations OR-Tools
    if "ortools" in results:
        st.markdown("---")
        st.subheader("Détail des opérations OR-Tools")
        ops_rows = []
        for j in range(NUM_JOBS):
            for op in results["ortools"]["schedule"][j]:
                ops_rows.append({"Job":JOBS[j],"Machine":MACHINES[op["machine"]],
                                  "Début (min)":op["start"],"Fin (min)":op["end"],
                                  "Durée (min)":op["duration"]})
        st.dataframe(pd.DataFrame(ops_rows), use_container_width=True, height=300)

# ── Statistiques ──────────────────────────────────────────────────────────────
with tabs[3]:
    total_work = sum(DURATIONS[j][o] for j in range(NUM_JOBS) for o in range(NUM_MACHINES))
    lb = max(total_work // NUM_MACHINES,
             max(sum(DURATIONS[j]) for j in range(NUM_JOBS)))
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuration")
        st.write(f"- **Jobs :** {NUM_JOBS}")
        st.write(f"- **Machines :** {NUM_MACHINES}")
        st.write(f"- **Travail total :** {total_work} min")
        st.write(f"- **Lower bound :** {lb} min")
        if use_setup:
            st.write(f"- **Setup time :** {SETUP_TIME} min")
        if use_breakdowns:
            st.write(f"- **Pannes :** {len(BREAKDOWNS)} créneaux bloqués")
        st.subheader("Durées des opérations (min)")
        df_dur = pd.DataFrame(DURATIONS, index=JOBS, columns=MACHINES)
        st.dataframe(df_dur, use_container_width=True)

    with col2:
        st.subheader("Performance OR-Tools")
        if "ortools" in results:
            ms_or = results["ortools"]["makespan"]
            util  = round(total_work / (ms_or * NUM_MACHINES) * 100, 1)
            gap   = round((ms_or - lb) / lb * 100, 1)
            st.write(f"- **Makespan :** {ms_or} min")
            st.write(f"- **Utilisation machines :** {util}%")
            st.write(f"- **Écart lower bound :** +{gap}%")
            st.write(f"- **Temps calcul :** {results['ortools'].get('time','—')} s")
        st.subheader("Routes technologiques")
        df_routes = pd.DataFrame(
            [[MACHINES[ROUTES[j][op]] for op in range(NUM_MACHINES)] for j in range(NUM_JOBS)],
            index=JOBS, columns=[f"Op {i+1}" for i in range(NUM_MACHINES)])
        st.dataframe(df_routes, use_container_width=True)
