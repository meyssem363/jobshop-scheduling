"""
Job Shop Scheduling Dashboard 
Comparaison : Aléatoire / SPT / OR-Tools
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
import time
import pandas as pd
from ortools.sat.python import cp_model

st.set_page_config(
    page_title="Job Shop Scheduling",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS custom ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #ccc;
    }
    .metric-card.green  { border-left-color: #1D9E75; }
    .metric-card.orange { border-left-color: #EF9F27; }
    .metric-card.red    { border-left-color: #E24B4A; }
    .metric-number { font-size: 2rem; font-weight: 600; }
    .badge { display:inline-block; padding:2px 8px; border-radius:4px;
             font-size:12px; font-weight:500; }
    .badge-good { background:#E1F5EE; color:#0F6E56; }
    .badge-mid  { background:#FAEEDA; color:#854F0B; }
    .badge-bad  { background:#FCEBEB; color:#A32D2D; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# ─── Génération des données ────────────────────────────────────────────────────
def generate_jobs(n_jobs: int, n_machines: int, max_duration: int, seed: int):
    rng = random.Random(seed)
    jobs = []
    for _ in range(n_jobs):
        machines = list(range(n_machines))
        rng.shuffle(machines)
        ops = [(m, rng.randint(1, max_duration)) for m in machines]
        jobs.append(ops)
    return jobs


# ─── Algorithme 1 : Aléatoire ─────────────────────────────────────────────────
def schedule_random(jobs, n_machines, seed):
    order = list(range(len(jobs)))
    random.Random(seed + 1).shuffle(order)
    return _schedule_by_order(jobs, n_machines, order)


# ─── Algorithme 2 : SPT (Shortest Processing Time) ────────────────────────────
def schedule_spt(jobs, n_machines):
    order = sorted(range(len(jobs)), key=lambda j: sum(d for _, d in jobs[j]))
    return _schedule_by_order(jobs, n_machines, order)


def _schedule_by_order(jobs, n_machines, order):
    mach_end = [0] * n_machines
    job_end  = [0] * len(jobs)
    schedule = [[] for _ in jobs]
    for j in order:
        for m, dur in jobs[j]:
            start = max(mach_end[m], job_end[j])
            end   = start + dur
            schedule[j].append({"machine": m, "start": start, "end": end, "duration": dur})
            mach_end[m] = end
            job_end[j]  = end
    makespan = max(mach_end)
    return {"schedule": schedule, "makespan": makespan}


# ─── Algorithme 3 : OR-Tools CP-SAT ──────────────────────────────────────────
def schedule_ortools(jobs, n_machines, time_limit=10):
    model = cp_model.CpModel()
    horizon = sum(d for job in jobs for _, d in job)

    all_tasks = {}
    machine_to_intervals = {m: [] for m in range(n_machines)}

    for j, job in enumerate(jobs):
        for o, (m, dur) in enumerate(job):
            suffix = f"j{j}_o{o}"
            start_var = model.new_int_var(0, horizon, f"start_{suffix}")
            end_var   = model.new_int_var(0, horizon, f"end_{suffix}")
            interval  = model.new_interval_var(start_var, dur, end_var, f"interval_{suffix}")
            all_tasks[(j, o)] = (start_var, end_var, interval)
            machine_to_intervals[m].append(interval)

    for m in range(n_machines):
        model.add_no_overlap(machine_to_intervals[m])

    for j, job in enumerate(jobs):
        for o in range(len(job) - 1):
            model.add(all_tasks[(j, o + 1)][0] >= all_tasks[(j, o)][1])

    makespan_var = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan_var, [all_tasks[(j, len(job) - 1)][1]
                                          for j, job in enumerate(jobs)])
    model.minimize(makespan_var)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers  = 4
    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = [[] for _ in jobs]
        for j, job in enumerate(jobs):
            for o, (m, dur) in enumerate(job):
                start = solver.value(all_tasks[(j, o)][0])
                end   = solver.value(all_tasks[(j, o)][1])
                schedule[j].append({"machine": m, "start": start, "end": end, "duration": dur})
        makespan = solver.objective_value
        return {"schedule": schedule, "makespan": int(makespan),
                "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"}

    return None


# ─── Diagramme de Gantt ────────────────────────────────────────────────────────
COLORS = plt.cm.tab10.colors + plt.cm.Set2.colors

def plot_gantt(result, n_machines, n_jobs, title="Gantt"):
    schedule = result["schedule"]
    makespan = result["makespan"]

    fig, ax = plt.subplots(figsize=(12, max(3, n_machines * 0.7 + 1)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    for j, ops in enumerate(schedule):
        color = COLORS[j % len(COLORS)]
        for op in ops:
            ax.barh(
                y=op["machine"], left=op["start"],
                width=op["duration"], height=0.6,
                color=color, alpha=0.85, edgecolor="white", linewidth=0.8
            )
            if op["duration"] > makespan * 0.04:
                ax.text(
                    op["start"] + op["duration"] / 2,
                    op["machine"],
                    f"J{j}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white"
                )

    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f"M{m}" for m in range(n_machines)], fontsize=10)
    ax.set_xlabel("Temps", fontsize=10)
    ax.set_title(f"{title} — Makespan : {makespan}", fontsize=12, fontweight="bold")
    ax.axvline(makespan, color="#E24B4A", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"Makespan = {makespan}")
    ax.set_xlim(0, makespan * 1.03)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    patches = [mpatches.Patch(color=COLORS[j % len(COLORS)], label=f"Job {j}")
               for j in range(n_jobs)]
    ax.legend(handles=patches, loc="upper right", ncol=min(5, n_jobs),
              fontsize=8, framealpha=0.7)

    plt.tight_layout()
    return fig


# ─── Graphique comparaison ────────────────────────────────────────────────────
def plot_comparison(results):
    labels   = [r["label"] for r in results if r["makespan"] is not None]
    values   = [r["makespan"] for r in results if r["makespan"] is not None]
    colors   = [r["color"] for r in results if r["makespan"] is not None]
    ref      = results[0]["makespan"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("white")

    # Barres makespan
    bars = ax1.bar(labels, values, color=[c + "88" for c in colors],
                   edgecolor=colors, linewidth=1.5, width=0.5)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ref * 0.01,
                 str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Makespan", fontsize=10)
    ax1.set_title("Comparaison des makespans", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(values) * 1.18)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.2)

    # Barres réduction
    reductions = [0] + [round((1 - v / ref) * 100, 1) for v in values[1:]]
    r_colors   = [colors[0]] + [colors[i] for i in range(1, len(labels))]
    bars2 = ax2.bar(labels, reductions, color=[c + "88" for c in r_colors],
                    edgecolor=r_colors, linewidth=1.5, width=0.5)
    for bar, val in zip(bars2, reductions):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"−{val}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
                     color="#0F6E56")
    ax2.set_ylabel("Réduction vs aléatoire (%)", fontsize=10)
    ax2.set_title("Gain par rapport à l'aléatoire", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.2)
    ax2.axhline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Job Shop Scheduling")
st.sidebar.markdown("---")
st.sidebar.subheader("Paramètres du problème")

n_jobs     = st.sidebar.slider("Nombre de jobs",     min_value=3,  max_value=20, value=10)
n_machines = st.sidebar.slider("Nombre de machines", min_value=2,  max_value=10, value=5)
max_dur    = st.sidebar.slider("Durée max par tâche", min_value=5, max_value=30, value=10)
seed       = st.sidebar.slider("Graine aléatoire",   min_value=1,  max_value=100, value=42)

st.sidebar.markdown("---")
st.sidebar.subheader("Méthodes à comparer")
show_rand = st.sidebar.checkbox("Aléatoire", value=True)
show_spt  = st.sidebar.checkbox("SPT (Shortest Processing Time)", value=True)
show_or   = st.sidebar.checkbox("OR-Tools (CP-SAT)", value=True)

or_time   = st.sidebar.slider("Limite temps OR-Tools (s)", min_value=1, max_value=30, value=5)

run_btn = st.sidebar.button("▶ Lancer l'optimisation", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Projet 2 · Semaine 3 · Dashboard Streamlit")
import math
st.sidebar.caption(f"Espace de solutions : {n_jobs}! ≈ {math.factorial(n_jobs):.2e}")

# ─── Titre principal ──────────────────────────────────────────────────────────
st.title("📊 Job Shop Scheduling — Dashboard Interactif")
st.markdown(
    f"**Problème :** {n_jobs} jobs × {n_machines} machines · "
    f"Durée opération : [1, {max_dur}] · Objectif : minimiser le **makespan**"
)

# ─── Session state ────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.jobs    = None

if run_btn or st.session_state.results is None:
    jobs = generate_jobs(n_jobs, n_machines, max_dur, seed)
    st.session_state.jobs = jobs

    results_raw = {}

    with st.spinner("Calcul en cours…"):
        if show_rand:
            t0 = time.time()
            results_raw["random"] = schedule_random(jobs, n_machines, seed)
            results_raw["random"]["time"] = round(time.time() - t0, 4)

        if show_spt:
            t0 = time.time()
            results_raw["spt"] = schedule_spt(jobs, n_machines)
            results_raw["spt"]["time"] = round(time.time() - t0, 4)

        if show_or:
            t0 = time.time()
            res = schedule_ortools(jobs, n_machines, time_limit=or_time)
            if res:
                results_raw["ortools"] = res
                results_raw["ortools"]["time"] = round(time.time() - t0, 2)

    st.session_state.results = results_raw

results = st.session_state.results
jobs    = st.session_state.jobs

if not results:
    st.warning("Sélectionnez au moins une méthode et cliquez sur Lancer.")
    st.stop()


# ─── Métriques ────────────────────────────────────────────────────────────────
ref = results.get("random", {}).get("makespan", None)
cols = st.columns(3)
meta = [
    ("random", "Aléatoire",    "#E24B4A", "badge-bad"),
    ("spt",    "SPT",          "#EF9F27", "badge-mid"),
    ("ortools","OR-Tools",     "#1D9E75", "badge-good"),
]
for i, (key, label, color, badge) in enumerate(meta):
    with cols[i]:
        if key in results:
            ms  = results[key]["makespan"]
            pct = f"−{round((1 - ms/ref)*100)}%" if ref and key != "random" else "référence"
            st.metric(label=label, value=ms,
                      delta=pct if key != "random" else None,
                      delta_color="inverse" if key != "random" else "off")
        else:
            st.metric(label=label, value="—")


# ─── Onglets ──────────────────────────────────────────────────────────────────
tabs = st.tabs(["📈 Gantt", "📊 Comparaison", "📋 Tableau", "ℹ️ Statistiques"])

# ── Onglet 1 : Gantt ──────────────────────────────────────────────────────────
with tabs[0]:
    gantt_method = st.selectbox(
        "Méthode à afficher",
        [k for k in ["random", "spt", "ortools"] if k in results],
        format_func={"random": "Aléatoire", "spt": "SPT", "ortools": "OR-Tools"}.get
    )
    if gantt_method in results:
        fig = plot_gantt(
            results[gantt_method], n_machines, n_jobs,
            title={"random": "Aléatoire", "spt": "SPT", "ortools": "OR-Tools"}[gantt_method]
        )
        st.pyplot(fig)
        plt.close(fig)

# ── Onglet 2 : Comparaison ────────────────────────────────────────────────────
with tabs[1]:
    compare_data = [
        {"label": "Aléatoire", "makespan": results.get("random",  {}).get("makespan"),  "color": "#E24B4A"},
        {"label": "SPT",       "makespan": results.get("spt",     {}).get("makespan"),  "color": "#EF9F27"},
        {"label": "OR-Tools",  "makespan": results.get("ortools", {}).get("makespan"),  "color": "#1D9E75"},
    ]
    compare_data = [x for x in compare_data if x["makespan"] is not None]
    if len(compare_data) >= 2:
        fig2 = plot_comparison(compare_data)
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("Sélectionnez au moins 2 méthodes pour la comparaison.")

# ── Onglet 3 : Tableau ────────────────────────────────────────────────────────
with tabs[2]:
    rows = []
    for key, label in [("random","Aléatoire"), ("spt","SPT"), ("ortools","OR-Tools")]:
        if key not in results: continue
        ms    = results[key]["makespan"]
        t     = results[key].get("time", "—")
        delta = f"−{round((1 - ms/ref)*100)}%" if ref and key != "random" else "référence"
        rows.append({
            "Méthode":   label,
            "Makespan":  ms,
            "Réduction": delta,
            "Temps (s)": t,
            "Statut":    results[key].get("status", "OK"),
        })
    if rows:
        df = pd.DataFrame(rows).set_index("Méthode")
        st.dataframe(df, use_container_width=True)

    if "ortools" in results:
        st.markdown("---")
        st.subheader("Détail des opérations (OR-Tools)")
        ops_rows = []
        for j, ops in enumerate(results["ortools"]["schedule"]):
            for o, op in enumerate(ops):
                ops_rows.append({
                    "Job": f"J{j}", "Opération": o,
                    "Machine": f"M{op['machine']}",
                    "Début": op["start"], "Fin": op["end"],
                    "Durée": op["duration"]
                })
        st.dataframe(pd.DataFrame(ops_rows), use_container_width=True, height=280)

# ── Onglet 4 : Stats ─────────────────────────────────────────────────────────
with tabs[3]:
    total_work = sum(d for job in jobs for _, d in job)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuration")
        st.write(f"- **Jobs :** {n_jobs}")
        st.write(f"- **Machines :** {n_machines}")
        st.write(f"- **Durée max opération :** {max_dur}")
        st.write(f"- **Graine :** {seed}")
        st.write(f"- **Travail total :** {total_work} unités")
        st.write(f"- **Lower bound théorique :** {max(total_work // n_machines, max(sum(d for _, d in job) for job in jobs))}")

    with col2:
        st.subheader("Performance OR-Tools")
        if "ortools" in results:
            ms_or  = results["ortools"]["makespan"]
            util   = round(total_work / (ms_or * n_machines) * 100, 1)
            lb     = max(total_work // n_machines,
                         max(sum(d for _, d in job) for job in jobs))
            gap    = round((ms_or - lb) / lb * 100, 1)
            st.write(f"- **Makespan :** {ms_or}")
            st.write(f"- **Utilisation machines :** {util}%")
            st.write(f"- **Statut :** {results['ortools'].get('status','—')}")
            st.write(f"- **Écart lower bound :** +{gap}%")
            st.write(f"- **Temps calcul :** {results['ortools'].get('time','—')} s")
