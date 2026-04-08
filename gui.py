"""
gui.py
======
CommonRoad Merge Safety Explorer
---------------------------------
PyQt5 application with four matplotlib panels:

  [Top-left]    Road scene   – animated bird's-eye view of the merge.
  [Top-right]   Safety over time – min TTC and max DRAC per frame.
  [Bottom-left] Sweep: Safety vs Merge Length.
  [Bottom-right] Sweep: Safety vs Curvature.

Left sidebar sliders control all scenario parameters.
"""
from __future__ import annotations

import sys
import threading

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Affine2D
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication, QDoubleSpinBox, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
    QSlider, QSizePolicy, QSpinBox, QSplitter, QStatusBar,
    QVBoxLayout, QWidget,
)

from safety_metrics import TTC_MAX, _get_state
from simulation import SimResult, run_simulation, sweep_curvature, sweep_merge_length

# ── colour palette ─────────────────────────────────────────────────────────
BG        = "#1e1e2e"
PANEL_BG  = "#2a2a3e"
FG        = "#cdd6f4"
ACCENT    = "#89b4fa"
WARN      = "#f38ba8"
SAFE_CLR  = "#a6e3a1"
ROAD_CLR  = "#45475a"
LANE_CLR  = "#6c7086"
CAR_MAIN  = "#89dceb"
CAR_RAMP  = "#f9e2af"
MERGE_CLR = "#cba6f7"
UNSAFE_TTC = 3.0


def _qss_label():
    return f"color: {FG}; background: transparent;"

def _qss_group():
    return (
        f"QGroupBox {{ color: {ACCENT}; font-weight: bold; border: 1px solid {LANE_CLR};"
        f" border-radius: 4px; margin-top: 6px; padding-top: 10px; }}"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; }}"
    )

def _qss_btn(bg=ACCENT):
    return (
        f"QPushButton {{ background: {bg}; color: {BG}; border: none; border-radius: 4px;"
        f" padding: 6px 10px; font-weight: bold; }}"
        f"QPushButton:hover {{ background: {FG}; }}"
    )


# ── signal bridge for cross-thread UI updates ──────────────────────────────

class _Bridge(QObject):
    sim_done   = pyqtSignal(object)
    sweep_done = pyqtSignal(object, object)
    status_msg = pyqtSignal(str)


# ── matplotlib canvas wrapper ──────────────────────────────────────────────

class _Canvas(FigureCanvasQTAgg):
    def __init__(self, fig):
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)


# ── main window ─────────────────────────────────────────────────────────────

class MergeWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CommonRoad Merge Safety Explorer")
        self.resize(1400, 820)
        self._apply_dark_palette()

        self._sim:   SimResult | None = None
        self._anim_step = 0
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_tick)

        self._bridge = _Bridge()
        self._bridge.sim_done.connect(self._on_sim_done)
        self._bridge.sweep_done.connect(self._on_sweeps_done)
        self._bridge.status_msg.connect(self._set_status)

        self._build_ui()
        self._run_sim()

    # ── dark theme ────────────────────────────────────────────────────────

    def _apply_dark_palette(self):
        p = QPalette()
        p.setColor(QPalette.Window,          QColor(BG[1:], 16))
        p.setColor(QPalette.WindowText,      QColor(FG[1:], 16))
        p.setColor(QPalette.Base,            QColor(PANEL_BG[1:], 16))
        p.setColor(QPalette.AlternateBase,   QColor(BG[1:], 16))
        p.setColor(QPalette.Text,            QColor(FG[1:], 16))
        p.setColor(QPalette.Button,          QColor(PANEL_BG[1:], 16))
        p.setColor(QPalette.ButtonText,      QColor(FG[1:], 16))
        self.setStyleSheet(f"background-color: {BG}; color: {FG};")

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        # sidebar
        sidebar = self._build_sidebar()
        sidebar.setFixedWidth(230)
        root_layout.addWidget(sidebar)

        # plots
        plot_widget = self._build_plots()
        root_layout.addWidget(plot_widget, stretch=1)

        # status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet(f"color: {FG}; background: {PANEL_BG};")
        self.status_bar.showMessage("Ready")

    def _build_sidebar(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background-color: {PANEL_BG}; border-radius: 6px;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # ── parameter group ───────────────────────────────────────────────
        grp_road = QGroupBox("Road Parameters")
        grp_road.setStyleSheet(_qss_group())
        g1 = QVBoxLayout(grp_road)

        self._sliders: dict[str, QDoubleSpinBox | QSpinBox] = {}

        self._add_dspin(g1, "merge_length", "Merge length (m)", 40.0, 300.0, 120.0, 5.0)
        self._add_dspin(g1, "curvature",   "Curvature (0–1)",   0.0,   1.0,   0.5, 0.05)
        lay.addWidget(grp_road)

        grp_traffic = QGroupBox("Traffic Parameters")
        grp_traffic.setStyleSheet(_qss_group())
        g2 = QVBoxLayout(grp_traffic)
        self._add_dspin(g2, "main_speed", "Main speed (m/s)", 15.0, 40.0, 28.0, 0.5)
        self._add_dspin(g2, "ramp_speed", "Ramp speed (m/s)", 10.0, 35.0, 22.0, 0.5)
        self._add_ispin(g2, "n_main",     "Main vehicles",     1,    6,    3)
        self._add_ispin(g2, "n_ramp",     "Ramp vehicles",     1,    4,    2)
        lay.addWidget(grp_traffic)

        # ── buttons ───────────────────────────────────────────────────────
        btn_sim = QPushButton("▶  Run Simulation")
        btn_sim.setStyleSheet(_qss_btn(ACCENT))
        btn_sim.clicked.connect(self._run_sim)
        lay.addWidget(btn_sim)

        btn_sweep = QPushButton("⚡  Run Sweeps")
        btn_sweep.setStyleSheet(_qss_btn(MERGE_CLR))
        btn_sweep.clicked.connect(self._run_sweeps)
        lay.addWidget(btn_sweep)

        btn_anim = QPushButton("⏯  Animate Scene")
        btn_anim.setStyleSheet(_qss_btn(SAFE_CLR))
        btn_anim.clicked.connect(self._toggle_animation)
        lay.addWidget(btn_anim)

        lay.addStretch()

        # legend
        leg = QLabel(
            f'<span style="color:{CAR_MAIN}">■</span> Main vehicle &nbsp;&nbsp;'
            f'<span style="color:{CAR_RAMP}">■</span> Ramp vehicle<br>'
            f'<span style="color:{WARN}">■</span> TTC &lt; {UNSAFE_TTC}s (unsafe)<br>'
            f'<span style="color:{ACCENT}">─</span> Min TTC &nbsp;&nbsp;'
            f'<span style="color:{MERGE_CLR}">┄</span> Max DRAC'
        )
        leg.setStyleSheet(f"color: {FG}; font-size: 9px;")
        leg.setWordWrap(True)
        lay.addWidget(leg)

        return w

    def _add_dspin(self, layout, key, label, lo, hi, val, step):
        lbl = QLabel(label)
        lbl.setStyleSheet(_qss_label())
        layout.addWidget(lbl)
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setStyleSheet(
            f"background: {BG}; color: {FG}; border: 1px solid {LANE_CLR}; border-radius: 3px;"
        )
        layout.addWidget(spin)
        self._sliders[key] = spin

    def _add_ispin(self, layout, key, label, lo, hi, val):
        lbl = QLabel(label)
        lbl.setStyleSheet(_qss_label())
        layout.addWidget(lbl)
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setStyleSheet(
            f"background: {BG}; color: {FG}; border: 1px solid {LANE_CLR}; border-radius: 3px;"
        )
        layout.addWidget(spin)
        self._sliders[key] = spin

    def _build_plots(self) -> QWidget:
        w = QWidget()
        lay = QGridLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._fig = Figure(facecolor=BG, tight_layout=True)
        self._ax_scene  = self._fig.add_subplot(2, 2, 1)
        self._ax_ttc    = self._fig.add_subplot(2, 2, 2)
        self._ax_swl    = self._fig.add_subplot(2, 2, 3)
        self._ax_swc    = self._fig.add_subplot(2, 2, 4)

        for ax in (self._ax_scene, self._ax_ttc, self._ax_swl, self._ax_swc):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=FG, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(LANE_CLR)
            ax.title.set_color(FG)
            ax.xaxis.label.set_color(FG)
            ax.yaxis.label.set_color(FG)

        self._canvas = _Canvas(self._fig)
        lay.addWidget(self._canvas)
        return w

    # ── parameter helpers ─────────────────────────────────────────────────

    def _get(self, key: str) -> float:
        v = self._sliders[key].value()
        return float(v)

    # ── simulation ────────────────────────────────────────────────────────

    def _run_sim(self):
        self._stop_animation()
        self._set_status("Running simulation…")
        params = {k: self._get(k) for k in self._sliders}

        def _worker():
            result = run_simulation(
                merge_length=params["merge_length"],
                curvature=params["curvature"],
                n_main=int(params["n_main"]),
                n_ramp=int(params["n_ramp"]),
                main_speed=params["main_speed"],
                ramp_speed=params["ramp_speed"],
            )
            self._bridge.sim_done.emit(result)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_sim_done(self, result: SimResult):
        self._sim = result
        self._anim_step = 0
        self._draw_scene(0)
        self._draw_ttc()
        self._set_status(
            f"Done  |  merge_length={result.merge_length:.0f} m  "
            f"curvature={result.curvature:.2f}  |  "
            f"min TTC = {result.min_ttc_series.min():.2f} s  "
            f"unsafe frames = {(result.min_ttc_series < UNSAFE_TTC).sum()}"
        )

    # ── sweeps ────────────────────────────────────────────────────────────

    def _run_sweeps(self):
        self._set_status("Running parameter sweeps (this may take ~10 s)…")
        params = {k: self._get(k) for k in self._sliders}

        def _worker():
            sw_len = sweep_merge_length(
                curvature=params["curvature"],
                n_main=int(params["n_main"]),
                n_ramp=int(params["n_ramp"]),
                main_speed=params["main_speed"],
                ramp_speed=params["ramp_speed"],
            )
            sw_cur = sweep_curvature(
                merge_length=params["merge_length"],
                n_main=int(params["n_main"]),
                n_ramp=int(params["n_ramp"]),
                main_speed=params["main_speed"],
                ramp_speed=params["ramp_speed"],
            )
            self._bridge.sweep_done.emit(sw_len, sw_cur)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_sweeps_done(self, sw_len: dict, sw_cur: dict):
        self._draw_sweep(self._ax_swl, sw_len, "Merge Length (m)")
        self._draw_sweep(self._ax_swc, sw_cur, "Curvature")
        self._canvas.draw_idle()
        self._set_status("Sweeps complete.")

    # ── drawing ───────────────────────────────────────────────────────────

    def _draw_scene(self, t: int):
        if self._sim is None:
            return
        sim = self._sim
        ax  = self._ax_scene
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        ax.set_aspect("equal")

        ml      = sim.merge_length
        total_x = ml + 300.0
        y_ramp  = 4.0 * 2.5   # LANE_WIDTH * 2.5

        # Road surface
        road = mpatches.FancyBboxPatch(
            (-15, -3.5), total_x + 30, 7,
            boxstyle="round,pad=0.5",
            facecolor=ROAD_CLR, edgecolor="none", zorder=0,
        )
        ax.add_patch(road)

        # Bezier merge lane centreline
        from merge_scenario import bezier_merge_centerline
        cl = bezier_merge_centerline(ml, y_ramp, 0.0, sim.curvature)
        ax.plot(cl[:, 0], cl[:, 1], "--", color=MERGE_CLR, lw=1.5,
                alpha=0.8, zorder=1)

        # Lane markings
        ax.axhline(0,  color=LANE_CLR, lw=1.0, ls="--", zorder=1)
        ax.axhline(-2, color=LANE_CLR, lw=0.5, ls="-",  zorder=1)
        ax.axhline(+2, color=LANE_CLR, lw=0.5, ls="-",  zorder=1)

        # Merge point marker
        ax.axvline(ml, color=MERGE_CLR, lw=0.8, ls=":", alpha=0.5, zorder=1)
        ax.text(ml, 12, "merge\npoint", color=MERGE_CLR,
                fontsize=6, ha="center", va="top")

        # Vehicles
        frame = sim.frames[min(t, len(sim.frames) - 1)]
        for obs in sim.scenario.dynamic_obstacles:
            st = _get_state(obs, t)
            if st is None:
                continue
            pos = np.array(st.position)
            ori = float(st.orientation)
            color = CAR_RAMP if obs.obstacle_id >= 300 else CAR_MAIN

            from matplotlib.patches import FancyBboxPatch
            patch = FancyBboxPatch(
                (-4.7 / 2, -2.0 / 2), 4.7, 2.0,
                boxstyle="round,pad=0.3",
                facecolor=color, edgecolor="white", linewidth=0.8, alpha=0.9,
                zorder=3,
            )
            tfm = Affine2D().rotate(ori).translate(pos[0], pos[1]) + ax.transData
            patch.set_transform(tfm)
            ax.add_patch(patch)
            ax.text(pos[0], pos[1] + 1.7, str(obs.obstacle_id),
                    color="white", fontsize=6, ha="center", zorder=5,
                    transform=ax.transData)

        # Safety annotation
        min_ttc = frame.min_ttc
        clr = SAFE_CLR if min_ttc >= UNSAFE_TTC else WARN
        ax.text(0.99, 0.97, f"TTC {min_ttc:.1f}s",
                transform=ax.transAxes, color=clr,
                fontsize=9, ha="right", va="top", fontweight="bold")

        ax.set_xlim(-15, total_x + 15)
        ax.set_ylim(-6, 16)
        ax.set_title(f"Scene  t = {t * sim.scenario.dt:.1f} s", color=FG, fontsize=9)
        ax.set_xlabel("x (m)", color=FG, fontsize=7)
        ax.set_ylabel("y (m)", color=FG, fontsize=7)
        ax.tick_params(colors=FG, labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor(LANE_CLR)
        self._canvas.draw_idle()

    def _draw_ttc(self):
        if self._sim is None:
            return
        sim = self._sim
        ax  = self._ax_ttc
        ax.clear()
        ax.set_facecolor(PANEL_BG)

        times   = sim.times
        min_ttc = sim.min_ttc_series
        max_dr  = sim.max_drac_series

        ax.plot(times, min_ttc, color=ACCENT, lw=1.5, label="min TTC (s)")
        ax.fill_between(times, 0, min_ttc,
                        where=(min_ttc < UNSAFE_TTC),
                        color=WARN, alpha=0.35, label=f"TTC < {UNSAFE_TTC}s")
        ax.axhline(UNSAFE_TTC, color=WARN, lw=1, ls="--", alpha=0.8)
        ax.set_ylabel("Min TTC (s)", color=FG, fontsize=7)
        ax.set_ylim(0, TTC_MAX + 0.5)
        ax.tick_params(colors=FG, labelsize=6)

        ax2 = ax.twinx()
        ax2.plot(times, max_dr, color=MERGE_CLR, lw=1.0, ls=":", alpha=0.8,
                 label="max DRAC (m/s²)")
        ax2.axhline(3.4, color=MERGE_CLR, lw=0.8, ls="--", alpha=0.5)
        ax2.set_ylabel("Max DRAC (m/s²)", color=MERGE_CLR, fontsize=7)
        ax2.tick_params(axis="y", colors=MERGE_CLR, labelsize=6)
        ax2.set_facecolor(PANEL_BG)
        for sp in ax2.spines.values():
            sp.set_edgecolor(LANE_CLR)

        ax.set_xlabel("Time (s)", color=FG, fontsize=7)
        ax.set_title(
            f"Safety Metrics  [L={sim.merge_length:.0f}m  κ={sim.curvature:.2f}]",
            color=FG, fontsize=9,
        )
        l1, n1 = ax.get_legend_handles_labels()
        l2, n2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, n1 + n2, facecolor=PANEL_BG, edgecolor=LANE_CLR,
                  labelcolor=FG, fontsize=6, loc="upper right")
        for sp in ax.spines.values():
            sp.set_edgecolor(LANE_CLR)
        self._canvas.draw_idle()

    def _draw_sweep(self, ax, data: dict, xlabel: str):
        ax.clear()
        ax.set_facecolor(PANEL_BG)

        xs  = data["params"]
        ttc = data["mean_min_ttc"]
        pct = data["pct_unsafe"]
        dr  = data["max_drac"]

        ax.plot(xs, ttc, "-o", color=ACCENT,  lw=1.5, ms=5, label="Mean min TTC (s)")
        ax.axhline(UNSAFE_TTC, color=ACCENT, lw=0.8, ls="--", alpha=0.6)
        ax.set_ylabel("Mean min TTC (s)", color=FG, fontsize=7)
        ax.set_ylim(0, TTC_MAX + 0.5)
        ax.tick_params(colors=FG, labelsize=6)

        ax2 = ax.twinx()
        ax2.plot(xs, pct, "-s", color=WARN, lw=1.5, ms=5,
                 label="% unsafe frames (TTC<3s)")
        ax2.set_ylabel("% unsafe frames", color=WARN, fontsize=7)
        ax2.tick_params(axis="y", colors=WARN, labelsize=6)
        ax2.set_ylim(0, 105)
        ax2.set_facecolor(PANEL_BG)
        for sp in ax2.spines.values():
            sp.set_edgecolor(LANE_CLR)

        ax.set_xlabel(xlabel, color=FG, fontsize=7)
        ax.set_title(f"Safety vs {xlabel}", color=FG, fontsize=9)
        l1, n1 = ax.get_legend_handles_labels()
        l2, n2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, n1 + n2, facecolor=PANEL_BG, edgecolor=LANE_CLR,
                  labelcolor=FG, fontsize=6, loc="best")
        for sp in ax.spines.values():
            sp.set_edgecolor(LANE_CLR)

    # ── animation ─────────────────────────────────────────────────────────

    def _toggle_animation(self):
        if self._anim_timer.isActive():
            self._stop_animation()
        else:
            self._start_animation()

    def _start_animation(self):
        if self._sim is None:
            return
        self._anim_step = 0
        self._anim_timer.start(80)  # ~12 fps

    def _stop_animation(self):
        self._anim_timer.stop()

    def _anim_tick(self):
        if self._sim is None:
            self._anim_timer.stop()
            return
        self._draw_scene(self._anim_step)
        self._anim_step = (self._anim_step + 1) % len(self._sim.frames)

    # ── status ────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self.status_bar.showMessage(msg)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MergeWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
