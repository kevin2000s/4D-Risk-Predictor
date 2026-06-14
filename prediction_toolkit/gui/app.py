#!/usr/bin/env python3
"""
SVD(128) + ExtraTrees 4D Transmission Risk Prediction - GUI Application
A BEAST-style desktop application for predicting transmission risk.

Usage:
    python app.py

Requirements:
    pip install ttkbootstrap pandas numpy matplotlib scikit-learn scipy
"""
import sys
import os
import threading
import traceback

# Add parent directories to path
# Support both development and PyInstaller frozen environments
if getattr(sys, 'frozen', False):
    # PyInstaller: sys.executable is the .exe, resources are in same dir
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'prediction_toolkit'))

try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *
    from ttkbootstrap.dialogs import Messagebox
    from ttkbootstrap.scrolled import ScrolledFrame, ScrolledText
except ImportError:
    print("Error: ttkbootstrap not installed.")
    print("Please run: pip install ttkbootstrap")
    sys.exit(1)

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('TkAgg')


class TransmissionRiskApp:
    """Main GUI Application"""

    def __init__(self, root):
        self.root = root
        self.root.title("4D Transmission Risk Predictor - SVD(128) + ExtraTrees")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Data storage
        self.snp_data = None
        self.env_data = None
        self.predictions = None
        self.snp_file_path = None
        self.env_file_path = None
        self.model_loaded = False

        # Style
        self.style = ttk.Style("darkly")
        self.colors = self.style.colors

        # Build UI
        self._build_menu()
        self._build_layout()
        self._build_sidebar()
        self._build_pages()

        # Show home page
        self.show_page("home")

        # Status bar
        self._build_statusbar()

    # ==================== UI BUILDING ====================

    def _build_menu(self):
        """Menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load SNP Data...", command=self._load_snp_dialog)
        file_menu.add_command(label="Load Environment Data...", command=self._load_env_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Export Predictions...", command=self._export_predictions)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Run Prediction", command=self._run_prediction)
        tools_menu.add_command(label="Clear All Data", command=self._clear_data)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._show_docs)
        help_menu.add_command(label="About", command=self._show_about)

    def _build_layout(self):
        """Main layout with sidebar and content area"""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=BOTH, expand=True)

        # Sidebar
        self.sidebar = ttk.Frame(self.main_frame, width=200, bootstyle="dark")
        self.sidebar.pack(side=LEFT, fill=Y)
        self.sidebar.pack_propagate(False)

        # Content area
        self.content = ttk.Frame(self.main_frame)
        self.content.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)

    def _build_sidebar(self):
        """Navigation sidebar"""
        # App title
        title_frame = ttk.Frame(self.sidebar, bootstyle="dark")
        title_frame.pack(fill=X, pady=(20, 10), padx=10)

        ttk.Label(title_frame, text="4D Risk", font=("Helvetica", 18, "bold"),
                  bootstyle="inverse-dark").pack()
        ttk.Label(title_frame, text="Predictor", font=("Helvetica", 12),
                  bootstyle="inverse-dark").pack()

        ttk.Separator(self.sidebar, bootstyle="light").pack(fill=X, padx=15, pady=10)

        # Navigation buttons
        self.nav_buttons = {}
        nav_items = [
            ("home", "Home", "🏠"),
            ("import", "Import Data", "📁"),
            ("preview", "Data Preview", "📊"),
            ("predict", "Run Prediction", "⚙️"),
            ("results", "Results", "📊"),
            ("about", "About", "ℹ️"),
        ]

        for page_id, label, icon in nav_items:
            btn = ttk.Button(
                self.sidebar,
                text=f"  {icon}  {label}",
                bootstyle="dark",
                command=lambda p=page_id: self.show_page(p),
                width=20
            )
            btn.pack(fill=X, padx=10, pady=3)
            self.nav_buttons[page_id] = btn

        ttk.Separator(self.sidebar, bootstyle="light").pack(fill=X, padx=15, pady=15)

        # Quick actions
        ttk.Label(self.sidebar, text="Quick Actions", font=("Helvetica", 10, "bold"),
                  bootstyle="inverse-dark").pack(padx=15, pady=(0, 5))

        ttk.Button(self.sidebar, text="  📂  Load SNP",
                   bootstyle="dark-outline",
                   command=self._load_snp_dialog).pack(fill=X, padx=10, pady=2)
        ttk.Button(self.sidebar, text="  🌡️  Load Env",
                   bootstyle="dark-outline",
                   command=self._load_env_dialog).pack(fill=X, padx=10, pady=2)
        ttk.Button(self.sidebar, text="  🚀  Run",
                   bootstyle="success-outline",
                   command=self._run_prediction).pack(fill=X, padx=10, pady=5)

    def _build_pages(self):
        """Build all pages"""
        self.pages = {}

        # Create a frame for each page
        for page_id in ["home", "import", "preview", "predict", "results", "about"]:
            page = ttk.Frame(self.content)
            self.pages[page_id] = page

        self._build_home_page()
        self._build_import_page()
        self._build_preview_page()
        self._build_predict_page()
        self._build_results_page()
        self._build_about_page()

    def _build_statusbar(self):
        """Status bar at bottom"""
        self.statusbar = ttk.Frame(self.root, height=25, bootstyle="secondary")
        self.statusbar.pack(side=BOTTOM, fill=X)

        self.status_label = ttk.Label(self.statusbar, text="Ready", bootstyle="inverse-secondary")
        self.status_label.pack(side=LEFT, padx=10)

        self.progress = ttk.Progressbar(self.statusbar, mode="determinate", length=200, bootstyle="success")
        self.progress.pack(side=RIGHT, padx=10, pady=3)

    # ==================== HOME PAGE ====================

    def _build_home_page(self):
        page = self.pages["home"]

        # Welcome card
        card = ttk.Frame(page, bootstyle="default")
        card.pack(fill=BOTH, expand=True, padx=20, pady=20)

        ttk.Label(card, text="Welcome to 4D Transmission Risk Predictor",
                  font=("Helvetica", 24, "bold")).pack(pady=(30, 10))

        ttk.Label(card, text="SVD(128) + ExtraTrees Full SNP Model",
                  font=("Helvetica", 14)).pack(pady=(0, 20))

        # Model info
        info_frame = ttk.Labelframe(card, text="Model Information", padding=15)
        info_frame.pack(fill=X, padx=50, pady=10)

        info_text = """Training Samples: 689  |  SNPs: 151,913  |  SVD Components: 128
SVD Explained Variance: 99.48%

Prediction Dimensions:
  • Network Hub          (CV R² = 0.884 ± 0.017, Test R² = 0.901)
  • Clone Advantage      (CV R² = 0.797 ° 0.047, Test R² = 0.869)
  • Persistence          (CV R² = 0.821 ° 0.055, Test R² = 0.900)
  • Spatial Connectivity (CV R² = 0.942 ° 0.021, Test R² = 0.927)

Feature Contribution:
  • Genomic (SVD Latent): 77.0%
  • Environmental: 23.0%"""

        ttk.Label(info_frame, text=info_text, font=("Consolas", 11),
                  justify=LEFT).pack(anchor=W)

        # Quick start buttons
        btn_frame = ttk.Frame(card)
        btn_frame.pack(pady=30)

        ttk.Button(btn_frame, text="📁  Import Data",
                   bootstyle="primary",
                   command=lambda: self.show_page("import"),
                   width=18).pack(side=LEFT, padx=10)

        ttk.Button(btn_frame, text="🚀  Run Prediction",
                   bootstyle="success",
                   command=lambda: self.show_page("predict"),
                   width=18).pack(side=LEFT, padx=10)

        ttk.Button(btn_frame, text="📚  View Results",
                   bootstyle="info",
                   command=lambda: self.show_page("results"),
                   width=18).pack(side=LEFT, padx=10)

    # ==================== IMPORT PAGE ====================

    def _build_import_page(self):
        page = self.pages["import"]

        ttk.Label(page, text="Import Data", font=("Helvetica", 20, "bold")).pack(anchor=W, pady=(0, 15))

        # SNP Data Card
        snp_card = ttk.Labelframe(page, text="SNP Data (Genomic)", padding=15, bootstyle="primary")
        snp_card.pack(fill=X, pady=10)

        self.snp_file_label = ttk.Label(snp_card, text="No file selected", bootstyle="secondary")
        self.snp_file_label.pack(anchor=W, pady=5)

        ttk.Button(snp_card, text="Browse...", command=self._load_snp_dialog,
                   bootstyle="primary-outline").pack(side=LEFT, padx=(0, 10))
        ttk.Button(snp_card, text="Clear", command=self._clear_snp,
                   bootstyle="danger-outline").pack(side=LEFT)

        # Format info
        format_text = """Supported format (like snp_sample_count.csv):
  Columns: CHROM, POS, TYPE, REF, ALT, sample_count, sample_ids
  The "sample_ids" column contains space-separated sample IDs"""
        ttk.Label(snp_card, text=format_text, font=("Consolas", 9),
                  bootstyle="secondary").pack(anchor=W, pady=(10, 0))

        # Environment Data Card
        env_card = ttk.Labelframe(page, text="Environment Data", padding=15, bootstyle="info")
        env_card.pack(fill=X, pady=10)

        self.env_file_label = ttk.Label(env_card, text="No file selected", bootstyle="secondary")
        self.env_file_label.pack(anchor=W, pady=5)

        ttk.Button(env_card, text="Browse...", command=self._load_env_dialog,
                   bootstyle="info-outline").pack(side=LEFT, padx=(0, 10))
        ttk.Button(env_card, text="Clear", command=self._clear_env,
                   bootstyle="danger-outline").pack(side=LEFT)

        env_text = """Required columns: sample_id, PM2.5, PM10, SO2, NO2, CO, O3, AQI
Missing values will be filled with column means."""
        ttk.Label(env_card, text=env_text, font=("Consolas", 9),
                  bootstyle="secondary").pack(anchor=W, pady=(10, 0))

        # Options
        opts_card = ttk.Labelframe(page, text="Options", padding=15)
        opts_card.pack(fill=X, pady=10)

        self.env_only_var = ttk.BooleanVar(value=False)
        ttk.Checkbutton(opts_card, text="Environment-only prediction (Spatial Connectivity only)",
                        variable=self.env_only_var).pack(anchor=W, pady=(10, 0))

        # Next button
        ttk.Button(page, text="Next: Preview Data →",
                   bootstyle="success",
                   command=lambda: self.show_page("preview")).pack(anchor=E, pady=20)

    # ==================== PREVIEW PAGE ====================

    def _build_preview_page(self):
        page = self.pages["preview"]

        ttk.Label(page, text="Data Preview", font=("Helvetica", 20, "bold")).pack(anchor=W, pady=(0, 15))

        # Notebook for tabs
        self.preview_notebook = ttk.Notebook(page)
        self.preview_notebook.pack(fill=BOTH, expand=True)

        # SNP tab
        self.snp_preview_frame = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.snp_preview_frame, text="SNP Data")

        self.snp_preview_text = ScrolledText(self.snp_preview_frame, height=20,
                                              font=("Consolas", 10))
        self.snp_preview_text.pack(fill=BOTH, expand=True)
        self.snp_preview_text.insert(END, "No SNP data loaded. Go to Import Data page.")
        # Access internal text widget for state control
        if hasattr(self.snp_preview_text, 'text'):
            self.snp_preview_text.text.config(state=DISABLED)
        else:
            self.snp_preview_text.config(state=DISABLED)

        # Env tab
        self.env_preview_frame = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.env_preview_frame, text="Environment Data")

        # Env stats frame
        self.env_stats_frame = ttk.Frame(self.env_preview_frame)
        self.env_stats_frame.pack(fill=X, pady=5)

        # Env table
        self.env_tree_frame = ttk.Frame(self.env_preview_frame)
        self.env_tree_frame.pack(fill=BOTH, expand=True)

        self.env_tree = ttk.Treeview(self.env_tree_frame, show="headings", height=15)
        self.env_tree.pack(side=LEFT, fill=BOTH, expand=True)

        env_scroll = ttk.Scrollbar(self.env_tree_frame, orient="vertical", command=self.env_tree.yview)
        env_scroll.pack(side=RIGHT, fill=Y)
        self.env_tree.configure(yscrollcommand=env_scroll.set)

        # Stats tab
        self.stats_frame = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.stats_frame, text="Statistics")

        self.stats_text = ScrolledText(self.stats_frame, height=20, font=("Consolas", 10))
        self.stats_text.pack(fill=BOTH, expand=True)
        self.stats_text.insert(END, "Load data to see statistics.")
        if hasattr(self.stats_text, 'text'):
            self.stats_text.text.config(state=DISABLED)
        else:
            self.stats_text.config(state=DISABLED)

    # ==================== PREDICT PAGE ====================

    def _build_predict_page(self):
        page = self.pages["predict"]

        ttk.Label(page, text="Run Prediction", font=("Helvetica", 20, "bold")).pack(anchor=W, pady=(0, 15))

        # Data status
        self.status_card = ttk.Labelframe(page, text="Data Status", padding=15)
        self.status_card.pack(fill=X, pady=10)

        self.snp_status_label = ttk.Label(self.status_card, text="SNP Data: Not loaded",
                                          bootstyle="danger")
        self.snp_status_label.pack(anchor=W, pady=2)

        self.env_status_label = ttk.Label(self.status_card, text="Environment Data: Not loaded",
                                          bootstyle="danger")
        self.env_status_label.pack(anchor=W, pady=2)

        # Run button
        self.run_btn = ttk.Button(page, text="▶  Start Prediction",
                                  bootstyle="success",
                                  command=self._run_prediction,
                                  width=25)
        self.run_btn.pack(pady=20)

        # Progress
        self.predict_progress = ttk.Progressbar(page, mode="determinate", length=500,
                                                bootstyle="success")
        self.predict_progress.pack(pady=10)
        self.predict_progress['value'] = 0

        # Log
        log_frame = ttk.Labelframe(page, text="Log", padding=5)
        log_frame.pack(fill=BOTH, expand=True, pady=10)

        self.log_text = ScrolledText(log_frame, height=15, font=("Consolas", 9))
        self.log_text.pack(fill=BOTH, expand=True)
        self.log_text.insert(END, "Ready to run prediction.\n")
        self._set_text_state(self.log_text, DISABLED)

    # ==================== RESULTS PAGE ====================

    def _build_results_page(self):
        page = self.pages["results"]

        ttk.Label(page, text="Prediction Results", font=("Helvetica", 20, "bold")).pack(anchor=W, pady=(0, 15))

        # Toolbar
        toolbar = ttk.Frame(page)
        toolbar.pack(fill=X, pady=5)

        ttk.Button(toolbar, text="📋 Table", command=self._show_result_table,
                   bootstyle="primary-outline").pack(side=LEFT, padx=2)
        ttk.Button(toolbar, text="📈 Scatter", command=self._show_scatter_plot,
                   bootstyle="info-outline").pack(side=LEFT, padx=2)
        ttk.Button(toolbar, text="🎻 Violin", command=self._show_violin_plot,
                   bootstyle="info-outline").pack(side=LEFT, padx=2)
        ttk.Button(toolbar, text="🔥 Heatmap", command=self._show_heatmap_plot,
                   bootstyle="info-outline").pack(side=LEFT, padx=2)
        ttk.Button(toolbar, text="🫧 Bubble", command=self._show_bubble_plot,
                   bootstyle="info-outline").pack(side=LEFT, padx=2)
        ttk.Button(toolbar, text="📊 Radar", command=self._show_radar_plot,
                   bootstyle="info-outline").pack(side=LEFT, padx=2)
        ttk.Button(toolbar, text="📊 Box+Dot", command=self._show_boxdot_plot,
                   bootstyle="info-outline").pack(side=LEFT, padx=2)
        ttk.Button(toolbar, text="📤 Export", command=self._export_predictions,
                   bootstyle="success-outline").pack(side=LEFT, padx=10)

        # Results area
        self.result_frame = ttk.Frame(page)
        self.result_frame.pack(fill=BOTH, expand=True, pady=10)

        # Placeholder
        self.result_placeholder = ttk.Label(self.result_frame,
                                            text="No predictions yet.\nRun prediction to see results.",
                                            font=("Helvetica", 14), bootstyle="secondary")
        self.result_placeholder.pack(expand=True)

    # ==================== ABOUT PAGE ====================

    def _build_about_page(self):
        page = self.pages["about"]

        card = ttk.Frame(page)
        card.pack(fill=BOTH, expand=True, padx=50, pady=50)

        ttk.Label(card, text="4D Transmission Risk Predictor",
                  font=("Helvetica", 22, "bold")).pack(pady=(20, 5))

        ttk.Label(card, text="Version 1.0.0",
                  font=("Helvetica", 12), bootstyle="secondary").pack()

        ttk.Separator(card).pack(fill=X, padx=50, pady=20)

        about_text = """This software implements a SVD(128) + ExtraTrees regression model
for predicting 4-dimensional transmission risk of bacterial pathogens.

Model Architecture:
  • Input: 151,913 SNP variants + 7 environmental features
  • Dimensionality Reduction: TruncatedSVD (128 components, 99.48% variance)
  • Regressor: ExtraTrees (300 estimators, max_depth=20)
  • Output: Network Hub, Clone Advantage, Persistence, Spatial Connectivity

Development Team:
  Peking University People's Hospital Hui Lab

Technologies:
  Python, scikit-learn, pandas, numpy, matplotlib, ttkbootstrap

For questions or support, please refer to the documentation."""

        ttk.Label(card, text=about_text, font=("Helvetica", 11),
                  justify=CENTER).pack(pady=10)

        ttk.Separator(card).pack(fill=X, padx=50, pady=20)

        ttk.Label(card, text="© 2025 All Rights Reserved",
                  font=("Helvetica", 10), bootstyle="secondary").pack()

    # ==================== NAVIGATION ====================

    def show_page(self, page_id):
        """Switch to a page"""
        for pid, page in self.pages.items():
            page.pack_forget()
            if pid in self.nav_buttons:
                self.nav_buttons[pid].config(bootstyle="dark")

        self.pages[page_id].pack(fill=BOTH, expand=True)
        if page_id in self.nav_buttons:
            self.nav_buttons[page_id].config(bootstyle="primary")

    # ==================== DATA LOADING ====================

    def _load_snp_dialog(self):
        """Open file dialog for SNP data"""
        path = filedialog.askopenfilename(
            title="Select SNP Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self._load_snp_file(path)

    def _load_snp_file(self, path):
        """Load SNP data from file"""
        try:
            self.snp_file_path = path
            self.snp_data = pd.read_csv(path)
            self.snp_file_label.config(text=f"Loaded: {os.path.basename(path)} ({len(self.snp_data)} rows)",
                                       bootstyle="success")
            self.snp_status_label.config(text=f"SNP Data: Loaded ({len(self.snp_data)} rows, {len(self.snp_data.columns)} cols)",
                                         bootstyle="success")
            self._update_snp_preview()
            self._update_stats()
            self._set_status(f"SNP data loaded: {os.path.basename(path)}")
        except Exception as e:
            Messagebox.show_error(f"Failed to load SNP data:\n{str(e)}", "Error")
            self._set_status("SNP load failed")

    def _load_env_dialog(self):
        """Open file dialog for environment data"""
        path = filedialog.askopenfilename(
            title="Select Environment Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self._load_env_file(path)

    def _load_env_file(self, path):
        """Load environment data from file"""
        try:
            self.env_file_path = path
            try:
                self.env_data = pd.read_csv(path, encoding='utf-8')
            except UnicodeDecodeError:
                self.env_data = pd.read_csv(path, encoding='gbk')

            self.env_file_label.config(text=f"Loaded: {os.path.basename(path)} ({len(self.env_data)} rows)",
                                       bootstyle="success")
            self.env_status_label.config(text=f"Environment Data: Loaded ({len(self.env_data)} rows)",
                                         bootstyle="success")
            self._update_env_preview()
            self._update_stats()
            self._set_status(f"Environment data loaded: {os.path.basename(path)}")
        except Exception as e:
            Messagebox.show_error(f"Failed to load environment data:\n{str(e)}", "Error")
            self._set_status("Environment load failed")

    def _clear_snp(self):
        self.snp_data = None
        self.snp_file_path = None
        self.snp_file_label.config(text="No file selected", bootstyle="secondary")
        self.snp_status_label.config(text="SNP Data: Not loaded", bootstyle="danger")
        self._update_snp_preview()

    def _clear_env(self):
        self.env_data = None
        self.env_file_path = None
        self.env_file_label.config(text="No file selected", bootstyle="secondary")
        self.env_status_label.config(text="Environment Data: Not loaded", bootstyle="danger")
        self._update_env_preview()

    def _clear_data(self):
        self._clear_snp()
        self._clear_env()
        self.predictions = None
        self._update_stats()
        self._set_status("All data cleared")

    # ==================== PREVIEW UPDATES ====================

    def _update_snp_preview(self):
        """Update SNP preview text"""
        self._set_text_state(self.snp_preview_text, NORMAL)
        self.snp_preview_text.delete(1.0, END)

        if self.snp_data is not None:
            self.snp_preview_text.insert(END, f"File: {self.snp_file_path}\n")
            self.snp_preview_text.insert(END, f"Shape: {self.snp_data.shape}\n")
            self.snp_preview_text.insert(END, f"Columns: {list(self.snp_data.columns)}\n\n")
            self.snp_preview_text.insert(END, self.snp_data.head(20).to_string())
        else:
            self.snp_preview_text.insert(END, "No SNP data loaded.\nGo to Import Data page to load SNP data.")

        self._set_text_state(self.snp_preview_text, DISABLED)

    def _update_env_preview(self):
        """Update environment preview table"""
        for widget in self.env_tree_frame.winfo_children():
            widget.destroy()

        self.env_tree = ttk.Treeview(self.env_tree_frame, show="headings", height=15)
        self.env_tree.pack(side=LEFT, fill=BOTH, expand=True)

        scroll = ttk.Scrollbar(self.env_tree_frame, orient="vertical", command=self.env_tree.yview)
        scroll.pack(side=RIGHT, fill=Y)
        self.env_tree.configure(yscrollcommand=scroll.set)

        if self.env_data is not None:
            cols = list(self.env_data.columns)
            self.env_tree['columns'] = cols
            for col in cols:
                self.env_tree.heading(col, text=col)
                self.env_tree.column(col, width=80)

            for _, row in self.env_data.head(100).iterrows():
                values = [str(v) for v in row.values]
                self.env_tree.insert("", END, values=values)

            # Stats
            for widget in self.env_stats_frame.winfo_children():
                widget.destroy()

            stats = self.env_data.describe()
            ttk.Label(self.env_stats_frame, text=stats.to_string(),
                      font=("Consolas", 9)).pack(anchor=W)

    def _update_stats(self):
        """Update statistics panel"""
        self._set_text_state(self.stats_text, NORMAL)
        self.stats_text.delete(1.0, END)

        if self.snp_data is not None:
            self.stats_text.insert(END, "=== SNP Data ===\n")
            self.stats_text.insert(END, f"Shape: {self.snp_data.shape}\n")
            self.stats_text.insert(END, f"Columns: {list(self.snp_data.columns)}\n")
            self.stats_text.insert(END, f"Memory: {self.snp_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n\n")

        if self.env_data is not None:
            self.stats_text.insert(END, "=== Environment Data ===\n")
            self.stats_text.insert(END, f"Shape: {self.env_data.shape}\n")
            self.stats_text.insert(END, f"Columns: {list(self.env_data.columns)}\n")
            self.stats_text.insert(END, f"Missing values:\n")
            missing = self.env_data.isna().sum()
            for col, count in missing.items():
                if count > 0:
                    self.stats_text.insert(END, f"  {col}: {count}\n")
            self.stats_text.insert(END, "\n")

        if self.snp_data is None and self.env_data is None:
            self.stats_text.insert(END, "Load data to see statistics.")

        self._set_text_state(self.stats_text, DISABLED)

    # ==================== PREDICTION ====================

    def _log(self, message):
        """Add message to log"""
        self._set_text_state(self.log_text, NORMAL)
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)
        self._set_text_state(self.log_text, DISABLED)
        self.root.update_idletasks()

    def _run_prediction(self):
        """Run prediction in background thread"""
        if self.env_data is None:
            Messagebox.show_warning("Please load environment data first!", "Missing Data")
            return

        if self.snp_data is None and not self.env_only_var.get():
            result = Messagebox.yesno(
                "No SNP data loaded.\n\nRun environment-only prediction?\n"
                "(Only Spatial Connectivity will be reliable)",
                "Confirm"
            )
            if result == "Yes":
                self.env_only_var.set(True)
            else:
                return

        self.run_btn.config(state=DISABLED)
        self.predict_progress['value'] = 0
        self._set_text_state(self.log_text, NORMAL)
        self.log_text.delete(1.0, END)
        self._set_text_state(self.log_text, DISABLED)

        thread = threading.Thread(target=self._prediction_worker)
        thread.daemon = True
        thread.start()

    def _prediction_worker(self):
        """Background prediction worker"""
        try:
            self._log("=" * 50)
            self._log("Starting 4D Transmission Risk Prediction")
            self._log("=" * 50)

            # Import model components
            self._log("[1/5] Loading model...")
            from prediction_toolkit.model import TransmissionRiskPredictor
            from prediction_toolkit.data_loader import SNPDataLoader, EnvDataLoader, align_samples

            predictor = TransmissionRiskPredictor()
            self.predict_progress['value'] = 15

            # Load environment data
            self._log("[2/5] Processing environment data...")
            env_loader = EnvDataLoader()

            # Find sample column
            sample_col = 'sample_id'
            if sample_col not in self.env_data.columns:
                for col in self.env_data.columns:
                    if col.lower() in ['user_genome', 'sample id', 'sample_id']:
                        sample_col = col
                        break

            X_env = self.env_data[env_loader.REQUIRED_COLS].fillna(self.env_data[env_loader.REQUIRED_COLS].mean()).values.astype(np.float32)
            env_samples = self.env_data[sample_col].astype(str).tolist()
            self.predict_progress['value'] = 30

            # Environment-only prediction
            if self.env_only_var.get() or self.snp_data is None:
                self._log("[3/5] Environment-only mode (SNP input skipped)")
                self._log("      Warning: Only Spatial Connectivity is reliable")
                results = predictor.predict_env_only(X_env, env_samples)
                self.predict_progress['value'] = 80

            else:
                # Load SNP data
                self._log("[3/5] Processing SNP data...")
                snp_loader = SNPDataLoader(predictor.snp_ids)

                # Parse long format SNP data (like snp_sample_count.csv)
                self._log("      Parsing long format SNP data...")
                # Long format - need to reconstruct

                # Check for required columns
                snp_df = self.snp_data.copy()
                if 'CHROM' not in snp_df.columns and 'sample_ids' not in snp_df.columns and 'affected_samples' not in snp_df.columns and '涉及样本' not in snp_df.columns:
                    # Try to detect columns
                    self._log("      Warning: Detecting column names...")

                # Create snp_id
                chrom_col = None
                pos_col = None
                samples_col = None

                for col in snp_df.columns:
                    col_lower = str(col).lower()
                    if 'chrom' in col_lower or col_lower == 'chr':
                        chrom_col = col
                    elif col_lower == 'pos' or col_lower == 'position':
                        pos_col = col
                    elif 'sample_ids' in col_lower or 'affected_samples' in col_lower or '涉及样本' in col_lower:
                        samples_col = col

                if chrom_col is None or pos_col is None:
                    # Assume first two columns are CHROM and POS
                    chrom_col = snp_df.columns[0]
                    pos_col = snp_df.columns[1]

                if samples_col is None:
                    # Find column with space-separated values
                    for col in snp_df.columns:
                        if snp_df[col].dtype == object:
                            sample_val = str(snp_df[col].dropna().iloc[0]) if len(snp_df[col].dropna()) > 0 else ""
                            if " " in sample_val and len(sample_val.split()) > 1:
                                samples_col = col
                                break

                if samples_col is None:
                    raise ValueError("Cannot find sample column in SNP data")

                snp_df['snp_id'] = snp_df[chrom_col].astype(str) + '_' + snp_df[pos_col].astype(str)
                snp_df = snp_df.drop_duplicates(subset=['snp_id'], keep='first')

                # Collect samples
                all_samples = set()
                for s in snp_df[samples_col].dropna():
                    all_samples.update(str(s).strip().split())
                all_samples = sorted(list(all_samples))
                sample_to_idx = {s: i for i, s in enumerate(all_samples)}

                self._log(f"      SNP variants: {len(snp_df)}")
                self._log(f"      Unique samples: {len(all_samples)}")

                # Build sparse matrix
                from scipy.sparse import csr_matrix
                row_indices = []
                col_indices = []

                for col_idx, (_, row) in enumerate(snp_df.iterrows()):
                    snp_id = str(row['snp_id'])
                    if snp_id not in snp_loader.snp_to_idx:
                        continue

                    ref_col_idx = snp_loader.snp_to_idx[snp_id]
                    samples_str = str(row[samples_col]).strip()
                    for sample in samples_str.split():
                        if sample in sample_to_idx:
                            row_indices.append(sample_to_idx[sample])
                            col_indices.append(ref_col_idx)

                data = np.ones(len(row_indices), dtype=np.float32)
                X_snp = csr_matrix(
                    (data, (row_indices, col_indices)),
                    shape=(len(all_samples), snp_loader.n_snps),
                    dtype=np.float32
                )
                snp_samples = all_samples

                self._log(f"      Sparse matrix: {X_snp.shape}")
                self.predict_progress['value'] = 50

                # Align samples
                self._log("[4/5] Aligning samples...")
                X_snp_a, X_env_a, aligned_samples = align_samples(
                    X_snp, snp_samples, X_env, env_samples
                )
                self._log(f"      Aligned: {len(aligned_samples)} samples")
                self.predict_progress['value'] = 65

                # Predict
                self._log("[5/5] Running prediction...")
                results = predictor.predict(X_snp_a, X_env_a, aligned_samples)
                self.predict_progress['value'] = 90

            # Store results
            self.predictions = pd.DataFrame(results)
            if '_note' in self.predictions.columns:
                self.predictions = self.predictions.drop(columns=['_note'])

            self._log("\n" + "=" * 50)
            self._log("Prediction Complete!")
            self._log("=" * 50)
            self._log(f"Samples: {len(self.predictions)}")
            for col in ['Network_Hub', 'Clone_Advantage', 'Persistence', 'Spatial_Connectivity']:
                if col in self.predictions.columns:
                    self._log(f"  {col}: mean={self.predictions[col].mean():.3f}, "
                              f"range=[{self.predictions[col].min():.3f}, {self.predictions[col].max():.3f}]")

            self.predict_progress['value'] = 100
            self._set_status(f"Prediction complete: {len(self.predictions)} samples")

            # Switch to results page
            self.root.after(500, self._show_results_after_predict)

        except Exception as e:
            self._log(f"\n[ERROR] {str(e)}")
            self._log(traceback.format_exc())
            self._set_status("Prediction failed")
            Messagebox.show_error(f"Prediction failed:\n{str(e)}", "Error")

        finally:
            self.run_btn.config(state=NORMAL)

    def _show_results_after_predict(self):
        """Show results page after prediction"""
        self._update_result_table()
        self.show_page("results")
        Messagebox.show_info(f"Prediction complete!\n{len(self.predictions)} samples predicted.", "Success")

    # ==================== RESULTS DISPLAY ====================

    def _update_result_table(self):
        """Update result table view"""
        if self.predictions is None:
            return

        # Clear result frame
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        # Create treeview
        tree_frame = ttk.Frame(self.result_frame)
        tree_frame.pack(fill=BOTH, expand=True)

        tree = ttk.Treeview(tree_frame, show="headings", height=20)
        tree.pack(side=LEFT, fill=BOTH, expand=True)

        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        scroll.pack(side=RIGHT, fill=Y)
        tree.configure(yscrollcommand=scroll.set)

        hscroll = ttk.Scrollbar(self.result_frame, orient="horizontal", command=tree.xview)
        hscroll.pack(fill=X)
        tree.configure(xscrollcommand=hscroll.set)

        # Columns
        cols = list(self.predictions.columns)
        tree['columns'] = cols
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=120)

        # Data
        for _, row in self.predictions.iterrows():
            values = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values]
            tree.insert("", END, values=values)

    def _show_result_table(self):
        """Show results as table"""
        self._update_result_table()

    # ===== ADVANCED CHART METHODS =====

    def _setup_dark_axes(self, ax):
        """Apply dark theme to a matplotlib axes"""
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white', labelsize=9)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _show_scatter_plot(self):
        """Enhanced scatter plots with regression lines"""
        if self.predictions is None:
            Messagebox.show_warning("No predictions to plot. Run prediction first.", "No Data")
            return

        for widget in self.result_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(14, 10), dpi=100)
        fig.patch.set_facecolor('#0f0f23')

        dims = ['Network_Hub', 'Clone_Advantage', 'Persistence', 'Spatial_Connectivity']
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        n = len(self.predictions)

        # 2x2 scatter with trend lines
        for i, (dim, color) in enumerate(zip(dims, colors)):
            if dim not in self.predictions.columns:
                continue
            ax = fig.add_subplot(2, 2, i + 1)
            self._setup_dark_axes(ax)

            values = self.predictions[dim].values
            x = np.arange(len(values))

            # Scatter with gradient color based on value
            scatter = ax.scatter(x, values, c=values, cmap='coolwarm',
                                s=80, alpha=0.8, edgecolors='white', linewidths=0.5)

            # Trend line
            z = np.polyfit(x, values, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), '--', color='yellow', alpha=0.7, linewidth=1.5, label='Trend')

            # Mean line
            ax.axhline(y=values.mean(), color='white', linestyle=':', alpha=0.4)
            ax.text(len(values)*0.02, values.mean()+0.03, f'mean={values.mean():.3f}',
                   color='white', fontsize=8)

            ax.set_title(f'{dim.replace("_", " ")}  (n={n})', color='white', fontsize=11, fontweight='bold')
            ax.set_xlabel('Sample Index', color='white')
            ax.set_ylabel('Score', color='white')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.15, color='white')

            # Colorbar
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors='white', labelsize=7)

        fig.suptitle('4D Score Scatter Analysis with Trend Lines', color='white', fontsize=14, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def _show_violin_plot(self):
        """Violin plots with swarm overlay"""
        if self.predictions is None:
            Messagebox.show_warning("No predictions to plot. Run prediction first.", "No Data")
            return

        for widget in self.result_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(12, 7), dpi=100)
        fig.patch.set_facecolor('#0f0f23')

        ax = fig.add_subplot(111)
        self._setup_dark_axes(ax)

        dims = ['Network_Hub', 'Clone_Advantage', 'Persistence', 'Spatial_Connectivity']
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        data_to_plot = []
        positions = []
        labels = []

        for i, dim in enumerate(dims):
            if dim in self.predictions.columns:
                data_to_plot.append(self.predictions[dim].values)
                positions.append(i)
                labels.append(dim.replace('_', '\n'))

        # Violin plot
        parts = ax.violinplot(data_to_plot, positions=positions, widths=0.6,
                              showmeans=True, showmedians=True)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.5)
            pc.set_edgecolor('white')
            pc.set_linewidth(1)

        for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
            if partname in parts:
                parts[partname].set_color('white')
                parts[partname].set_linewidth(1.5)

        # Overlay swarm-like scatter
        for i, (data, color) in enumerate(zip(data_to_plot, colors)):
            jitter = np.random.normal(i, 0.08, size=len(data))
            ax.scatter(jitter, data, c=color, s=25, alpha=0.6, edgecolors='white', linewidths=0.2, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, color='white', fontsize=10)
        ax.set_ylabel('Score', color='white', fontsize=11)
        ax.set_title('4D Score Distribution - Violin + Swarm Plot', color='white', fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', alpha=0.15, color='white')

        # Add sample count annotation
        for i, data in enumerate(data_to_plot):
            ax.annotate(f'n={len(data)}', xy=(i, 0.95), ha='center', color='white',
                       fontsize=9, fontweight='bold')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def _show_heatmap_plot(self):
        """Sample x Dimension heatmap with dendrogram-style sorting"""
        if self.predictions is None:
            Messagebox.show_warning("No predictions to plot. Run prediction first.", "No Data")
            return

        for widget in self.result_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(12, 9), dpi=100)
        fig.patch.set_facecolor('#0f0f23')

        dims = ['Network_Hub', 'Clone_Advantage', 'Persistence', 'Spatial_Connectivity']
        dims_present = [d for d in dims if d in self.predictions.columns]

        if len(dims_present) == 0:
            return

        # Prepare data matrix (samples x dimensions)
        data_matrix = self.predictions[dims_present].values
        sample_labels = self.predictions['sample_id'].values

        # Sort by composite score for better visualization
        composite = data_matrix.mean(axis=1)
        sort_idx = np.argsort(composite)[::-1]
        data_matrix = data_matrix[sort_idx]
        sample_labels = sample_labels[sort_idx]

        # Show top 50 samples max
        max_show = min(50, len(data_matrix))
        data_matrix = data_matrix[:max_show]
        sample_labels = sample_labels[:max_show]

        ax = fig.add_subplot(111)
        self._setup_dark_axes(ax)

        im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(dims_present)))
        ax.set_xticklabels([d.replace('_', '\n') for d in dims_present], color='white', fontsize=10)
        ax.set_yticks(np.arange(len(sample_labels)))
        ax.set_yticklabels(sample_labels, color='white', fontsize=7)

        # Add value annotations
        for i in range(len(sample_labels)):
            for j in range(len(dims_present)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black" if data_matrix[i, j] > 0.5 else "white",
                              fontsize=6)

        ax.set_title(f'Sample x Dimension Heatmap (Top {max_show} by Mean Score)', color='white', fontsize=13, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Score (0-1)', color='white', fontsize=10)
        cbar.ax.tick_params(colors='white', labelsize=8)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def _show_bubble_plot(self):
        """3D bubble chart: Network vs Clone vs Persistence, bubble=Spatial, color=composite"""
        if self.predictions is None:
            Messagebox.show_warning("No predictions to plot. Run prediction first.", "No Data")
            return

        for widget in self.result_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(13, 10), dpi=100)
        fig.patch.set_facecolor('#0f0f23')

        dims = ['Network_Hub', 'Clone_Advantage', 'Persistence', 'Spatial_Connectivity']
        if not all(d in self.predictions.columns for d in dims):
            Messagebox.show_warning("All 4 dimensions required for bubble plot.", "Missing Data")
            return

        # Main 3D scatter (simulated with 2D + size + color)
        ax = fig.add_subplot(2, 2, 1)
        self._setup_dark_axes(ax)

        x = self.predictions['Network_Hub'].values
        y = self.predictions['Clone_Advantage'].values
        z = self.predictions['Persistence'].values
        size = self.predictions['Spatial_Connectivity'].values
        composite = (x + y + z) / 3

        scatter = ax.scatter(x, y, s=size*500+20, c=composite, cmap='plasma',
                            alpha=0.7, edgecolors='white', linewidths=0.5)
        ax.set_xlabel('Network Hub', color='white')
        ax.set_ylabel('Clone Advantage', color='white')
        ax.set_title('Bubble: x=Network, y=Clone, size=Spatial, color=Composite', color='white', fontsize=10, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.15, color='white')
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046)
        cbar.ax.tick_params(colors='white', labelsize=7)

        # Pairwise correlations
        for idx, (xdim, ydim) in enumerate([('Network_Hub', 'Persistence'), ('Clone_Advantage', 'Spatial_Connectivity'), ('Network_Hub', 'Spatial_Connectivity')]):
            ax = fig.add_subplot(2, 2, idx + 2)
            self._setup_dark_axes(ax)
            xv = self.predictions[xdim].values
            yv = self.predictions[ydim].values

            # Hexbin for density
            hb = ax.hexbin(xv, yv, gridsize=15, cmap='viridis', alpha=0.8, mincnt=1)
            ax.set_xlabel(xdim.replace('_', ' '), color='white')
            ax.set_ylabel(ydim.replace('_', ' '), color='white')

            # Correlation
            corr = np.corrcoef(xv, yv)[0, 1]
            ax.set_title(f'{xdim.replace("_", " ")} vs {ydim.replace("_", " ")}\nr={corr:.3f}',
                        color='white', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.15, color='white')
            cbar = fig.colorbar(hb, ax=ax, fraction=0.046)
            cbar.ax.tick_params(colors='white', labelsize=7)

        fig.suptitle('4D Multi-Dimensional Bubble & Density Analysis', color='white', fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def _show_radar_plot(self):
        """Enhanced radar plot with individual sample overlays"""
        if self.predictions is None:
            Messagebox.show_warning("No predictions to plot. Run prediction first.", "No Data")
            return

        for widget in self.result_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(10, 10), dpi=100)
        fig.patch.set_facecolor('#1a1a2e')

        ax = fig.add_subplot(111, projection='polar')

        dims = ['Network_Hub', 'Clone_Advantage', 'Persistence', 'Spatial_Connectivity']
        n_dims = len(dims)
        angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
        angles += angles[:1]

        # Plot a few representative samples
        n_show = min(5, len(self.predictions))
        sample_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']

        for i in range(n_show):
            values = [self.predictions[d].iloc[i] for d in dims]
            values += values[:1]
            color = sample_colors[i % len(sample_colors)]
            ax.plot(angles, values, 'o-', linewidth=2, alpha=0.8,
                   color=color, markersize=5, label=self.predictions['sample_id'].iloc[i])
            ax.fill(angles, values, alpha=0.1, color=color)

        # Plot mean profile
        mean_values = [self.predictions[d].mean() for d in dims]
        mean_values += mean_values[:1]
        ax.plot(angles, mean_values, 's-', linewidth=3, color='yellow',
               markersize=9, label='Mean', zorder=10)
        ax.fill(angles, mean_values, alpha=0.2, color='yellow')

        # Styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([d.replace('_', ' ') for d in dims], color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.set_ylim(0, 1)
        ax.set_title('4D Risk Profile Radar\n(Individual Samples + Mean)',
                    color='white', fontsize=14, fontweight='bold', pad=30)
        ax.grid(True, alpha=0.4, color='cyan', linewidth=0.5)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10,
                 labelcolor='white', facecolor='#1a1a2e', edgecolor='white')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def _show_boxdot_plot(self):
        """Box plot with individual data points and mean/median annotations"""
        if self.predictions is None:
            Messagebox.show_warning("No predictions to plot. Run prediction first.", "No Data")
            return

        for widget in self.result_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(13, 8), dpi=100)
        fig.patch.set_facecolor('#0f0f23')

        dims = ['Network_Hub', 'Clone_Advantage', 'Persistence', 'Spatial_Connectivity']
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']

        # Left: Box + swarm
        ax1 = fig.add_subplot(2, 1, 1)
        self._setup_dark_axes(ax1)

        positions = []
        data_to_plot = []
        for i, dim in enumerate(dims):
            if dim in self.predictions.columns:
                positions.append(i)
                data_to_plot.append(self.predictions[dim].values)

        bp = ax1.boxplot(data_to_plot, positions=positions, patch_artist=True,
                        widths=0.4, showmeans=False, showfliers=False)

        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
            patch.set_edgecolor('white')
            patch.set_linewidth(1.5)

        for whisker in bp['whiskers']:
            whisker.set_color('white')
            whisker.set_linewidth(1)
        for cap in bp['caps']:
            cap.set_color('white')
        for median in bp['medians']:
            median.set_color('yellow')
            median.set_linewidth(2.5)

        # Overlay individual points with jitter
        for i, (data, color) in enumerate(zip(data_to_plot, colors)):
            jitter = np.random.normal(i, 0.06, size=len(data))
            ax1.scatter(jitter, data, c=color, s=35, alpha=0.7,
                       edgecolors='white', linewidths=0.3, zorder=5)

            # Annotations
            mean = np.mean(data)
            median = np.median(data)
            ax1.annotate(f'μ={mean:.3f}', xy=(i, mean), xytext=(i+0.25, mean+0.05),
                        fontsize=8, color='white', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='white', alpha=0.5))
            ax1.plot([i-0.15, i+0.15], [mean, mean], color='white', linewidth=2, linestyle='--')

        ax1.set_xticks(positions)
        ax1.set_xticklabels([d.replace('_', '\n') for d in dims[:len(data_to_plot)]],
                           color='white', fontsize=11)
        ax1.set_ylabel('Score', color='white', fontsize=11)
        ax1.set_title('Box Plot + Individual Points with Mean/Median', color='white', fontsize=13, fontweight='bold')
        ax1.set_ylim(-0.05, 1.15)
        ax1.grid(axis='y', alpha=0.15, color='white')

        # Right: Stacked histogram
        ax2 = fig.add_subplot(2, 1, 2)
        self._setup_dark_axes(ax2)

        bins = np.linspace(0, 1, 21)
        for i, (dim, color) in enumerate(zip(dims, colors)):
            if dim in self.predictions.columns:
                data = self.predictions[dim].values
                ax2.hist(data, bins=bins, alpha=0.5, color=color,
                        label=dim.replace('_', ' '), edgecolor='white', linewidth=0.5)

                # KDE overlay
                from scipy.stats import gaussian_kde
                if len(data) > 1:
                    kde = gaussian_kde(data)
                    x_range = np.linspace(0, 1, 200)
                    kde_vals = kde(x_range)
                    ax2.plot(x_range, kde_vals * len(data) * (bins[1]-bins[0]),
                            color=color, linewidth=2, linestyle='--')

        ax2.set_xlabel('Score', color='white', fontsize=11)
        ax2.set_ylabel('Frequency', color='white', fontsize=11)
        ax2.set_title('Stacked Histogram with KDE Density Curves', color='white', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9, labelcolor='white', loc='upper right')
        ax2.grid(axis='y', alpha=0.15, color='white')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    # ==================== TEXT WIDGET HELPERS ====================

    def _set_text_state(self, widget, state):
        """Safely set state on ScrolledText widget"""
        if hasattr(widget, 'text'):
            widget.text.config(state=state)
        else:
            widget.config(state=state)

    # ==================== EXPORT ====================

    def _export_predictions(self):
        """Export predictions to CSV"""
        if self.predictions is None:
            Messagebox.show_warning("No predictions to export. Run prediction first.", "No Data")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="predictions.csv"
        )
        if path:
            try:
                self.predictions.to_csv(path, index=False, encoding='utf-8-sig')
                self._set_status(f"Exported: {path}")
                Messagebox.show_info(f"Predictions exported to:\n{path}", "Export Complete")
            except Exception as e:
                Messagebox.show_error(f"Export failed:\n{str(e)}", "Error")

    # ==================== UTILITIES ====================

    def _set_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def _show_docs(self):
        """Show documentation"""
        Messagebox.show_info(
            "Documentation\n\n"
            "1. Load SNP data (long or wide format)\n"
            "2. Load environment data (CSV with required columns)\n"
            "3. Preview data to verify\n"
            "4. Run prediction\n"
            "5. View results and export\n\n"
            "For detailed instructions, see README.md",
            "Help"
        )

    def _show_about(self):
        """Show about dialog"""
        self.show_page("about")


def main():
    """Main entry point"""
    # Set DPI awareness for Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = ttk.Window(themename="darkly")
    app = TransmissionRiskApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
