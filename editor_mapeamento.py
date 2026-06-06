"""
editor_mapeamento.py — Editor visual de mapeamento de condições fNIRS
Timeline de eventos individuais com labels editáveis por ocorrência.

Uso:
    python editor_mapeamento.py --arquivo dp372_pre.nirs --nome dp372_pre.nirs
    python editor_mapeamento.py --pasta dados/
    python editor_mapeamento.py --saida mapeamento_condicoes.csv
"""

import argparse
import re
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


BG          = "#f5f6fa"
BG_CARD     = "#ffffff"
BG_HEADER   = "#2c3e50"
FG_HEADER   = "#ffffff"
BG_ROW_A    = "#f0f4f8"
BG_ROW_B    = "#ffffff"
BG_COL_HDR  = "#eaecf0"
ACCENT      = "#3498db"
ACCENT_HOV  = "#2980b9"
SUCCESS     = "#27ae60"
SUCCESS_HOV = "#219a52"
DANGER      = "#e74c3c"
DANGER_HOV  = "#c0392b"
FG_COND     = "#7f8c8d"
FG_DIM      = "#b2bec3"
FG_LABEL    = "#2c3e50"
FG_TIME     = "#636e72"

FONT_BASE   = ("Segoe UI", 10)
FONT_BOLD   = ("Segoe UI", 10, "bold")
FONT_SMALL  = ("Segoe UI", 9)
FONT_FILE   = ("Segoe UI", 11, "bold")
FONT_MONO   = ("Consolas", 10)
FONT_TIME   = ("Consolas", 9)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pasta",   default=None)
    p.add_argument("--arquivo", default=None)
    p.add_argument("--nome",    default=None)
    p.add_argument("--saida",   default="mapeamento_condicoes.csv")
    return p.parse_args()


def ler_ocorrencias(nirs_path: Path) -> list[dict]:
    """
    Returns list of dicts sorted by onset time:
      {onset_s: float, cond: str, idx: int}
    One entry per individual trial/occurrence.
    """
    sys.path.insert(0, str(nirs_path.parent))
    import io, contextlib
    try:
        from pipeline_spm import load_nirs, FS
        with contextlib.redirect_stdout(io.TextIOWrapper(io.BytesIO(), encoding="utf-8")):
            rec = load_nirs(str(nirs_path))
        rows = []
        for cond, onsets in rec["events"].items():
            for idx, onset in enumerate(sorted(onsets), 1):
                rows.append({
                    "onset_s": float(onset) / float(FS),
                    "cond": cond,
                    "idx": idx,
                })
        return sorted(rows, key=lambda r: r["onset_s"])
    except Exception as e:
        return [{"onset_s": 0.0, "cond": f"[ERRO: {e}]", "idx": 1}]


class HoverButton(tk.Button):
    def __init__(self, master, bg_normal, bg_hover, **kw):
        super().__init__(master, bg=bg_normal, activebackground=bg_hover,
                         relief="flat", cursor="hand2", **kw)
        self.bind("<Enter>", lambda _: self.config(bg=bg_hover))
        self.bind("<Leave>", lambda _: self.config(bg=bg_normal))


class App(tk.Tk):
    def __init__(self, pasta_inicial: str | None, saida: str):
        super().__init__()
        self.title("Editor de Mapeamento — fNIRS")
        self.configure(bg=BG)
        self.minsize(720, 480)
        self.geometry("860x640")

        self.saida = Path(saida)
        # [(arquivo_nome, [ocorrencia_dict, ...])]
        self._itens: list[tuple[str, list[dict]]] = []
        # [{arquivo, cond, idx, onset_s, var, frame, default}]
        self.entries: list[dict] = []

        self._build_toolbar()
        self._build_body()
        self._build_footer()

        if pasta_inicial:
            self._carregar_pasta(Path(pasta_inicial))

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=BG_HEADER, pady=10, padx=14)
        bar.pack(fill="x")

        tk.Label(bar, text="Mapeamento de Condições fNIRS",
                 bg=BG_HEADER, fg=FG_HEADER, font=FONT_FILE).pack(side="left")

        HoverButton(bar, ACCENT, ACCENT_HOV,
                    text="  Abrir pasta  ", fg="white", font=FONT_BOLD,
                    padx=8, pady=4,
                    command=self._selecionar_pasta).pack(side="right")

        self._lbl_pasta = tk.Label(bar, text="Nenhuma pasta selecionada",
                                   bg=BG_HEADER, fg="#bdc3c7", font=FONT_SMALL)
        self._lbl_pasta.pack(side="left", padx=12)

    def _build_body(self):
        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(container, orient="vertical", command=self._canvas.yview)
        self._scroll_frame = tk.Frame(self._canvas, bg=BG)

        self._scroll_frame.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        )
        self._canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        self._canvas.configure(yscrollcommand=sb.set)

        sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        self._canvas.bind_all("<MouseWheel>",
            lambda e: self._canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        self._lbl_empty = tk.Label(self._scroll_frame,
                                   text="Selecione uma pasta para começar.",
                                   bg=BG, fg=FG_COND, font=FONT_BASE, pady=40)
        self._lbl_empty.pack()

    def _build_footer(self):
        footer = tk.Frame(self, bg=BG, pady=10, padx=14)
        footer.pack(fill="x", side="bottom")

        self._lbl_status = tk.Label(footer, text="", bg=BG, fg=FG_COND, font=FONT_SMALL)
        self._lbl_status.pack(side="left")

        HoverButton(footer, SUCCESS, SUCCESS_HOV,
                    text="  Salvar CSV  ", fg="white", font=FONT_BOLD,
                    padx=12, pady=6,
                    command=self._salvar).pack(side="right")

        HoverButton(footer, "#95a5a6", "#7f8c8d",
                    text="Resetar labels", fg="white", font=FONT_SMALL,
                    padx=8, pady=6,
                    command=self._resetar).pack(side="right", padx=8)

    # ── Lógica ───────────────────────────────────────────────────────────────

    def _selecionar_pasta(self):
        pasta = filedialog.askdirectory(title="Selecione a pasta com arquivos .nirs")
        if pasta:
            self._carregar_pasta(Path(pasta))

    def _carregar_pasta(self, pasta: Path):
        nirs_files = sorted(pasta.glob("*.nirs"))
        if not nirs_files:
            messagebox.showwarning("Aviso", f"Nenhum arquivo .nirs encontrado em:\n{pasta}")
            return
        self._lbl_pasta.config(text=str(pasta))
        self._lbl_empty.config(text=f"Lendo {len(nirs_files)} arquivo(s)...")
        self._lbl_status.config(text="Carregando...")
        self.update()
        threading.Thread(target=self._ler_arquivos, args=(nirs_files,), daemon=True).start()

    def _ler_arquivos(self, nirs_files: list[Path]):
        itens = [(f.name, ler_ocorrencias(f)) for f in nirs_files]
        self.after(0, lambda: self._popular_itens(itens))

    @staticmethod
    def _default_label(cond: str) -> str:
        stripped = re.sub(r'\d+$', '', cond).strip()
        return stripped if stripped else cond

    def _popular_itens(self, itens: list[tuple[str, list[dict]]]):
        self._itens = itens
        self._redraw()

    def _redraw(self):
        for w in self._scroll_frame.winfo_children():
            w.destroy()
        self.entries.clear()

        if not self._itens:
            tk.Label(self._scroll_frame, text="Nenhum arquivo carregado.",
                     bg=BG, fg=FG_COND, font=FONT_BASE, pady=40).pack()
            return

        # Cabeçalho de colunas
        col_hdr = tk.Frame(self._scroll_frame, bg=BG_COL_HDR, padx=16, pady=6)
        col_hdr.pack(fill="x", pady=(8, 0))
        tk.Label(col_hdr, text="Tempo", bg=BG_COL_HDR, fg=FG_COND,
                 font=FONT_SMALL, width=9, anchor="w").pack(side="left")
        tk.Label(col_hdr, text="Evento", bg=BG_COL_HDR, fg=FG_COND,
                 font=FONT_SMALL, width=8, anchor="w").pack(side="left")
        tk.Label(col_hdr, text="Label de saída", bg=BG_COL_HDR, fg=FG_COND,
                 font=FONT_SMALL, anchor="w").pack(side="left", padx=(16, 0))

        for nome, ocorrs in self._itens:
            # Divisor de arquivo
            div = tk.Frame(self._scroll_frame, bg="#dfe6e9", padx=16, pady=5)
            div.pack(fill="x", pady=(8, 0))
            tk.Label(div, text=nome, bg="#dfe6e9", fg="#636e72",
                     font=FONT_SMALL, anchor="w").pack(side="left")
            conds_unicas = len({o["cond"] for o in ocorrs})
            tk.Label(div, text=f"{conds_unicas} condições · {len(ocorrs)} eventos",
                     bg="#dfe6e9", fg=FG_DIM, font=FONT_SMALL).pack(side="right")

            for i, ocorr in enumerate(ocorrs):
                self._add_row(nome, ocorr, row_idx=i)

        n_total = sum(len(o) for _, o in self._itens)
        self._lbl_status.config(
            text=f"{len(self._itens)} arquivo(s) · {n_total} eventos · {len(self.entries)} mapeamentos"
        )

    def _add_row(self, arquivo: str, ocorr: dict, row_idx: int = 0):
        cond    = ocorr["cond"]
        idx     = ocorr["idx"]
        onset_s = ocorr["onset_s"]
        default = self._default_label(cond)

        bg = BG_ROW_A if row_idx % 2 == 0 else BG_ROW_B
        frame = tk.Frame(self._scroll_frame, bg=bg, padx=16, pady=4)
        frame.pack(fill="x")

        # Tempo de onset
        mins = int(onset_s // 60)
        secs = onset_s % 60
        time_str = f"{mins}:{secs:05.2f}" if mins > 0 else f"{onset_s:.2f}s"
        tk.Label(frame, text=time_str, bg=bg, fg=FG_TIME,
                 font=FONT_TIME, width=9, anchor="w").pack(side="left")

        # Rótulo do evento (ex: "D_3")
        ev_label = f"{cond}_{idx}"
        tk.Label(frame, text=ev_label, bg=bg, fg=FG_LABEL,
                 font=FONT_MONO, width=7, anchor="w").pack(side="left")

        # Seta
        tk.Label(frame, text="→", bg=bg, fg=FG_DIM,
                 font=FONT_BASE, width=3).pack(side="left")

        # Campo de label editável
        var = tk.StringVar(value=default)
        entry = tk.Entry(frame, textvariable=var, font=FONT_MONO,
                         relief="solid", bd=1, fg=FG_LABEL,
                         bg="white", insertbackground=ACCENT,
                         highlightthickness=1, highlightcolor=ACCENT,
                         highlightbackground="#dfe6e9", width=24)
        entry.pack(side="left", ipady=3, padx=(4, 0))
        entry.bind("<FocusIn>",  lambda e, w=entry: w.config(bg="#eaf4fd"))
        entry.bind("<FocusOut>", lambda e, w=entry: w.config(bg="white"))

        entry_data = {
            "arquivo": arquivo,
            "cond": cond,
            "idx": idx,
            "onset_s": onset_s,
            "var": var,
            "frame": frame,
            "default": default,
        }

        # Botão remover
        HoverButton(frame, DANGER, DANGER_HOV,
                    text="×", fg="white", font=FONT_BOLD,
                    width=2, padx=4, pady=1,
                    command=lambda d=entry_data: self._remove_row(d)
                    ).pack(side="left", padx=(8, 0))

        self.entries.append(entry_data)

    def _remove_row(self, entry_data: dict):
        if entry_data in self.entries:
            self.entries.remove(entry_data)
        entry_data["frame"].destroy()
        n = len(self.entries)
        self._lbl_status.config(text=f"{n} mapeamentos definidos", fg=FG_COND)

    def _resetar(self):
        for e in self.entries:
            e["var"].set(e["default"])

    def _salvar(self):
        if not self.entries:
            messagebox.showwarning("Aviso", "Nenhum mapeamento definido.")
            return

        linhas = ["arquivo;condicao;ocorrencia;label"]
        for e in self.entries:
            label = e["var"].get().strip() or e["cond"]
            linhas.append(f"{e['arquivo']};{e['cond']};{e['idx']};{label}")

        try:
            self.saida.write_text("\n".join(linhas) + "\n", encoding="utf-8")
            self._lbl_status.config(
                text=f"Salvo: {self.saida}  ({len(self.entries)} mapeamentos)",
                fg=SUCCESS
            )
            messagebox.showinfo("Salvo",
                f"Mapeamento salvo em:\n{self.saida.resolve()}\n\n"
                f"{len(self.entries)} linha(s).")
        except Exception as ex:
            messagebox.showerror("Erro ao salvar", str(ex))


def main():
    args = parse_args()
    app = App(pasta_inicial=args.pasta, saida=args.saida)

    if args.arquivo and args.nome:
        nirs_path = Path(args.arquivo)
        nome = args.nome
        def _pre_popular():
            ocorrs = ler_ocorrencias(nirs_path)
            app._popular_itens([(nome, ocorrs)])
            app._lbl_pasta.config(text=f"Arquivo: {nome}")
        app.after(100, _pre_popular)

    app.mainloop()


if __name__ == "__main__":
    main()
