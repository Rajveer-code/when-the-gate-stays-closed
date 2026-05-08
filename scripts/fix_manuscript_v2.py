"""
fix_manuscript_v2.py
====================
Comprehensive docx fix for new version 3 of the manuscript.
Applies all corrections from the user's instruction list and re-embeds figures.

Run from repo root:
    python scripts/fix_manuscript_v2.py
"""
import sys, zipfile, io, os, copy, random
sys.stdout.reconfigure(encoding="utf-8")
import lxml.etree as ET

DOCX_IN  = r"C:\Users\Asus\Downloads\when_the_gate_stays_closed_FINAL_SUBMISSION (3).docx"
DOCX_OUT = "paper/when_the_gate_stays_closed_FINAL_SUBMISSION.docx"
FIG_DIR  = "figures"

W   = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14 = "http://schemas.microsoft.com/office/word/2010/wordml"

def wtag(n):
    return f"{{{W}}}{n}"

# ── Load docx ─────────────────────────────────────────────────────────────────
with open(DOCX_IN, "rb") as f:
    docx_bytes = f.read()

with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as zin:
    files = {n: zin.read(n) for n in zin.namelist()}

doc_xml = files["word/document.xml"]
root    = ET.fromstring(doc_xml)
ns      = {"w": W, "w14": W14}
paras   = root.findall(".//w:p", ns)
tbls    = root.findall(".//w:tbl", ns)
print(f"Loaded: {len(paras)} paragraphs, {len(tbls)} tables")

# ── Helpers ───────────────────────────────────────────────────────────────────
def para_text(p):
    return "".join(t.text or "" for t in p.findall(".//w:t", ns))

def set_para_text(p, new_text, preserve_rpr=True):
    """Rewrite paragraph to a single run with new_text, keeping first run rPr."""
    first_r = p.find(".//w:r", ns)
    rpr = None
    if preserve_rpr and first_r is not None:
        rpr_el = first_r.find("w:rPr", ns)
        if rpr_el is not None:
            rpr = copy.deepcopy(rpr_el)
    for r in list(p.findall(".//w:r", ns)):
        r.getparent().remove(r)
    new_r = ET.SubElement(p, wtag("r"))
    if rpr is not None:
        new_r.insert(0, rpr)
    new_t = ET.SubElement(new_r, wtag("t"))
    new_t.text = new_text
    if new_text and (new_text[0] == " " or new_text[-1] == " "):
        new_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return True

# ════════════════════════════════════════════════════════════════════════════
# FIX 1: Update Table 5 (docx index 5) — subperiod Sharpe values
#         Manuscript "Table 6" — subperiod analysis
# ════════════════════════════════════════════════════════════════════════════
print("\n[FIX 1] Table 5 — Subperiod Sharpe values from subperiod_analysis.csv")
tbl5  = tbls[5]
rows5 = tbl5.findall("w:tr", ns)
print(f"  Table 5 rows: {len(rows5)}")

# Ground truth from subperiod_analysis.csv (Sharpe rounded to 2 d.p.)
# Columns: [P1_AnnRet(1), P1_Sharpe(2), P1_MDD(3), P2_AnnRet(4), P2_Sharpe(5),
#           P2_MDD(6), P3_AnnRet(7), P3_Sharpe(8), P3_MDD(9)]
# Ann.Ret and MDD kept as in original docx (only Sharpe columns updated per instruction)
SHARPE_DATA = {
    # strategy_name_substring: (P1_sharpe, P2_sharpe, P3_sharpe)
    "TopK1":       ("-1.60", "+0.76", "-0.59"),
    "Equal-Weight":("-1.08", "+1.72", "+0.43"),
    "Random":      ("-0.79", "+0.42", "-0.32"),
    "SPY":         ("-1.58", "+1.18", "+0.50"),
}
# Sharpe columns are at cell indices 2, 5, 8 (0-indexed within row)
SHARPE_COL_IDX = [2, 5, 8]

updated_rows = 0
for row in rows5[1:]:   # skip header row
    cells = row.findall("w:tc", ns)
    if not cells:
        continue
    strat_text = para_text(cells[0].find("w:p", ns) or cells[0]).strip()

    matched = None
    for key, vals in SHARPE_DATA.items():
        if key in strat_text:
            matched = vals
            break

    if matched is None:
        print(f"  WARNING: Could not match '{strat_text}'")
        continue

    p1_s, p2_s, p3_s = matched
    new_sharpes = {2: p1_s, 5: p2_s, 8: p3_s}
    for col_i, val in new_sharpes.items():
        if col_i < len(cells):
            p_el = cells[col_i].find("w:p", ns)
            if p_el is None:
                p_el = ET.SubElement(cells[col_i], wtag("p"))
            set_para_text(p_el, val)

    print(f"  '{strat_text}': P1={p1_s}, P2={p2_s}, P3={p3_s}")
    updated_rows += 1

print(f"  Updated {updated_rows}/{len(rows5)-1} rows")

# ════════════════════════════════════════════════════════════════════════════
# FIX 2: N=100 p-value 0.948 → 0.052 (upper-tail correction)
# ════════════════════════════════════════════════════════════════════════════
print("\n[FIX 2] N=100 p-value: 0.948 → 0.052 (upper-tail)")
count_948 = 0

# Table 7 (universe robustness table at docx index 7)
tbl7 = tbls[7]
for row in tbl7.findall("w:tr", ns):
    for cell in row.findall("w:tc", ns):
        for p_el in cell.findall(".//w:p", ns):
            full = para_text(p_el)
            if "0.948" in full:
                set_para_text(p_el, full.replace("0.948", "0.052"))
                print(f"  Table 7 cell: '{full}' → '0.052'")
                count_948 += 1

# Paragraph text throughout document
for i, p in enumerate(paras):
    full = para_text(p)
    if "0.948" in full:
        set_para_text(p, full.replace("0.948", "0.052"))
        print(f"  Para {i}: 0.948 → 0.052 in: {full[:120]}")
        count_948 += 1

print(f"  Total replacements: {count_948}")

# ════════════════════════════════════════════════════════════════════════════
# FIX 3: Para 293 — rewrite with correct permutation p-values
# ════════════════════════════════════════════════════════════════════════════
print("\n[FIX 3] Para 293 — permutation p-values 0.797/0.754 → 0.599/0.601")
p293 = paras[293]
old_text = para_text(p293)
print(f"  Old (first 120): {old_text[:120]}")

CLEAN_293 = (
    "The permutation test — Type A (IID bootstrap of the 1,512 daily IC values) "
    "and Type B (block bootstrap, block = 5 days) — yields empirical p-values "
    "of 0.599 and 0.601 respectively, both far from the 0.05 threshold. Under Type A, "
    "the observed mean IC lies at the centre of the null distribution, confirming the "
    "observed IC is indistinguishable from the null. Under Type B, which preserves "
    "short-term serial correlation, the conclusion is unchanged. For the TopK1 Sharpe "
    "comparison, the observed Sharpe (−0.16) falls at the 25.8th percentile of the "
    "null distribution (permutation p = 0.742), confirming the strategy performs no "
    "differently from random stock selection. The permutation p-value reported in Table 4 "
    "(0.742) refers to the Sharpe-based test; the IC-level permutation p-values are "
    "0.599 (Type A) and 0.601 (Type B). Figure 6 illustrates both null distributions."
)
set_para_text(p293, CLEAN_293)
print(f"  New (first 120): {para_text(p293)[:120]}")

# ════════════════════════════════════════════════════════════════════════════
# FIX 4: Para 295 — Figure 6 caption with corrected p-values + IC note
# ════════════════════════════════════════════════════════════════════════════
print("\n[FIX 4] Para 295 — Figure 6 caption")
p295 = paras[295]
old_295 = para_text(p295)
print(f"  Old (first 160): {old_295[:160]}")

CLEAN_295 = (
    "Figure 6. Permutation null distributions for the IC-level test. "
    "Type A (IID bootstrap, left): p = 0.599. "
    "Type B (block bootstrap, block = 5 days, right): p = 0.601. "
    "Both tests confirm the gate-closed decision. "
    "The observed mean IC lies at the centre of the null distribution under both permutation schemes. "
    "The permutation test is applied to the daily IC series derived from prediction outputs "
    "(mean = −0.00127); the canonical mean IC (−0.0005) is computed via the ICGDF HAC "
    "procedure in Section 4.4. Both series confirm gate-closed."
)
set_para_text(p295, CLEAN_295)
print(f"  New (first 160): {para_text(p295)[:160]}")

# ════════════════════════════════════════════════════════════════════════════
# FIX 5: Para 230 — Table footnote p-value references
# ════════════════════════════════════════════════════════════════════════════
print("\n[FIX 5] Para 230 — Table footnote (perm p-values)")
p230 = paras[230]
full_230 = para_text(p230)
if "0.797" in full_230 or "0.754" in full_230:
    new_230 = (full_230
               .replace("0.797 (Type A temporal)", "0.599 (Type A IID bootstrap)")
               .replace("0.754 (Type B block)", "0.601 (Type B block bootstrap)")
               .replace("0.797", "0.599")
               .replace("0.754", "0.601"))
    set_para_text(p230, new_230)
    print(f"  Para 230 updated (0.797/0.754 → 0.599/0.601)")
else:
    print(f"  Para 230: no 0.797/0.754 found — first 100: {full_230[:100]}")

# ════════════════════════════════════════════════════════════════════════════
# FIX 6: Section 5.3 intro (para 293 is the body; also add sentence to
#         any remaining para in Section 5.3 if needed)
#         Already covered by FIX 3 + FIX 4 (Fig 6 caption).
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# FIX 7: Add Sharpe (OOS) column to Table 9 (IC comparison)
# ════════════════════════════════════════════════════════════════════════════
print("\n[FIX 7] Table 9 — Add Sharpe (OOS) column")
tbl9  = tbls[9]
rows9 = tbl9.findall("w:tr", ns)
headers9 = [para_text(c.find("w:p", ns) or c).strip()
            for c in rows9[0].findall("w:tc", ns)]
print(f"  Current headers ({len(headers9)} cols): {headers9}")

if "Sharpe (OOS)" not in headers9:
    new_vals_9 = ["Sharpe (OOS)", "0.57", "−0.16"]   # Momentum, ML Ensemble

    for ri, (row, val) in enumerate(zip(rows9, new_vals_9)):
        existing_cells = row.findall("w:tc", ns)
        # Clone last cell as template
        template = copy.deepcopy(existing_cells[-1])

        # Update cell width
        tcp = template.find("w:tcPr", ns)
        if tcp is not None:
            tcw = tcp.find("w:tcW", ns)
            if tcw is not None:
                tcw.set(wtag("w"), "1247")
                tcw.set(wtag("type"), "dxa")

        # Remove existing paragraph elements
        for p_el in list(template.findall("w:p", ns)):
            template.remove(p_el)

        # Build new paragraph + run
        p_el  = ET.SubElement(template, wtag("p"))
        p_el.set(f"{{{W14}}}paraId", f"{random.randint(0x10000000, 0x7FFFFFFF):08X}")
        p_el.set(f"{{{W14}}}textId", "77777777")
        r_el  = ET.SubElement(p_el, wtag("r"))
        rpr   = ET.SubElement(r_el, wtag("rPr"))

        fonts = ET.SubElement(rpr, wtag("rFonts"))
        fonts.set(wtag("ascii"),  "Times New Roman")
        fonts.set(wtag("hAnsi"), "Times New Roman")

        if ri == 0:          # header row: bold + white text
            ET.SubElement(rpr, wtag("b"))
            color_el = ET.SubElement(rpr, wtag("color"))
            color_el.set(wtag("val"), "FFFFFF")
        sz = ET.SubElement(rpr, wtag("sz"))
        sz.set(wtag("val"), "18")

        t_el = ET.SubElement(r_el, wtag("t"))
        t_el.text = val
        row.append(template)

    new_headers = [para_text(c.find("w:p", ns) or c).strip()
                   for c in rows9[0].findall("w:tc", ns)]
    print(f"  New headers ({len(new_headers)} cols): {new_headers}")
    for ri, row in enumerate(rows9[1:], 1):
        last = row.findall("w:tc", ns)[-1]
        print(f"  Row {ri} Sharpe(OOS): {para_text(last.find('w:p', ns) or last)}")
else:
    print("  Sharpe (OOS) already present — skipping")

# ════════════════════════════════════════════════════════════════════════════
# FIX 8: Re-embed all 16 figures (300 DPI from figures/ directory)
# ════════════════════════════════════════════════════════════════════════════
print("\n[FIX 8] Re-embedding 16 figures (300 DPI)")
fig_map = {
    "word/media/image1.png":  "fig01_hac_bandwidth.png",
    "word/media/image2.png":  "fig02_power_analysis.png",
    "word/media/image3.png":  "fig03_fold_level_ic.png",
    "word/media/image4.png":  "fig04_strategy_performance.png",
    "word/media/image5.png":  "fig05_sharpe_vs_k.png",
    "word/media/image6.png":  "fig06_permutation_ic.png",
    "word/media/image7.png":  "fig07_permutation_sharpe.png",
    "word/media/image8.png":  "fig08_subperiod_heatmap.png",
    "word/media/image9.png":  "fig09_tc_sensitivity.png",
    "word/media/image10.png": "fig10_factor_regression.png",
    "word/media/image11.png": "fig11_universe_robustness.png",
    "word/media/image12.png": "fig12_shap_importance.png",
    "word/media/image13.png": "fig13_dm_test.png",
    "word/media/image14.png": "fig14_vix_conditioned.png",
    "word/media/image15.png": "fig15_ic_gate_summary.png",
    "word/media/image16.png": "figA1_pipeline.png",
}
imgs_ok = 0
for docx_key, fig_file in fig_map.items():
    fig_path = os.path.join(FIG_DIR, fig_file)
    if os.path.exists(fig_path):
        with open(fig_path, "rb") as fh:
            files[docx_key] = fh.read()
        kb = os.path.getsize(fig_path) // 1024
        print(f"  {docx_key} ← {fig_file} ({kb} KB)")
        imgs_ok += 1
    else:
        print(f"  MISSING: {fig_path}")
print(f"  Embedded {imgs_ok}/16 figures")

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════
files["word/document.xml"] = ET.tostring(root, xml_declaration=True,
                                          encoding="UTF-8", standalone=True)
buf = io.BytesIO()
with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as zin:
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name in zin.namelist():
            zout.writestr(name, files.get(name, zin.read(name)))

os.makedirs("paper", exist_ok=True)
with open(DOCX_OUT, "wb") as f:
    f.write(buf.getvalue())

size_kb = os.path.getsize(DOCX_OUT) // 1024
print(f"\n✓ Saved: {DOCX_OUT} ({size_kb} KB)")
