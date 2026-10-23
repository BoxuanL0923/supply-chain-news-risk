"""Render GitHub-safe PNG workflow diagrams with local, synthetic labels only."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
FONT = "/System/Library/Fonts/Supplemental/Verdana.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Verdana Bold.ttf"
NAVY = "#14213d"
GRAY = "#506075"
PALETTE = {
    "data": ("#ffffff", "#b9c3cf"),
    "tool": ("#ffffff", "#b9c3cf"),
    "forecast": ("#ffffff", "#b9c3cf"),
    "opt": ("#ffffff", "#b9c3cf"),
    "skill": ("#ffffff", "#b9c3cf"),
    "graph": ("#eeeaff", "#6447a5"),
    "supply": ("#e9f5e8", "#327d51"),
    "gate": ("#e9f5e8", "#327d51"),
}
ACCENTS = {"graph", "supply", "gate"}


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def canvas(width, height, title, subtitle):
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.text((58, 33), title, font=font(38, True), fill=NAVY)
    draw.text((60, 85), subtitle, font=font(21), fill=GRAY)
    return image, draw


def box(draw, xy, title, lines, kind, *, tag=None, emphasis=()):
    fill, stroke = PALETTE[kind]
    draw.rounded_rectangle(xy, radius=20, fill=fill, outline=stroke, width=3)
    x0, y0, x1, _ = xy
    if tag:
        draw.text((x0 + 21, y0 + 15), tag.upper(), font=font(16, kind in ACCENTS),
                  fill=stroke if kind in ACCENTS else GRAY)
        title_y = y0 + 43
    else:
        title_y = y0 + 22
    draw.text((x0 + 21, title_y), title,
              font=font(23 if kind in ACCENTS else 22, kind in ACCENTS),
              fill=stroke if kind in ACCENTS else NAVY)
    for idx, line in enumerate(lines):
        active = idx in emphasis
        draw.text((x0 + 21, title_y + 35 + idx * 27), line,
                  font=font(18, active), fill=stroke if active else NAVY)


def arrow(draw, points, *, dashed=False, color="#53657d", width=5):
    for start, end in zip(points, points[1:]):
        if dashed:
            import math
            distance = math.dist(start, end)
            count = max(1, int(distance // 23))
            for i in range(0, count, 2):
                a = i / count
                b = min(i + 1, count) / count
                draw.line(((start[0] + (end[0] - start[0]) * a,
                            start[1] + (end[1] - start[1]) * a),
                           (start[0] + (end[0] - start[0]) * b,
                            start[1] + (end[1] - start[1]) * b)), fill=color, width=width)
        else:
            draw.line((start, end), fill=color, width=width, joint="curve")
    import math
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    angle = math.atan2(y1 - y0, x1 - x0)
    size = 15
    wing = 0.55
    draw.polygon([(x1, y1),
                  (x1 - size * math.cos(angle - wing), y1 - size * math.sin(angle - wing)),
                  (x1 - size * math.cos(angle + wing), y1 - size * math.sin(angle + wing))],
                 fill=color)


def senior():
    im, d = canvas(3250, 860, "Graph structure to prompt: technical workflow",
                   "Left to right: dated graph -> native tools -> inspected paths -> typed selector input -> Skill + Python gate -> risk evaluation")
    panels = [
        ((60, 175, 540, 660), "Dated graph + pool",
         ["graph.py: supplier -> customer", "Edge vintage + entity mapping", "data.prepare: historical hops <= 2", "Graph.paths: hop 3 needs via", "Outcome labels remain sealed"], "supply", "GRAPH DATA"),
        ((590, 175, 1070, 660), "Native graph tools",
         ["toolkit.exposure_paths(hops, via)", "search_events -> inspect_event", "Inspect exact path prefixes", "Tool returns roles, IDs, vintage", "Co-mention is retrieval only"], "graph", "RETRIEVAL AGENT"),
        ((1120, 175, 1600, 660), "Source-bound path join",
         ["workflow.validate_evidence", "Join inspected news to endpoint", "Attach eligible verified_paths", "Cutoff and source receipts checked", "Product continuity still unknown"], "supply", "HOST VALIDATION"),
        ((1650, 175, 2130, 660), "Selector graph projection",
         ["domain_signals.finance_graph_signal", "Roles + h1/h2/h3 path motif", "Finance hypothesis + evidence gap", "channel_status = path_only", "No raw edge ID or quote to ranker"], "graph", "PYTHON PROJECTION"),
        ((2180, 175, 2660, 660), "DSPy + Skill + Python",
         ["ReAct calls local Skill-read tool", "Then calls Python select_context", "Tool receipt: C1,C2... aliases", "Host checks order, time + budget", "GEPA tunes tool-using prompt"], "tool", "SELECTOR"),
        ((2710, 175, 3190, 660), "Forecast + GEPA",
         ["Runner.score uses selected evidence", "Five-session risk vs OLS/GARCH", "GEPA: train feedback, valid choice", "Holdout: OWN, H1, H2, H3", "Nested H2-H1 / H3-H2 regression"], "forecast", "ESTIMAND"),
    ]
    for xy, title, lines, kind, tag in panels:
        box(d, xy, title, lines, kind, tag=tag,
            emphasis=(0, 1, 2) if kind in ACCENTS else ())
    for left, right in zip(panels, panels[1:]):
        arrow(d, [(left[0][2], 410), (right[0][0], 410)], width=6)
    d.text((60, 732), "Selector receives allowlisted bins; native retrieval and the scorer may receive source text and raw path fields.",
           font=font(23, True), fill=NAVY)
    d.text((60, 775), "The completed 50-firm pool ends at hop 2. Hop-3 tool and ranker contracts have synthetic tests; empirical retrieval remains to run.",
           font=font(22), fill=GRAY)
    im.save(OUT / "senior_workflow.png", optimize=True)


def academic():
    im, d = canvas(2980, 770, "Academic workflow: graph information to volatility risk",
                   "A dated network and a news event yield a path hypothesis; evidence and forecast tests determine its predictive value")
    panels = [
        ((50, 170, 570, 585), "Define G_t and news",
         ["Directed supplier -> customer edges", "Dated mapping and graph vintage", "News availability before origin t", "Historical pool: paths h <= 2", "Hop 3 requires a new candidate run"], "supply", "OBSERVABLES"),
        ((640, 170, 1160, 585), "Construct path motifs",
         ["Join news endpoint to focal firm", "Ordered roles (r1, r2, r3); h <= 3", "Inspect each prefix before hop 3", "Co-mention != supply edge", "Shortest path alone is not a channel"], "graph", "GRAPH THEORY"),
        ((1230, 170, 1750, 585), "Classify evidence tier",
         ["Supplier -> input risk hypothesis", "Customer -> demand hypothesis", "h2/h3: product continuity gap", "Direct receipt exists; h > 1 pending", "No transitive revenue-% weights"], "gate", "FINANCE MECHANISM"),
        ((1820, 170, 2340, 585), "Build selector input",
         ["Python: motif, risk + gap bins", "DSPy calls local Skill-read tool", "Then calls Python select_context", "Host checks order, time + budget", "Selected evidence enters fixed scorer"], "tool", "PROMPT + TOOL"),
        ((2410, 170, 2930, 585), "Estimate contribution",
         ["y = log(mean next 5 daily r^2)", "Residual to OLS/GARCH anchor", "Same cases: 1, <=2, <=3 hops", "Nested H2-H1 / H3-H2 regression", "Sealed test; compute may differ"], "forecast", "EMPIRICAL TEST"),
    ]
    for xy, title, lines, kind, tag in panels:
        box(d, xy, title, lines, kind, tag=tag,
            emphasis=(0, 1, 2) if kind in ACCENTS else ())
    for left, right in zip(panels, panels[1:]):
        arrow(d, [(left[0][2], 380), (right[0][0], 380)], width=6)
    d.text((50, 659), "A hop-3 event can change the five-session risk forecast only if its path-specific evidence adds held-out predictive value.",
           font=font(22, True), fill=NAVY)
    d.text((50, 702), "The existing 50-firm run cannot test that claim: its candidate pool stopped at two hops; channel receipts are direct-only.",
           font=font(21), fill=GRAY)
    im.save(OUT / "academic_workflow.png", optimize=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    senior()
    academic()
    for path in (OUT / "senior_workflow.png", OUT / "academic_workflow.png"):
        with Image.open(path) as image:
            image.verify()
        print(f"{path.relative_to(ROOT)}: valid PNG")
