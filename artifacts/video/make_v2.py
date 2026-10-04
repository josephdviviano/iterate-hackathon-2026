import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

W, H = 19.2, 10.8
BLUE, ORANGE, GREEN, GREY = "#1E6FD9", "#E8871E", "#2E9E5B", "#555555"

def frame(title, sub):
    fig = plt.figure(figsize=(W, H), dpi=100); fig.patch.set_facecolor("white")
    fig.text(0.04, 0.91, title, fontsize=40, fontweight="bold", color="#111")
    fig.text(0.04, 0.86, sub, fontsize=20, color=GREY)
    return fig

def bullets(fig, items, x=0.04, y=0.74, w=None):
    for head, body in items:
        fig.text(x, y, head, fontsize=26, fontweight="bold", color=BLUE)
        fig.text(x, y - 0.05, body, fontsize=22, color="#222")
        y -= 0.15

# slide 1: workflow
fig = frame("An agent that knows when it is wrong", "Track 2.3: flag what you do not know, test there, accept refutation")
ax = fig.add_axes([0.02, 0.05, 0.96, 0.76]); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
def box(x, y, w, h, text, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", fc=c + "22", ec=c, lw=2.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=19, color="#111")
def arrow(a, b, c="#333", rad=0.0, label=None, lx=0, ly=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=28, lw=2.5, color=c, connectionstyle=f"arc3,rad={rad}"))
    if label: ax.text(lx, ly, label, fontsize=17, color=c, ha="center", fontweight="bold")
box(2, 38, 17, 12, "Observations\n(game frames)", GREY)
box(25, 38, 19, 12, "Devin writes\nK programs", BLUE)
box(50, 38, 19, 12, "Admit only if\nexact replay", BLUE)
box(75, 38, 22, 12, "Committee\npredicts next frame", BLUE)
box(75, 8, 22, 12, "Agree:\ncommit", GREEN)
box(45, 8, 22, 12, "Disagree:\nflag + probe there", ORANGE)
box(13, 8, 24, 12, "Drop refuted programs,\nrewrite on new data", ORANGE)
arrow((19.6, 44), (24.4, 44)); arrow((44.6, 44), (49.4, 44)); arrow((69.6, 44), (74.4, 44))
arrow((86, 37.4), (86, 20.6), GREEN); arrow((80, 37.4), (60, 20.6), ORANGE)
arrow((44.4, 14), (37.6, 14), ORANGE); arrow((25, 20.6), (31, 37.4), ORANGE, label="falsify", lx=22, ly=29)
ax.text(56, 29, "uncertainty =\nvote entropy", fontsize=17, color=ORANGE, ha="center")
ax.text(86, 3, "conformal set: 90% coverage, or abstain", fontsize=16, color=GREEN, ha="center")
fig.savefig("s1.png", dpi=100, facecolor="white"); plt.close(fig)

# slide 2: what it looks like
fig = frame("What agreement and disagreement look like", "Real frames, game sk48: before, observed, and each distinct prediction")
bullets(fig, [("Split", "the vote is right; the flag fires"), ("Unanimous", "right, the common case"), ("Unanimous and wrong", "a shared blind spot")], y=0.70)
ax = fig.add_axes([0.36, 0.04, 0.62, 0.80]); ax.imshow(mpimg.imread("../figures/examples_sk48.png")); ax.axis("off")
fig.savefig("s2.png", dpi=100, facecolor="white"); plt.close(fig)

# slide 3: results
fig = frame("Results", "7 ARC-AGI-3 levels, OPINE-World's program contract; every number in RESULTS.md")
bullets(fig, [("Flags its errors", "unanimous 5% wrong, split 34% (R34)"),
              ("Calibrated", "95% coverage at a 90% target (R35)"),
              ("Scales with K", "AUROC 0.68 to 0.92 (R37)"),
              ("Repairs itself", "live play 0.84 to 1.00 after probing (R36)")], y=0.72)
ax = fig.add_axes([0.47, 0.12, 0.52, 0.62]); ax.imshow(mpimg.imread("../figures/ksweep.png")); ax.axis("off")
fig.savefig("s3.png", dpi=100, facecolor="white"); plt.close(fig)
