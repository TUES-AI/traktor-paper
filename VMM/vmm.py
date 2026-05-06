"""
Vision Memory Model (VMM)
Novelty detection from laptop camera at 1 fps.

Primary signal  : cosine distance to nearest neighbour in an embedding memory bank.
                  "Have I seen anything like this before?"
Secondary signal: RND prediction error (slow LR, updated every N steps).
Final novelty   : weighted combination, normalised with Welford running stats.

Architecture:
  Encoder  : MobileNetV3-Small (frozen) 
  → Linear(576, 128) → L2-norm
  Memory   : rolling bank of K diverse embeddings, add only when novel
  Target   : fixed random MLP 128→256→256→128
  Predictor: channel-attention gate + MLP, LR=1e-5, updated every 5 steps
"""

import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_DIM       = 128
HIDDEN_DIM      = 256
MEMORY_SIZE     = 500       # max embeddings stored
MEMORY_KNOWN_DIST = 0.08    # below this, update an existing visual-place cluster
MEMORY_UPDATE_RATE = 0.05   # centroid adaptation rate for familiar views
NOVEL_DIST_THR  = 0.12      # cosine distance → "novel"  (pure memory decision)
MEMORY_NORM_DIST = 0.18     # distance that saturates memory novelty at 1.0
RND_WEIGHT      = 0.65
MEMORY_WEIGHT   = 0.35
RND_LR          = 3e-4      # train aggressively — novelty flag controls when, not LR
RND_UPDATE_FREQ = 1         # update every step
WARMUP_STEPS    = 5
CAPTURE_FPS     = 1
DEVICE          = torch.device("cpu")

TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((112, 112)),   # low-res rover camera, faster on Pi CPU
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]),
])


# ── Encoder ───────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone  = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.proj     = nn.Linear(576, EMBED_DIM)
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


# ── RND networks ─────────────────────────────────────────────────────────────
class TargetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, EMBED_DIM),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, z): return self.net(z)


class PredictorNet(nn.Module):
    """Channel-attention gate: softmax(Wz) * z focuses on discriminative dims."""
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.net  = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, EMBED_DIM),
        )

    def forward(self, z):
        return self.net(torch.softmax(self.gate(z), dim=-1) * z)


# ── Welford running normaliser ────────────────────────────────────────────────
class RunningNorm:
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0

    def update(self, x):
        self.n += 1
        d = x - self.mean; self.mean += d / self.n
        self.M2 += d * (x - self.mean)

    @property
    def std(self): return np.sqrt(self.M2 / max(self.n - 1, 1))


# ── Memory bank ───────────────────────────────────────────────────────────────
class MemoryBank:
    """Stores visual-place cluster centroids instead of individual frames."""
    def __init__(self, maxlen=MEMORY_SIZE):
        self.bank   = []
        self.counts = []
        self.last_seen = []
        self.maxlen = maxlen

    def query(self, z: torch.Tensor):
        """Return cosine distance to nearest neighbour (0=identical, up to 2)."""
        if not self.bank:
            return 1.0, None
        bank_t = torch.stack(self.bank)          # (N, D)
        sims   = (bank_t @ z.T).squeeze(-1)      # cosine sim (already L2-normed)
        best = int(sims.argmax().item())
        return float(1.0 - sims[best].item()), best

    def update(self, z: torch.Tensor, dist: float, cluster_idx, step: int):
        z = z.detach().squeeze(0)
        if cluster_idx is not None and dist < MEMORY_KNOWN_DIST:
            eta = min(MEMORY_UPDATE_RATE, 1.0 / (self.counts[cluster_idx] + 1))
            centroid = F.normalize((1.0 - eta) * self.bank[cluster_idx] + eta * z, dim=0)
            self.bank[cluster_idx] = centroid.detach()
            self.counts[cluster_idx] += 1
            self.last_seen[cluster_idx] = step
            return cluster_idx, False
        if len(self.bank) >= self.maxlen:
            evict = int(np.argmin(self.last_seen))
            self.bank.pop(evict)
            self.counts.pop(evict)
            self.last_seen.pop(evict)
        self.bank.append(z.detach())
        self.counts.append(1)
        self.last_seen.append(step)
        return len(self.bank) - 1, True


# ── VMM ───────────────────────────────────────────────────────────────────────
class VMM:
    def __init__(self):
        self.encoder   = Encoder().to(DEVICE).eval()
        self.target    = TargetNet().to(DEVICE).eval()
        self.predictor = PredictorNet().to(DEVICE)
        self.opt       = torch.optim.Adam(self.predictor.parameters(), lr=RND_LR)
        self.memory    = MemoryBank()
        self.rnd_norm  = RunningNorm()
        self.nov_norm  = RunningNorm()
        self.step      = 0
        self.history   = []

    @torch.no_grad()
    def _embed(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t   = TRANSFORM(rgb).unsqueeze(0).to(DEVICE)
        return self.encoder(t)   # (1, 128)

    def observe(self, frame_bgr):
        z = self._embed(frame_bgr)

        # ── Memory signal (visual-place cluster distance) ─────────
        mem_dist, cluster_idx = self.memory.query(z)
        if self.step >= WARMUP_STEPS:
            cluster_idx, created_cluster = self.memory.update(z, mem_dist, cluster_idx, self.step)
        else:
            created_cluster = False

        # ── RND (trains always, reported separately) ──────────────
        with torch.no_grad():
            t_out = self.target(z)
        p_out   = self.predictor(z)
        rnd_raw = F.mse_loss(p_out, t_out).item()

        loss = F.mse_loss(p_out, t_out.detach())
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        self.rnd_norm.update(rnd_raw)
        rnd_norm = float(np.clip(rnd_raw / (self.rnd_norm.mean + 1e-8), 0.0, 1.0))
        mem_norm = float(np.clip(mem_dist / MEMORY_NORM_DIST, 0.0, 1.0))
        combined = float(np.clip(RND_WEIGHT * rnd_norm + MEMORY_WEIGHT * mem_norm, 0.0, 1.0))

        # ── Decision: combined learned + explicit memory novelty ────────────
        is_novel = (combined > 0.5) and (self.step >= WARMUP_STEPS)

        self.nov_norm.update(mem_dist)
        self.step += 1
        self.history.append(mem_dist)

        return {
            "novelty":   combined,
            "mem_dist":  mem_dist,
            "mem_norm":  mem_norm,
            "rnd_norm":  rnd_norm,
            "cluster_id": cluster_idx,
            "new_cluster": created_cluster,
            "is_novel":  is_novel,
            "bank_size": len(self.memory.bank),
            "step":      self.step,
            "mean":      self.nov_norm.mean,
            "std":       self.nov_norm.std,
        }


# ── Overlay ───────────────────────────────────────────────────────────────────
def draw_overlay(frame, r, history):
    h, w  = frame.shape[:2]
    vis   = frame.copy()
    color = (0, 0, 220) if r["is_novel"] else (0, 190, 60)
    label = "NOVEL" if r["is_novel"] else "FAMILIAR"

    cv2.rectangle(vis, (0, 0), (w-1, h-1), color, 12)
    cv2.putText(vis, label, (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 2)

    # Memory distance bar (primary signal)
    bx, by, bw, bh = 20, 68, 300, 20
    fill = int(np.clip(r["mem_dist"] / 1.0, 0, 1) * bw)
    cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (50,50,50), -1)
    cv2.rectangle(vis, (bx, by), (bx+fill, by+bh), color, -1)
    cv2.line(vis, (bx + int(NOVEL_DIST_THR * bw), by),
                  (bx + int(NOVEL_DIST_THR * bw), by+bh), (255,255,0), 2)
    cv2.putText(vis, f"mem dist {r['mem_dist']:.3f}  (thr {NOVEL_DIST_THR})",
                (bx, by+bh+16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210,210,210), 1)

    stats = [
        f"step      : {r['step']}",
        f"bank size : {r['bank_size']} / {MEMORY_SIZE}",
        f"rnd ratio : {r['rnd_norm']:.3f}",
        f"mem norm  : {r.get('mem_norm', 0.0):.3f}",
        f"combined  : {r['novelty']:.3f}",
    ]
    for i, s in enumerate(stats):
        cv2.putText(vis, s, (20, 120 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # Sparkline
    if len(history) > 1:
        ph, pw = 70, 300
        px, py = 20, h - ph - 20
        cv2.rectangle(vis, (px, py), (px+pw, py+ph), (40,40,40), -1)
        lo, hi = min(history), max(history)
        rng    = max(hi - lo, 1e-8)
        pts    = []
        for i, v in enumerate(history[-pw:]):
            x = px + int(i * pw / max(len(history[-pw:])-1, 1))
            y = py + ph - int((v - lo) / rng * ph)
            pts.append((x, y))
        for i in range(1, len(pts)):
            cv2.line(vis, pts[i-1], pts[i], (80,200,255), 1)
        # threshold line
        if hi > lo:
            ty = py + ph - int((NOVEL_DIST_THR - lo) / rng * ph)
            cv2.line(vis, (px, np.clip(ty, py, py+ph)),
                          (px+pw, np.clip(ty, py, py+ph)), (255,255,0), 1)
        cv2.putText(vis, "novelty history  (yellow=threshold)", (px, py-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,150,150), 1)

    return vis


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading VMM...")
    vmm    = VMM()
    result = {"novelty":0,"mem_dist":0,"rnd_norm":0,"is_novel":False,
              "bank_size":0,"step":0,"mean":0,"std":0}
    print("Done. Press Q to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    interval = 1.0 / CAPTURE_FPS
    last_t   = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - last_t >= interval:
            last_t = now
            result = vmm.observe(frame)
            tag    = "NOVEL !" if result["is_novel"] else "familiar"
            print(f"step {result['step']:4d} | "
                  f"mem_dist {result['mem_dist']:.3f} | "
                  f"rnd {result['rnd_norm']:.3f} | "
                  f"bank {result['bank_size']:3d} | {tag}")

        vis = draw_overlay(frame, result, vmm.history)
        cv2.imshow("VMM — Vision Memory Model", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if vmm.history:
        novel = sum(1 for v in vmm.history if v > NOVEL_DIST_THR)
        print(f"\n── Summary ──────────────────────")
        print(f"Total steps : {vmm.step}")
        print(f"Novel steps : {novel}  ({100*novel/vmm.step:.1f}%)")
        print(f"Bank size   : {len(vmm.memory.bank)}")


if __name__ == "__main__":
    main()
