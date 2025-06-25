# Supervised & SSL Template

A minimal, Docker-first PyTorch skeleton for computer-vision **and** NLP projects.

---

## 🚀 Quick start

1. **Clone** the repo (or [fork](https://github.com/) first):
   ```bash
   git clone <your-fork-url> my_project && cd my_project
   ```
2. **Launch an experiment** (single-GPU):
   ```bash
   docker compose up train
   ```
   The first run builds the CUDA 11.8 container and downloads CIFAR-10.
3. **Evaluate the best checkpoint**:
   ```bash
   docker compose up eval
   ```
4. **Hyper-parameter search** with [Ray Tune](https://docs.ray.io/):
   ```bash
   docker compose up search
   ```

All outputs and logs live in `outputs/<date>/<time>` thanks to [Hydra](https://hydra.cc/).

---

## 🗂️ Repo layout

```
.
├── docker/               # CUDA 11.8 runtime image
│   └── Dockerfile
├── docker-compose.yml    # train / eval / search services
├── configs/              # Hydra YAMLs
├── src/                  # Python packages
│   ├── data/             # data-loading modules
│   ├── models/           # neural nets
│   └── utils/            # logging, metrics, ckpt I/O
├── requirements.txt      # pip packages (GPU wheels)
└── README.md             # this file
```

> **Tip**: run `grep -R --line-number "TODO" src/` to find extension points. 📌

---

### Why this template?

* **Single command** per task – `train`, `eval`, `search`.
* **100 % reproducible** in Docker; cloud VMs are one-liner spins 🟢.
* **Plain PyTorch** loop you can hack; migrate to Lightning later if you wish.
* **Config-driven** – change models/datasets from YAML without touching Python.
* **Ready for Ray Tune** to scale sweeps beyond one GPU.

Happy experimenting! 🎉 