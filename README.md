# Sidon Record Search & Certification Engine

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Certified](https://img.shields.io/badge/C≤0.98173-certified-blue)
![Python](https://img.shields.io/badge/python-3.11+-yellow)

A high-performance research infrastructure for discovering and certifying new world records for the Sidon constant $C$ in

$$h(N) \le N^{1/2} + C N^{1/4} + O(1).$$

**Latest certified result**: Record bundle `records/certified/record_1771350836_46537e4f6f8d/` (2026-02-17) proves $C \le \mathbf{0.98173}$ with 19,736 dual certificates.

| Tier | Tooling | Output |
| --- | --- | --- |
| Discovery | `search.py` + adaptive gradients | Logs in `records/top_k.json`, ledger entries |
| Candidate clean | `rounding.py` | Six-decimal parameters ready for verification |
| Certified | `paranoid.py`, `certify.py`, `cert_check.py` | Full LP witness bundle |

Quick links: [Methodology](docs/methodology.md) · [TeX submission](Attacks/MO_problems/opus/30.tex) · [Certified data](records/certified/record_1771350836_46537e4f6f8d/)

## 🚀 Quick Start

### 1. Start the Hunt
Search for new candidate parameters $(\tau, \alpha, \mathbf{c})$ using parallel coordinate descent.
```bash
python3 search.py
```

### 2. Verify Stability
Stress-test a candidate record by running it multiple times in fresh processes.
```bash
python3 paranoid.py --input best_found.json
```

### 3. Generate Formal Certificates
Run an exhaustive audit of rounded parameters and export dual LP certificates for every cell intersection.
```bash
python3 certify.py --input best_found.json
```

### 4. Audit Certificates
Verify the mathematical validity of the exported certificates using high-precision dual reconstruction.
```bash
python3 cert_check.py --bundle records/certified/record_<id>/
```

---

## 🏛️ System Architecture

The system uses a **three-tier reliability model** to ensure publication-grade results.

### Tier 1: Discovery (`search.py`)
*   **Fast & Parallel:** Uses `ProcessPoolExecutor` to evaluate hundreds of candidates per hour.
*   **Optimized Verification:** Employs graph-theoretic Bellman-Ford filtering to skip redundant calculations.
*   **Automated Ledger:** Every improvement is logged in `records/ledger.jsonl` with full environment metadata.

### Tier 2: Candidate Clean (`rounding.py`)
*   Automatically transforms high-precision "discovery" floats into human-readable "clean" decimals (6 places).
*   Ensures that rounding doesn't accidentally break the record before proceeding to certification.

### Tier 3: Certified (`certify.py` + `cert_check.py`)
*   **LP Dual Certificates:** Exports the mathematical "witness" for all 24,822 cell intersections.
*   **Rational Reconstruction:** Uses `Decimal` arithmetic to prove the bound holds strictly, moving beyond solver tolerances.

---

## 📂 Data & Records

*   **`best_found.json`**: Current champion parameters used as the baseline for the search.
*   **`records/ledger.jsonl`**: The unalterable audit trail of all improvements found.
*   **`records/top_k.json`**: Summary of the best records and candidates.
*   **`records/certified/`**: Formal certification bundles containing thousands of LP certificates. The newest bundle `record_1771350836_46537e4f6f8d/` certifies $C \le 0.98172529686$ with 19,736 LP certificates and dual reconstructions.

## 📄 Research Documentation

*   **`research_paper.md`**: Human-readable markdown version of the findings.
*   **`research_paper.tex`**: Publication-ready LaTeX manuscript.

---

## 🛠️ Configuration

Adjust search behavior, thresholds, and rounding in `config.py`:
*   `AUTO_CERTIFY_THRESHOLD`: Minimum gap to CHO record to trigger candidate status.
*   `DEFAULT_ROUNDING`: Decimal precision for certified records.
*   `SAFETY_BUFFER`: Numerical slack for rigorous claims.
