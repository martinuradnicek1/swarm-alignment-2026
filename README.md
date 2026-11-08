# Choosing a cooperative rule is not enough

This project studies how shared-rule selection, discussion and retained interaction context affect cooperation and adherence to individual objectives in two- and three-agent teams. It was developed for Track 2, *The Science of Agent Networks*, at the [Apart Swarm Alignment Hackathon 2026](https://apartresearch.com/sprints/swarm-alignment-2026).

The research report accompanies the hackathon submission separately; it is not included in this repository. A public report link will be added when available.

## Reproduce the results

From the repository root, create an environment and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Reproduction was tested with Python 3.9.

The command verifies 6,303 decisions against stored input and answer fingerprints, independently recalculates payoffs, checks six numerical result files and regenerates three figures. Outputs are written to `reproduced/`, including `verification.json`; the released data and results are left unchanged. No API keys or model calls are needed.

To skip figure generation, use `python reproduce.py --no-figures`. To choose an output directory, add `--output path/to/output`.

## Repository contents

| Path | Contents |
| --- | --- |
| `games.py` | Game rules, mediator selection and payoff calculations. |
| `prompts_base.py`, `prompts.py` | Prompt construction and reconstruction of recorded visible inputs. |
| `analysis.py`, `reproduce.py` | Statistical analyses and the reproduction entry point. |
| `verify.py` | Fingerprint, payoff and numerical-result checks. |
| `figures.py` | Figure generation. |
| `data/` | Cleaned decisions, shared messages and fingerprint records. |
| `results/`, `figures/` | Reference numerical results and figures. |
| `experiment.json` | Experimental settings, scope and provenance. |
| `checksums.json` | SHA-256 hashes of the other released files. |

## Scope and prior work

This package provides **offline reproduction from released decisions**. It does not include live experiment collection code or raw provider logs. The checks establish consistency of the released records and analyses; they do not independently authenticate the original provider calls.

The games and mechanism families build on [CoopEval](https://arxiv.org/html/2604.15267). The experiment implementation and interventions were developed during the sprint; standalone packaging and verification were added after data collection. The experiments use one Gemma agent with one or two Sol peers and do not establish behavior at larger swarm scales.
