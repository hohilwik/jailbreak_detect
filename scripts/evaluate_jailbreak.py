"""
Evaluate the trained jailbreak detector on edu data and known jailbreak data.
- False Positive Rate (FPR) = (edu samples predicted as jailbreak) / (total educational samples)
- False Negative Rate (FNR) = (jailbreak samples predicted as safe) / (total jailbreak samples)
"""

import os
import json
import pickle
import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

# config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# env vars / defaults
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
# where the model files live
SVM_DIR = ARTIFACTS_DIR / "atk_svm_models"
JAILBREAK_DIR = ARTIFACTS_DIR / "jailbreak"
EMBEDDINGS_DATA_DIR = Path(os.getenv("EMBEDDINGS_DATA_DIR", "local_data_mdbr"))
INDEX_PATH = ARTIFACTS_DIR / "topic_index.json"

# all helper functions

def load_topic_index() -> dict:
    # load the topic index
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Topic index not found at {INDEX_PATH}. Run original build first.")
    with open(INDEX_PATH, "r") as f:
        return json.load(f)

def get_topic_embeddings_path(topic_id: str) -> Path:
    # return path where the embeddings for this topic are stored
    return EMBEDDINGS_DATA_DIR / topic_id / "questions_embeddings.npy"

def load_full_question_embeddings(topic_id: str) -> np.ndarray:
    # load the embeddings array for an edu topic
    emb_file = get_topic_embeddings_path(topic_id)
    if not emb_file.exists():
        raise FileNotFoundError(f"Embeddings not found at {emb_file}.")
    return np.load(emb_file)

# JailbreakDetector class simplified for prediction

class JailbreakDetector:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.attack_models = self._load_models("attack_class")
        self.bypass_models = self._load_models("bypass_technique")

    def _load_models(self, category: str) -> dict:
        models = {}
        pattern = f"jailbreak_{category}_*.pkl"
        for path in SVM_DIR.glob(pattern):
            stem = path.stem
            prefix = f"jailbreak_{category}_"
            if not stem.startswith(prefix):
                continue
            class_val = stem[len(prefix):]
            with open(path, "rb") as f:
                models[class_val] = pickle.load(f)
        log.info(f"Loaded {len(models)} {category} models.")
        return models

    def predict(self, embedding: np.ndarray) -> dict:
        # return prediction dict for a single embedding
        # attack class probabilities
        attack_probs = {}
        for cls, model in self.attack_models.items():
            emb_scaled = model["scaler"].transform(embedding.reshape(1, -1))
            prob = model["svm"].predict_proba(emb_scaled)[0][1]
            attack_probs[cls] = float(prob)

        # bypass technique probabilities
        bypass_probs = {}
        for cls, model in self.bypass_models.items():
            emb_scaled = model["scaler"].transform(embedding.reshape(1, -1))
            prob = model["svm"].predict_proba(emb_scaled)[0][1]
            bypass_probs[cls] = float(prob)

        max_attack = max(attack_probs.values()) if attack_probs else 0.0
        max_bypass = max(bypass_probs.values()) if bypass_probs else 0.0
        is_jailbreak = bool(max(max_attack, max_bypass) >= self.threshold)

        return {
            "is_jailbreak": is_jailbreak,
            "attack_class_probs": attack_probs,
            "bypass_technique_probs": bypass_probs,
        }

# evaluate various metrics

def evaluate_fpr(detector: JailbreakDetector, max_samples: int = None) -> float:
    # returns FPR = false_positives / total_samples
    topic_index = load_topic_index()
    total_fp = 0
    total_samples = 0

    log.info("Evaluating on educational data...")
    for topic_name, info in tqdm(topic_index.items(), desc="Topics"):
        topic_id = info["topic_id"]
        try:
            embeddings = load_full_question_embeddings(topic_id)
        except FileNotFoundError:
            log.warning(f"Embeddings missing for {topic_name} ({topic_id}), skipping.")
            continue

        # can limit samples per topic for speed
        if max_samples and len(embeddings) > max_samples:
            idx = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings = embeddings[idx]

        for emb in embeddings:
            pred = detector.predict(emb)
            if pred["is_jailbreak"]:
                total_fp += 1
        total_samples += len(embeddings)

    if total_samples == 0:
        log.error("No educational samples found")
        return float('nan')

    fpr = total_fp / total_samples
    log.info(f"Educational samples: {total_samples}, False Positives: {total_fp}, FPR = {fpr:.4f}")
    return fpr


def evaluate_fnr(detector: JailbreakDetector, max_samples: int = None) -> float:
    # returns FNR = false_negatives / total_samples.
    # load jailbreak embeddings and metadata
    emb_file = JAILBREAK_DIR / "embeddings.npy"
    meta_file = JAILBREAK_DIR / "metadata.json"
    if not emb_file.exists() or not meta_file.exists():
        raise FileNotFoundError("Jailbreak embeddings or metadata not found. Run --prepare first")

    jailbreak_emb = np.load(emb_file)
    with open(meta_file, "r") as f:
        jailbreak_meta = json.load(f)

    # added subsample option
    if max_samples and len(jailbreak_emb) > max_samples:
        idx = np.random.choice(len(jailbreak_emb), max_samples, replace=False)
        jailbreak_emb = jailbreak_emb[idx]
        jailbreak_meta = [jailbreak_meta[i] for i in idx]

    total_fn = 0
    log.info("Evaluating on jailbreak data...")
    for i, emb in enumerate(tqdm(jailbreak_emb, desc="Jailbreak samples")):
        pred = detector.predict(emb)
        if not pred["is_jailbreak"]:
            total_fn += 1

    total_samples = len(jailbreak_emb)
    fnr = total_fn / total_samples
    log.info(f"Jailbreak samples: {total_samples}, False Negatives: {total_fn}, FNR = {fnr:.4f}")
    return fnr

# added various arg options for the script
# reduced version of the train script options

def main():
    parser = argparse.ArgumentParser(description="Evaluate jailbreak detector")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection threshold (default: 0.5)")
    parser.add_argument("--max-edu-samples", type=int, default=None,
                        help="Maximum number of educational samples per topic (for speed)")
    parser.add_argument("--max-jailbreak-samples", type=int, default=None,
                        help="Maximum number of jailbreak samples to evaluate")
    parser.add_argument("--fpr-only", action="store_true",
                        help="Only compute FPR (skip jailbreak evaluation)")
    parser.add_argument("--fnr-only", action="store_true",
                        help="Only compute FNR (skip educational evaluation)")
    args = parser.parse_args()

    # init detector
    log.info(f"Loading models with threshold = {args.threshold}")
    detector = JailbreakDetector(threshold=args.threshold)

    results = {}
    if not args.fnr_only:
        fpr = evaluate_fpr(detector, max_samples=args.max_edu_samples)
        results["fpr"] = fpr
    if not args.fpr_only:
        fnr = evaluate_fnr(detector, max_samples=args.max_jailbreak_samples)
        results["fnr"] = fnr

    # print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("\n")
    if "fpr" in results:
        print(f"False Positive Rate (edu marked jailbreak): {results['fpr']:.4f}")
    if "fnr" in results:
        print(f"False Negative Rate (jailbreak marked safe):       {results['fnr']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()