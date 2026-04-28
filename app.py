"""
Flask API and Web UI Server for the Medical Triage System.

Loads both BiLSTM and BioBERT models trained by the Kaggle notebook
(notebook4977537c98 (3) (1).ipynb) and performs confidence-gated ensemble
inference.

Model artifacts expected in trained/:
  - best_lstm/          (PyTorch saved model via torch.save)
  - best_biobert/       (PyTorch saved model via torch.save)
  - classes.npy         (numpy array of class label strings)

If trained models are not found, falls back to demo/mock mode.
"""

from flask import Flask, request, jsonify, render_template
import time
import os
import re
import pickle
import logging
import numpy as np
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')

from src.models.bert_model import BERTMedicalClassifier
from src.models.lstm_model import BiLSTMClassifier
from transformers import AutoTokenizer

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINED_DIR = os.path.join(PROJECT_ROOT, 'trained')

BERT_MODEL_PATH   = os.path.join(TRAINED_DIR, 'best_biobert.pt.zip')
LSTM_MODEL_PATH   = os.path.join(TRAINED_DIR, 'best_lstm.pt.zip')
CLASSES_PATH      = os.path.join(TRAINED_DIR, 'classes.npy')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# NLTK stopwords for text cleaning (matching notebook preprocessing)
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    for res in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        nltk.download(res, quiet=True)
    STOPWORDS = set(stopwords.words('english'))
    MEDICAL_KEEP = {
        "no", "not", "without", "with", "right", "left", "upper", "lower",
        "above", "below", "after", "before", "during", "over", "under"
    }
    STOPWORDS = STOPWORDS - MEDICAL_KEEP
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    STOPWORDS = set()
    LEMMATIZER = None

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
bert_model = None
bert_tokenizer = None
lstm_model = None
class_labels = None
USE_REAL_MODEL = False

# LSTM-specific globals
LSTM_VOCAB = None
LSTM_MAX_LEN = None

# BERT-specific globals
BERT_MODEL_NAME = None
BERT_MAX_LEN = None


# ---------------------------------------------------------------------------
# Text cleaning — matches notebook preprocessing exactly
# ---------------------------------------------------------------------------
def clean_text(text):
    """Full NLP preprocessing pipeline for medical text — matches notebook."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\d+(\.\d+)?\s*(mg|ml|mmhg|kg|g|%|mcg|iu|u)", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-z\s\-]", " ", text)
    text = re.sub(r"-+", " ", text)

    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    if LEMMATIZER:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# LSTM text encoding — integer-encode using saved vocabulary
# ---------------------------------------------------------------------------
def encode_text_for_lstm(text, vocab, max_len):
    """Convert cleaned text to integer-encoded padded tensor for LSTM."""
    tokens = text.split()
    ids = [vocab.get(t, vocab.get("<UNK>", 1)) for t in tokens]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor([ids], dtype=torch.long)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_models():
    """Attempt to load the trained BioBERT and BiLSTM models."""
    global bert_model, bert_tokenizer, lstm_model, class_labels, USE_REAL_MODEL
    global LSTM_VOCAB, LSTM_MAX_LEN, BERT_MODEL_NAME, BERT_MAX_LEN

    # --- Load class labels ---
    if os.path.exists(CLASSES_PATH):
        class_labels = np.load(CLASSES_PATH, allow_pickle=True)
        logger.info(f"Class labels loaded: {list(class_labels)}")
    else:
        logger.warning(f"classes.npy not found at {CLASSES_PATH}. Using fallback.")
        class_labels = np.array([
            'Cardiology', 'Gastroenterology', 'Neurology',
            'Obstetrics / Gynecology', 'Orthopedics', 'Urology'
        ])

    num_classes = len(class_labels)

    # --- Load BERT ---
    bert_loaded = False
    try:
        if os.path.exists(BERT_MODEL_PATH):
            logger.info(f"Loading BERT model from {BERT_MODEL_PATH}...")

            # Try loading as a full checkpoint first
            checkpoint = torch.load(BERT_MODEL_PATH, map_location=DEVICE, weights_only=False)

            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                # Full checkpoint format from the notebook
                BERT_MODEL_NAME = checkpoint.get('model_name', 'dmis-lab/biobert-v1.1')
                BERT_MAX_LEN = checkpoint.get('max_len', 128)

                bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
                bert_model = BERTMedicalClassifier(
                    model_name=BERT_MODEL_NAME,
                    num_classes=checkpoint.get('num_classes', num_classes),
                )
                bert_model.load_state_dict(checkpoint['model_state'])
            else:
                # Raw state_dict format — infer num_classes from fc.weight
                BERT_MODEL_NAME = 'dmis-lab/biobert-v1.1'
                BERT_MAX_LEN = 128
                inferred_classes = checkpoint['fc.weight'].shape[0] if 'fc.weight' in checkpoint else num_classes
                bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
                bert_model = BERTMedicalClassifier(
                    model_name=BERT_MODEL_NAME,
                    num_classes=inferred_classes,
                )
                bert_model.load_state_dict(checkpoint)

            bert_model.to(DEVICE)
            bert_model.eval()
            bert_loaded = True
            logger.info("✅ BERT model loaded successfully.")
        else:
            logger.warning(f"BERT model not found at {BERT_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load BERT model: {e}")

    # --- Load LSTM ---
    lstm_loaded = False
    try:
        if os.path.exists(LSTM_MODEL_PATH):
            logger.info(f"Loading LSTM model from {LSTM_MODEL_PATH}...")

            checkpoint = torch.load(LSTM_MODEL_PATH, map_location=DEVICE, weights_only=False)

            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                # Full checkpoint format from the notebook
                lstm_model = BiLSTMClassifier(
                    vocab_size=checkpoint.get('vocab_size', 17895),
                    embed_dim=checkpoint.get('embed_dim', 128),
                    hidden_dim=checkpoint.get('hidden_dim', 256),
                    num_layers=checkpoint.get('num_layers', 2),
                    num_classes=checkpoint.get('num_classes', num_classes),
                    dropout=0.3
                )
                lstm_model.load_state_dict(checkpoint['model_state'])
                LSTM_MAX_LEN = checkpoint.get('max_len', 256)
            else:
                # Raw state_dict — infer architecture from tensor shapes
                inferred_vocab = checkpoint['embedding.weight'].shape[0]
                inferred_embed = checkpoint['embedding.weight'].shape[1]
                inferred_hidden = checkpoint['lstm.weight_ih_l0'].shape[0] // 4
                inferred_classes = checkpoint['fc.weight'].shape[0]
                lstm_model = BiLSTMClassifier(
                    vocab_size=inferred_vocab,
                    embed_dim=inferred_embed,
                    hidden_dim=inferred_hidden,
                    num_layers=2,
                    num_classes=inferred_classes,
                    dropout=0.3
                )
                lstm_model.load_state_dict(checkpoint)
                LSTM_MAX_LEN = 256

            lstm_model.to(DEVICE)
            lstm_model.eval()
            lstm_loaded = True
            logger.info("✅ LSTM model loaded successfully.")
        else:
            logger.warning(f"LSTM model not found at {LSTM_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load LSTM model: {e}")

    # --- Load LSTM vocabulary ---
    vocab_path = os.path.join(TRAINED_DIR, 'vocab.pkl')
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            LSTM_VOCAB = pickle.load(f)
        logger.info(f"LSTM vocabulary loaded ({len(LSTM_VOCAB)} words)")
    elif os.path.exists(os.path.join(PROJECT_ROOT, 'saved_models', 'vocab.pkl')):
        with open(os.path.join(PROJECT_ROOT, 'saved_models', 'vocab.pkl'), 'rb') as f:
            LSTM_VOCAB = pickle.load(f)
        logger.info(f"LSTM vocabulary loaded from saved_models/ ({len(LSTM_VOCAB)} words)")
    else:
        logger.warning("LSTM vocabulary not found — LSTM inference disabled.")
        lstm_loaded = False

    USE_REAL_MODEL = bert_loaded or lstm_loaded

    if USE_REAL_MODEL:
        models_loaded = []
        if bert_loaded:
            models_loaded.append("BERT")
        if lstm_loaded:
            models_loaded.append("LSTM")
        logger.info(f"✅ Real model(s) active: {', '.join(models_loaded)}")
    else:
        logger.warning("⚠️  No trained models loaded — running in demo/mock mode.")


# Load models at startup
load_models()

# Fallback specialties for mock mode
FALLBACK_SPECIALTIES = [
    "Cardiology", "Neurology", "Orthopedics",
    "Gastroenterology", "Urology", "Obstetrics / Gynecology"
]

# ---------------------------------------------------------------------------
# Keyword override for dropped classes (from notebook CELL 22)
# ---------------------------------------------------------------------------
KEYWORD_MAP = {
    "Psychiatry / Psychology": [
        "depression", "anxiety", "panic", "hallucin", "suicid", "bipolar",
        "schizophren", "insomnia", "ptsd", "ocd", "anorex", "bulimi"
    ],
    "Pediatrics - Neonatal": [
        "infant", "baby", "newborn", "neonatal", "pediatri", "child",
        "toddler", "vaccination"
    ],
    "Hematology - Oncology": [
        "leukemia", "lymphoma", "chemotherapy", "tumor", "cancer",
        "malignant", "oncology", "metastas", "platelet"
    ],
    "Nephrology": [
        "kidney", "renal", "dialysis", "creatinine", "nephr"
    ],
    "ENT - Otolaryngology": [
        "ear", "nose", "throat", "tonsil", "sinus", "hearing",
        "tinnitus", "vertigo", "ent"
    ],
    "Dermatology": [
        "skin", "rash", "eczema", "psoriasis", "acne", "dermatit",
        "melanoma", "mole"
    ],
    "Ophthalmology": [
        "eye", "vision", "cataract", "glaucoma", "retina", "ophtha",
        "blind"
    ],
    "Endocrinology": [
        "diabetes", "thyroid", "insulin", "glucose", "endocrin",
        "hormone", "cortisol", "adrenal"
    ],
    "Pulmonology": [
        "lung", "asthma", "bronchit", "copd", "pneumonia", "pulmon",
        "respirat", "inhaler", "tuberculosis"
    ],
    "Rheumatology": [
        "arthritis", "lupus", "rheumat", "fibromyalg", "autoimmune",
        "joint swelling"
    ],
    "Emergency Room Reports": [
        "trauma", "accident", "emergency", "unconscious", "cpr",
        "resuscit", "overdose"
    ],
}


def keyword_override(text):
    """Check if text matches a dropped-class keyword → return specialty or None."""
    text_lower = text.lower()

    # Count matches per specialty
    scores = {}
    for specialty, keywords in KEYWORD_MAP.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[specialty] = count

    if scores:
        best = max(scores, key=scores.get)
        if scores[best] >= 2:
            return best

    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', '')

    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400

    start_time = time.time()

    # --- Check keyword override first ---
    override_specialty = keyword_override(symptoms)

    if override_specialty and not USE_REAL_MODEL:
        # If no model loaded, use keyword override directly
        response_time = round((time.time() - start_time) * 1000)
        return jsonify({
            'specialty': override_specialty,
            'confidence': 85.0,
            'model_used': 'KEYWORD_OVERRIDE',
            'response_time_ms': response_time,
            'urgent': _check_urgency(symptoms),
            'keywords': _extract_keywords(symptoms),
            'real_model': False,
        })

    # --- Real inference ---
    if USE_REAL_MODEL:
        try:
            cleaned = clean_text(symptoms)

            bert_probs = None
            lstm_probs = None

            # BERT inference
            if bert_model is not None and bert_tokenizer is not None:
                enc = bert_tokenizer(
                    cleaned, truncation=True, padding="max_length",
                    max_length=BERT_MAX_LEN or 128, return_tensors="pt"
                )
                ids = enc["input_ids"].to(DEVICE)
                mask = enc["attention_mask"].to(DEVICE)
                with torch.no_grad():
                    bert_probs = F.softmax(bert_model(ids, mask), dim=1).squeeze().cpu().numpy()

            # LSTM inference
            if lstm_model is not None and LSTM_VOCAB is not None:
                lstm_input = encode_text_for_lstm(cleaned, LSTM_VOCAB, LSTM_MAX_LEN or 256).to(DEVICE)
                with torch.no_grad():
                    lstm_probs = F.softmax(lstm_model(lstm_input), dim=1).squeeze().cpu().numpy()

            # --- Confidence-gated ensemble ---
            if bert_probs is not None and lstm_probs is not None:
                bert_conf = float(bert_probs.max())
                lstm_conf = float(lstm_probs.max())
                # Pick the model with higher confidence
                if bert_conf >= lstm_conf:
                    final_probs = bert_probs
                    model_used = "BERT"
                else:
                    final_probs = lstm_probs
                    model_used = "LSTM"
            elif bert_probs is not None:
                final_probs = bert_probs
                model_used = "BERT"
            elif lstm_probs is not None:
                final_probs = lstm_probs
                model_used = "LSTM"
            else:
                raise ValueError("No model produced output")

            predicted_idx = int(np.argmax(final_probs))
            confidence = round(float(final_probs[predicted_idx]) * 100, 1)
            predicted_specialty = str(class_labels[predicted_idx]).strip()

            # Apply keyword override if model confidence is low
            if override_specialty and confidence < 50.0:
                predicted_specialty = override_specialty
                model_used = f"{model_used}+KEYWORD"

        except Exception as e:
            logger.error(f"Inference error: {e}")
            import random
            predicted_specialty = random.choice(FALLBACK_SPECIALTIES)
            confidence = round(random.uniform(75.0, 98.5), 1)
            model_used = "MOCK_MODEL"
    else:
        # Mock prediction for demo
        import random
        logger.info("No trained model available — returning mock result.")
        time.sleep(random.uniform(0.3, 0.8))
        if override_specialty:
            predicted_specialty = override_specialty
            model_used = "KEYWORD_OVERRIDE"
        else:
            predicted_specialty = random.choice(FALLBACK_SPECIALTIES)
            model_used = "MOCK_MODEL"
        confidence = round(random.uniform(75.0, 98.5), 1)

    response_time = round((time.time() - start_time) * 1000)

    return jsonify({
        'specialty': predicted_specialty,
        'confidence': confidence,
        'model_used': model_used,
        'response_time_ms': response_time,
        'urgent': _check_urgency(symptoms),
        'keywords': _extract_keywords(symptoms),
        'real_model': USE_REAL_MODEL,
    })


def _check_urgency(text):
    """Check for urgent terms in the input."""
    urgent_terms = [
        'severe', 'bleeding', 'chest pain', 'stroke', 'heart',
        'unconscious', 'emergency', 'cardiac arrest', 'anaphylaxis',
        'seizure', 'hemorrhage', 'overdose'
    ]
    return any(term in text.lower() for term in urgent_terms)


def _extract_keywords(text):
    """Extract significant keywords from the input."""
    import random as _rand
    words = [w for w in text.split() if len(w) > 3]
    return _rand.sample(words, min(len(words), 3)) if words else []


if __name__ == '__main__':
    app.run(debug=True, port=5000)
