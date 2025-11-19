import numpy as np
import librosa
import warnings
import tensorflow as tf
import math
import itertools
import plotly.graph_objects as go
from collections import Counter
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from joblib import Parallel, delayed
import multiprocessing
from sklearn.decomposition import PCA
import plotly.io as pio
import os
import json

# Import Gemini for AI feedback
import google.generativeai as genai
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ========================================
# CONFIGURATION
# ========================================
SR = 22050
SEG_LEN = 3.0
HOP = 1.5
N_MFCC = 20
N_MELS = 128
N_FORMANTS = 4
PCA_COMPONENTS = 10
DTW_TOP_PERCENT = 0.3
N_JOBS = min(4, multiprocessing.cpu_count())

# Feedback thresholds
PITCH_CORR_TH = 0.5
DTW_DIST_TH = 60
ENERGY_CORR_TH = 0.5
TEMPO_DIFF_TH = 6

# ========================================
# GEMINI API KEY & MODEL ROTATION
# ========================================
load_dotenv()


GEMINI_API_KEYS = [
    os.environ.get("GEMINI_API_KEY_1"),
    os.environ.get("GEMINI_API_KEY_2"),
    os.environ.get("GEMINI_API_KEY_3"),
    os.environ.get("GEMINI_API_KEY_4"),
]

# Filter out None/undefined keys
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key is not None and key != '']


TEXT_ONLY_GEMINI_MODELS = [
    "gemini-2.5-pro",           
    "gemini-2.5-flash",       
    "gemini-pro-latest",      
    "gemini-flash-latest",      
    "gemini-2.0-flash-001",  
    "gemini-2.0-flash-lite-001"
]

# Track current indices for rotation
current_key_index = 0
current_model_index = 0

print(f"✅ Loaded {len(GEMINI_API_KEYS)} Gemini API key(s)")
print(f"✅ Available models: {len(TEXT_ONLY_GEMINI_MODELS)}")

def get_next_api_key():
    """Get the next API key in rotation."""
    global current_key_index
    if not GEMINI_API_KEYS:
        raise ValueError("No Gemini API keys configured")
    key = GEMINI_API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
    return key

def get_next_model():
    """Get the next model in rotation."""
    global current_model_index
    model = TEXT_ONLY_GEMINI_MODELS[current_model_index]
    current_model_index = (current_model_index + 1) % len(TEXT_ONLY_GEMINI_MODELS)
    return model

def is_rate_limit_error(error):
    """Check if error is a rate limit error."""
    error_message = str(error).lower()
    error_code = getattr(error, 'code', None) or getattr(error, 'status', None)
    return (
        error_code == 429 or
        error_code == 'RESOURCE_EXHAUSTED' or
        'rate limit' in error_message or
        'quota' in error_message or
        'resource exhausted' in error_message or
        'too many requests' in error_message
    )

def call_gemini_with_retry(prompt, max_retries=None):
    """
    Call Gemini API with automatic retry logic for rate limits.
    Rotates through API keys and models on rate limit errors.
    """
    if max_retries is None:
        max_retries = len(GEMINI_API_KEYS) * len(TEXT_ONLY_GEMINI_MODELS)
    
    last_error = None
    
    for attempt in range(max_retries):
        api_key = get_next_api_key()
        model_name = get_next_model()
        
        try:
            print(f"Attempt {attempt + 1}: Using model {model_name} with key ...{api_key[-4:]}")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            print(f"✅ Success with model {model_name}")
            return response.text
        
        except Exception as error:
            print(f"❌ Attempt {attempt + 1} failed with model {model_name}: {str(error)}")
            last_error = error
            
            # If it's a rate limit error, try next key/model combination
            if is_rate_limit_error(error):
                print("⚠️ Rate limit detected, rotating to next API key and model...")
                continue
            
            # If it's not a rate limit error, throw immediately
            raise error
    
    # All retries exhausted
    raise RuntimeError(f"All API keys and models exhausted. Last error: {str(last_error)}")

# ========================================
# GEMINI FEEDBACK FUNCTION
# ========================================
def get_gemini_feedback(analysis_data):
    """
    Generate AI coaching feedback using Gemini based on analysis results.
    
    Args:
        analysis_data: Dictionary containing analysis results with keys like:
            - prob_good, prediction, overall_assessment, feedback_lines
            - comp_feats: dict with mfcc_cosine, pitch_corr, dtw_mean, energy_corr, tempo_diff, chroma_cosine
    
    Returns:
        Dictionary with parsed Gemini feedback or error information
    """
    if not GEMINI_API_KEYS:
        return {
            'error': 'Gemini API not configured. Please add API keys to .env file.',
            'status': 'error'
        }
    
    try:
        # Extract metrics from analysis data
        comp = analysis_data.get('comp_feats', {})
        prob_good = analysis_data.get('prob_good', 0)
        mfcc = comp.get('mfcc_cosine', 0)
        pitch_corr = comp.get('pitch_corr', 0)
        dtw = comp.get('dtw_mean', 0)
        energy_corr = comp.get('energy_corr', 0)
        tempo_diff = comp.get('tempo_diff', 0)
        chroma = comp.get('chroma_cosine', 0)
        
        # Build comprehensive prompt
        prompt = f"""You are an expert vocal coach AI assistant.

DO NOT include any introductory sentence, disclaimer, or natural language outside the structure.
Respond with a valid JSON object only, following this schema:

{{
  "strengths": "<1-2 sentences about performance strengths>",
  "metrics": {{
    "mfcc_similarity": {{
      "label": "MFCC Similarity",
      "value": "{mfcc:.2%}",
      "meaning": "How similar the overall timbre/quality of user's and coach's voices are.",
      "feedback": "<specific actionable feedback based on the value>"
    }},
    "pitch_correlation": {{
      "label": "Pitch Correlation",
      "value": "{float(pitch_corr):.2f}",
      "meaning": "How accurately the user's pitch matches the coach's (Pearson correlation, -1 to 1).",
      "feedback": "<specific actionable feedback based on the value>"
    }},
    "dtw_distance": {{
      "label": "DTW Distance",
      "value": "{float(dtw):.1f}",
      "meaning": "Difference in phrasing/timing (lower is better).",
      "feedback": "<specific actionable feedback based on the value>"
    }},
    "energy_correlation": {{
      "label": "Energy Correlation",
      "value": "{float(energy_corr):.2f}",
      "meaning": "How closely loudness/intensity matches (-1 to 1).",
      "feedback": "<specific actionable feedback based on the value>"
    }},
    "tempo_diff": {{
      "label": "Tempo Difference",
      "value": "{float(tempo_diff):.1f} BPM",
      "meaning": "Difference in song tempo (lower is better).",
      "feedback": "<specific actionable feedback based on the value>"
    }},
    "chroma_similarity": {{
      "label": "Chroma Similarity",
      "value": "{chroma:.2%}",
      "meaning": "Similarity of harmonic structure (chords/keys).",
      "feedback": "<specific actionable feedback based on the value>"
    }}
  }},
  "next_steps": "<2-3 sentences with specific practice recommendations>"
}}

Respond ONLY with valid JSON that follows this schema.
Do NOT include markdown code fences, triple backticks, or any extra description or comments.
"""
        
        # Call Gemini API with retry logic
        response_text = call_gemini_with_retry(prompt)
        
        # Clean up response - remove markdown code fences if present
        response_text = response_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON response
        try:
            feedback_data = json.loads(response_text)
            return {
                'status': 'success',
                'feedback': feedback_data
            }
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response text: {response_text}")
            return {
                'status': 'success',
                'raw_response': response_text,
                'parse_error': str(e)
            }
    
    except Exception as e:
        print(f"Error in get_gemini_feedback: {str(e)}")
        return {
            'error': str(e),
            'status': 'error'
        }

# ========================================
# HELPER FUNCTIONS
# ========================================

def fix_length(y, size):
    """Pad with zeros or truncate to make array length == size."""
    if len(y) == size:
        return y
    if len(y) < size:
        pad_width = size - len(y)
        return np.pad(y, (0, pad_width), mode='constant')
    return y[:size]

def safe_pearson(x, y):
    """Safe pearson correlation: handles constant arrays and small lengths."""
    try:
        if len(x) < 2 or len(y) < 2:
            return 0.0
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            return 0.0
        r, _ = pearsonr(x, y)
        if np.isnan(r):
            return 0.0
        return float(r)
    except Exception:
        return 0.0

def estimate_formants(y, sr, n_formants=4):
    """Estimate formant frequencies using LPC."""
    try:
        order = 2 + sr // 1000
        a = librosa.lpc(y, order=order)
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]
        angz = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angz * (sr / (2 * np.pi))
        freqs = np.sort(freqs[freqs > 90])
        formants = freqs[:n_formants] if len(freqs) >= n_formants else np.pad(freqs, (0, n_formants - len(freqs)), constant_values=0)
        return formants
    except Exception:
        return np.zeros(n_formants)

def estimate_vibrato(f0, sr, hop_length=512):
    """Estimate vibrato rate and extent from pitch contour."""
    try:
        f0_clean = f0[~np.isnan(f0)]
        f0_clean = f0_clean[f0_clean > 0]
        if len(f0_clean) < 10:
            return 0.0, 0.0
        
        peaks, _ = find_peaks(f0_clean)
        if len(peaks) > 1:
            time_per_frame = hop_length / sr
            peak_intervals = np.diff(peaks) * time_per_frame
            vibrato_rate = 1.0 / np.mean(peak_intervals) if len(peak_intervals) > 0 else 0.0
        else:
            vibrato_rate = 0.0
        
        vibrato_extent = np.std(f0_clean)
        return vibrato_rate, vibrato_extent
    except Exception:
        return 0.0, 0.0

def estimate_voice_onset(y, sr):
    """Estimate voice onset time."""
    try:
        rms = librosa.feature.rms(y=y)[0]
        threshold = np.max(rms) * 0.1
        onset_idx = np.argmax(rms > threshold)
        onset_time = librosa.frames_to_time(onset_idx, sr=sr)
        return onset_time
    except Exception:
        return 0.0

# ========================================
# SEGMENTATION FUNCTIONS
# ========================================

def make_segments(y, sr, seg_len=SEG_LEN, hop=HOP):
    """Split audio into overlapping segments."""
    seg_samples = int(seg_len * sr)
    hop_samples = int(hop * sr)
    segments = []
    
    if len(y) < seg_samples:
        segments.append(fix_length(y, seg_samples))
        return segments
    
    for start in range(0, len(y) - seg_samples + 1, hop_samples):
        seg = y[start:start + seg_samples]
        if len(seg) == seg_samples:
            segments.append(seg)
    
    return segments

def align_segments_dtw_optimized(coach_segments, user_segments, sr=SR, n_mfcc=N_MFCC,
                                  pca_components=PCA_COMPONENTS, top_percent=DTW_TOP_PERCENT):
    """Optimized DTW alignment with pre-filtering, PCA, and top candidates."""
    coach_mfccs = [librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc) for seg in coach_segments]
    user_mfccs = [librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc) for seg in user_segments]
    
    coach_means = [np.mean(mfcc, axis=1) for mfcc in coach_mfccs]
    user_means = [np.mean(mfcc, axis=1) for mfcc in user_mfccs]
    
    pre_filter_pairs = []
    for i, c_mean in enumerate(coach_means):
        for j, u_mean in enumerate(user_means):
            euc_dist = euclidean(c_mean, u_mean)
            pre_filter_pairs.append((i, j, euc_dist))
    
    pre_filter_pairs.sort(key=lambda x: x[2])
    n_candidates = int(len(pre_filter_pairs) * top_percent)
    n_candidates = max(n_candidates, min(len(coach_segments), len(user_segments)))
    top_candidates = pre_filter_pairs[:n_candidates]
    
    all_mfccs = coach_mfccs + user_mfccs
    max_frames = max(mfcc.shape[1] for mfcc in all_mfccs)
    
    mfcc_padded = []
    for mfcc in all_mfccs:
        if mfcc.shape[1] < max_frames:
            padded = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode='constant')
        else:
            padded = mfcc[:, :max_frames]
        mfcc_padded.append(padded.T)
    
    pca = PCA(n_components=min(pca_components, n_mfcc))
    all_frames = np.vstack(mfcc_padded)
    pca.fit(all_frames)
    
    coach_mfccs_pca = [pca.transform(mfcc.T).T for mfcc in coach_mfccs]
    user_mfccs_pca = [pca.transform(mfcc.T).T for mfcc in user_mfccs]
    
    dtw_pairs = []
    for i, j, _ in top_candidates:
        try:
            c_mfcc = coach_mfccs_pca[i]
            u_mfcc = user_mfccs_pca[j]
            D, _ = librosa.sequence.dtw(X=c_mfcc, Y=u_mfcc, metric='cosine')
            dtw_dist = float(np.mean(D))
            dtw_pairs.append((i, j, dtw_dist))
        except Exception:
            dtw_pairs.append((i, j, 1e6))
    
    dtw_pairs.sort(key=lambda x: x[2])
    
    max_pairs = min(len(coach_segments), len(user_segments))
    used_coach = set()
    used_user = set()
    aligned_pairs = []
    
    for c_idx, u_idx, dist in dtw_pairs:
        if c_idx not in used_coach and u_idx not in used_user:
            aligned_pairs.append((c_idx, u_idx, dist))
            used_coach.add(c_idx)
            used_user.add(u_idx)
            if len(aligned_pairs) >= max_pairs:
                break
    
    return aligned_pairs

# ========================================
# FEATURE EXTRACTION
# ========================================

def extract_sequence_features(y, sr=SR, n_mfcc=N_MFCC, n_mels=N_MELS, n_formants=N_FORMANTS):
    """Extract comprehensive time-series features for RNN input."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    
    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0 = np.nan_to_num(f0)
        f0 = f0.reshape(1, -1)
    except Exception:
        f0 = np.zeros((1, mfcc.shape[1]))
    
    min_frames = min(mfcc.shape[1], chroma.shape[1], spectral_centroid.shape[1],
                     spectral_rolloff.shape[1], spectral_contrast.shape[1], zcr.shape[1], rms.shape[1], f0.shape[1])
    
    features = np.vstack([mfcc[:, :min_frames], chroma[:, :min_frames], spectral_centroid[:, :min_frames],
                          spectral_rolloff[:, :min_frames], spectral_contrast[:, :min_frames],
                          zcr[:, :min_frames], rms[:, :min_frames], f0[:, :min_frames]])
    
    return features.T

# ========================================
# COMPARISON FEATURES
# ========================================

def compute_comparison_features(y1, y2, sr=SR, n_mfcc=N_MFCC):
    """Compute comprehensive comparison features between two audio segments."""
    features = {}
    
    try:
        mf1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=n_mfcc)
        mf2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=n_mfcc)
        mf1_mean = np.mean(mf1, axis=1)
        mf2_mean = np.mean(mf2, axis=1)
        mfcc_cosine = 1.0 - cosine(mf1_mean, mf2_mean) if not np.allclose(mf1_mean, 0) and not np.allclose(mf2_mean, 0) else 0.0
        features['mfcc_cosine'] = float(mfcc_cosine) if not np.isnan(mfcc_cosine) else 0.0
    except Exception:
        features['mfcc_cosine'] = 0.0
        mf1 = mf2 = None
    
    try:
        f0_1, _, _ = librosa.pyin(y1, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0_2, _, _ = librosa.pyin(y2, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0_1, f0_2 = np.nan_to_num(f0_1), np.nan_to_num(f0_2)
        features['pitch_corr'] = safe_pearson(f0_1, f0_2)
    except Exception:
        features['pitch_corr'] = 0.0
    
    try:
        if mf1 is not None and mf2 is not None:
            D, _ = librosa.sequence.dtw(X=mf1, Y=mf2, metric='cosine')
            features['dtw_mean'] = float(np.mean(D))
        else:
            features['dtw_mean'] = 1e6
    except Exception:
        features['dtw_mean'] = 1e6
    
    try:
        rms1, rms2 = librosa.feature.rms(y=y1).flatten(), librosa.feature.rms(y=y2).flatten()
        L = min(len(rms1), len(rms2))
        features['energy_corr'] = safe_pearson(rms1[:L], rms2[:L]) if L >= 2 else 0.0
    except Exception:
        features['energy_corr'] = 0.0
    
    try:
        t1, _ = librosa.beat.beat_track(y=y1, sr=sr)
        t2, _ = librosa.beat.beat_track(y=y2, sr=sr)
        features['tempo_diff'] = abs(float(t1) - float(t2))
    except Exception:
        features['tempo_diff'] = 0.0
    
    try:
        chroma1, chroma2 = librosa.feature.chroma_stft(y=y1, sr=sr), librosa.feature.chroma_stft(y=y2, sr=sr)
        chroma1_mean, chroma2_mean = np.mean(chroma1, axis=1), np.mean(chroma2, axis=1)
        chroma_sim = 1.0 - cosine(chroma1_mean, chroma2_mean) if not np.allclose(chroma1_mean, 0) and not np.allclose(chroma2_mean, 0) else 0.0
        features['chroma_cosine'] = float(chroma_sim) if not np.isnan(chroma_sim) else 0.0
    except Exception:
        features['chroma_cosine'] = 0.0
    
    return features

def compute_soft_label(comp_features):
    """Compute a soft (probabilistic) label based on comparison features."""
    mfcc_score = np.clip(comp_features['mfcc_cosine'], 0, 1)
    pitch_score = np.clip((comp_features['pitch_corr'] + 1) / 2, 0, 1)
    dtw_score = np.clip(1.0 - (comp_features['dtw_mean'] / 200.0), 0, 1)
    energy_score = np.clip((comp_features['energy_corr'] + 1) / 2, 0, 1)
    tempo_score = np.clip(1.0 - (comp_features['tempo_diff'] / 30.0), 0, 1)
    chroma_score = np.clip(comp_features['chroma_cosine'], 0, 1)
    
    soft_label = (0.25 * pitch_score + 0.25 * mfcc_score + 0.20 * dtw_score +
                  0.15 * energy_score + 0.10 * chroma_score + 0.05 * tempo_score)
    
    return float(np.clip(soft_label, 0, 1))

# ========================================
# FEEDBACK GENERATION
# ========================================

def generate_feedback(coach_path, user_path, model, sr=SR):
    """Generate detailed feedback by comparing user singing to coach."""
    yc, _ = librosa.load(coach_path, sr=sr, mono=True)
    yu, _ = librosa.load(user_path, sr=sr, mono=True)
    
    user_seq = extract_sequence_features(yu, sr)
    
    expected_len = model.input_shape[1]
    n_features = model.input_shape[2]
    
    if user_seq.shape[0] < expected_len:
        padded_seq = np.zeros((expected_len, n_features))
        padded_seq[:user_seq.shape[0], :] = user_seq
    else:
        padded_seq = user_seq[:expected_len, :]
    
    X_input = np.expand_dims(padded_seq, axis=0)
    prob_good = float(model.predict(X_input, verbose=0)[0][0])
    prediction = prob_good
    
    comp_feats = compute_comparison_features(yc, yu, sr)
    
    if prediction >= 0.8:
        overall_assessment = "Good match!"
    elif prediction >= 0.6:
        overall_assessment = "Fair"
    else:
        overall_assessment = "Needs Improvement"
    
    feedback_lines = []
    
    if comp_feats['pitch_corr'] < PITCH_CORR_TH:
        feedback_lines.append(f"**Pitch**: Weak alignment (correlation={comp_feats['pitch_corr']:.2f}). Focus on matching the coach's pitch more closely. Practice with a tuner.")
    else:
        feedback_lines.append(f"**Pitch**: Good alignment (correlation={comp_feats['pitch_corr']:.2f}). Keep up the good work!")
    
    if comp_feats['dtw_mean'] > DTW_DIST_TH:
        feedback_lines.append(f"**Timing**: Mismatch detected (DTW={comp_feats['dtw_mean']:.1f}). Pay attention to timing, especially in transitions and phrase endings.")
    else:
        feedback_lines.append(f"**Timing**: Well-aligned (DTW={comp_feats['dtw_mean']:.1f}). Your timing is solid!")
    
    if comp_feats['energy_corr'] < ENERGY_CORR_TH:
        feedback_lines.append(f"**Dynamics**: Differs from coach (correlation={comp_feats['energy_corr']:.2f}). Try to match the loudness and intensity variations.")
    else:
        feedback_lines.append(f"**Dynamics**: Well-matched (correlation={comp_feats['energy_corr']:.2f}). Your volume control is excellent!")
    
    if comp_feats['tempo_diff'] > TEMPO_DIFF_TH:
        feedback_lines.append(f"**Tempo**: Differs by {comp_feats['tempo_diff']:.1f} BPM. Try to stay closer to the coach's tempo. Use a metronome if needed.")
    else:
        feedback_lines.append(f"**Tempo**: Close match (difference={comp_feats['tempo_diff']:.1f} BPM). Great tempo control!")
    
    return {
        'prob_good': prob_good,
        'prediction': prediction,
        'overall_assessment': overall_assessment,
        'feedback_lines': feedback_lines,
        'comp_feats': comp_feats
    }

# ========================================
# SEGMENT ANALYSIS
# ========================================

def analyze_segments(coach_path, user_path, model, progress_callback=None, sr=SR, seg_len=SEG_LEN, hop=HOP):
    """Perform detailed per-segment analysis comparing user to coach."""
    y_c, _ = librosa.load(coach_path, sr=sr, mono=True)
    y_u, _ = librosa.load(user_path, sr=sr, mono=True)
    
    segs_c = make_segments(y_c, sr, seg_len, hop)
    segs_u = make_segments(y_u, sr, seg_len, hop)
    
    num_segments = min(len(segs_c), len(segs_u))
    segment_results = []
    
    for i in range(num_segments):
        if progress_callback:
            progress_callback(i + 1, num_segments)
        
        c_seg, u_seg = segs_c[i], segs_u[i]
        
        u_features = extract_sequence_features(u_seg, sr)
        expected_len = model.input_shape[1]
        n_features = model.input_shape[2]
        
        if u_features.shape[0] < expected_len:
            padded_features = np.zeros((expected_len, n_features))
            padded_features[:u_features.shape[0], :] = u_features
        else:
            padded_features = u_features[:expected_len, :]
        
        X_input = np.expand_dims(padded_features, axis=0)
        prob_good = float(model.predict(X_input, verbose=0)[0][0])
        
        comp_feats = compute_comparison_features(c_seg, u_seg, sr)
        
        model_score = prob_good
        mfcc_score, pitch_score, energy_score = np.clip(comp_feats['mfcc_cosine'], 0, 1), np.clip(comp_feats['pitch_corr'], 0, 1), np.clip(comp_feats['energy_corr'], 0, 1)
        dtw_score = 1 - np.clip(comp_feats['dtw_mean'] / 100, 0, 1)
        
        composite_score = (model_score * 0.40 + mfcc_score * 0.20 + pitch_score * 0.20 + energy_score * 0.10 + dtw_score * 0.10) * 100
        
        if composite_score >= 80:
            rating = "Excellent ☆☆☆☆☆"
        elif composite_score >= 65:
            rating = "Good ☆☆☆☆"
        elif composite_score >= 50:
            rating = "Fair ☆☆☆"
        elif composite_score >= 35:
            rating = "Needs Work ☆☆"
        else:
            rating = "Poor ☆"
        
        weaknesses = []
        if pitch_score < 0.4:
            weaknesses.append("Pitch")
        if dtw_score < 0.7:
            weaknesses.append("Timing")
        if energy_score < 0.4:
            weaknesses.append("Dynamics")
        if comp_feats['tempo_diff'] > 10:
            weaknesses.append("Tempo")
        
        start_time = i * hop
        end_time = start_time + seg_len
        
        segment_results.append({
            'segment_num': i + 1,
            'time_range': f"{start_time:.1f}s - {end_time:.1f}s",
            'composite_score': composite_score,
            'rating': rating,
            'model_confidence': model_score * 100,
            'mfcc_similarity': mfcc_score * 100,
            'pitch_corr': pitch_score * 100,
            'energy_corr': energy_score * 100,
            'dtw_distance': comp_feats['dtw_mean'],
            'tempo_diff': comp_feats['tempo_diff'],
            'weaknesses': weaknesses if weaknesses else ["None - well done!"]
        })
    
    return segment_results

def save_uploaded_files(coach_file, user_file):
    """Save uploaded Streamlit files to temporary disk files."""
    coach_temp = f"temp_coach.{coach_file.name.split('.')[-1]}"
    user_temp = f"temp_user.{user_file.name.split('.')[-1]}"
    
    with open(coach_temp, "wb") as f:
        f.write(coach_file.getbuffer())
    with open(user_temp, "wb") as f:
        f.write(user_file.getbuffer())
    
    return coach_temp, user_temp

def make_pitch_contour_figure(coach_path, user_path, sr=SR, fmin_note='C2', fmax_note='C7', hop_length=512, y_max=800):
    """Returns a Plotly figure with pitch contours and voiced-region shading for coach_path and user_path."""
    import streamlit as st
    
    # Neon/arcade theme
    bg_color = st.get_option("theme.backgroundColor") or "#0E0E10"
    text_color = st.get_option("theme.textColor") or "#F8F8FF"
    coach_color = "#39FF14"  # neon green
    user_color = "#FF2AEF"   # neon pink
    grid_color = "rgba(255,255,255,0.08)"
    
    pio.templates.default = "plotly_dark"
    
    y_c, _ = librosa.load(coach_path, sr=sr, mono=True)
    y_u, _ = librosa.load(user_path, sr=sr, mono=True)
    
    f0_c, _, _ = librosa.pyin(y_c, fmin=librosa.note_to_hz(fmin_note), fmax=librosa.note_to_hz(fmax_note), sr=sr, hop_length=hop_length)
    f0_u, _, _ = librosa.pyin(y_u, fmin=librosa.note_to_hz(fmin_note), fmax=librosa.note_to_hz(fmax_note), sr=sr, hop_length=hop_length)
    
    t_c = librosa.times_like(f0_c, sr=sr, hop_length=hop_length) if f0_c is not None else np.array([])
    t_u = librosa.times_like(f0_u, sr=sr, hop_length=hop_length) if f0_u is not None else np.array([])
    
    fig = go.Figure()
    
    if len(t_c) and len(f0_c):
        fig.add_trace(go.Scatter(
            x=t_c, y=f0_c, mode='lines', name='Coach',
            line=dict(color=coach_color, width=2),
            hovertemplate="Coach<br>Time: %{x:.2f}s<br>Freq: %{y:.1f} Hz"
        ))
    
    if len(t_u) and len(f0_u):
        fig.add_trace(go.Scatter(
            x=t_u, y=f0_u, mode='lines', name='User',
            line=dict(color=user_color, width=2),
            hovertemplate="User<br>Time: %{x:.2f}s<br>Freq: %{y:.1f} Hz"
        ))
    
    def _voiced_intervals(bool_mask, times):
        intervals = []
        if len(bool_mask) == 0:
            return intervals
        idx = np.where(bool_mask)[0]
        if idx.size == 0:
            return intervals
        
        for k, g in itertools.groupby(enumerate(idx), lambda x: x[0] - x[1]):
            group = list(map(lambda x: x[1], g))
            start_idx, end_idx = group[0], group[-1]
            start_t = float(times[start_idx])
            end_t = float(times[end_idx]) + (times[1] - times[0]) if len(times) > 1 else float(times[end_idx])
            intervals.append((start_t, end_t))
        
        return intervals
    
    voiced_mask_c = (~np.isnan(f0_c)) & (f0_c > 0) if f0_c is not None else np.array([])
    voiced_mask_u = (~np.isnan(f0_u)) & (f0_u > 0) if f0_u is not None else np.array([])
    
    for (s, e) in _voiced_intervals(voiced_mask_c, t_c):
        fig.add_shape(
            type="rect", x0=s, x1=e, xref="x", y0=0, y1=y_max, yref="y",
            fillcolor=coach_color, opacity=0.13, line_width=0, layer="below"
        )
    
    for (s, e) in _voiced_intervals(voiced_mask_u, t_u):
        fig.add_shape(
            type="rect", x0=s, x1=e, xref="x", y0=0, y1=y_max, yref="y",
            fillcolor=user_color, opacity=0.09, line_width=0, layer="below"
        )
    
    fig.update_layout(
        title="Pitch Contour Comparison (Coach vs User)",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        yaxis=dict(range=[0, y_max]),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1,
            font=dict(color=text_color)
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_dark",
        height=420,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=0.7, gridcolor=grid_color)
    fig.update_yaxes(showgrid=True, gridwidth=0.7, gridcolor=grid_color)
    
    return fig