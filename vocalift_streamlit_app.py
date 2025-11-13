"""
VocaLift Streamlit App
Upload audio, run AI-powered vocal analysis, and visualize results.
"""

import streamlit as st
import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from collections import Counter
from tensorflow.keras.models import load_model

# Import all utility functions from vocalift_utils
from vocalift_utils import (
    SR, generate_feedback, analyze_segments, save_uploaded_files,
    make_pitch_contour_figure, compute_soft_label
)

# Model paths
MODEL_PATH = "model/vocalift_model (3).h5"


# ========================================
# INITIALIZE SESSION STATE
# ========================================
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "pitch_contour_fig" not in st.session_state:
    st.session_state.pitch_contour_fig = None
if "full_song_result" not in st.session_state:
    st.session_state.full_song_result = None
if "segment_results" not in st.session_state:
    st.session_state.segment_results = None
if "analysis_audio_files" not in st.session_state:
    st.session_state.analysis_audio_files = {}
if "analysis_to_run" not in st.session_state:
    st.session_state.analysis_to_run = None

# ========================================
# DISPLAY FUNCTIONS
# ========================================
def show_full_song_analysis(result):
    """Display full song analysis results."""
    st.markdown('<h2 class="full-song">📊 Full Song Analysis Results</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Confidence", f"{result['prob_good'] * 100:.1f}%")
    with col2:
        if result['prediction'] >= 0.8:
            assessment_color = "🟢"
        elif result['prediction'] >= 0.6:
            assessment_color = "🟡"
        else:
            assessment_color = "🔴"
        st.metric("Assessment", f"{assessment_color} {result['overall_assessment']}")
    with col3:
        st.metric("Similarity Score", f"{compute_soft_label(result['comp_feats']) * 100:.1f}/100")
    
    st.markdown("---")
    st.subheader("📝 Feedback")
    for i, line in enumerate(result['feedback_lines'], 1):
        st.markdown(f"{i}. {line}")
    
    st.markdown("---")
    st.subheader("📈 Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Similarity**")
        st.metric("MFCC", f"{result['comp_feats']['mfcc_cosine'] * 100:.1f}%")
        st.metric("Pitch", f"{result['comp_feats']['pitch_corr']:.2f}")
        st.metric("Energy", f"{result['comp_feats']['energy_corr']:.2f}")
    with col2:
        st.markdown("**Timing & Tempo**")
        st.metric("DTW Distance", f"{result['comp_feats']['dtw_mean']:.1f}")
        st.metric("Tempo Diff", f"{result['comp_feats']['tempo_diff']:.1f} BPM")

def show_segment_analysis(segment_results):
    """Display segment-by-segment analysis with Plotly charts and details."""
    bg_color = st.get_option("theme.backgroundColor") or "#0E0E10"
    secondary_bg = st.get_option("theme.secondaryBackgroundColor") or "#202023"
    text_color = st.get_option("theme.textColor") or "#F8F8FF"
    primary_color = st.get_option("theme.primaryColor") or "#39FF14"
    
    pio.templates.default = "plotly_dark"
    bar_color = "#39FF14"
    avg_line_color = "#FF2AEF"
    exc_line_color = "#39FF14"
    fair_line_color = "#C4C400"
    poor_line_color = "#FF0059"
    heatmap_colorscale = [[0.0, "#0E0E10"], [0.25, "#FF2AEF"], [0.5, "#39FF14"], [0.75, "#FFD700"], [1.0, "#FFFFFF"]]
    
    scores = [r['composite_score'] for r in segment_results]
    avg_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
    max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
    std_score = float(np.std(scores)) if len(scores) > 0 else 0.0
    segment_nums = [r['segment_num'] for r in segment_results]
    seg_labels = [str(n) for n in segment_nums]
    
    hover_texts = [
        f"<b>Segment {r['segment_num']}</b><br>"
        f"Time: {r.get('time_range', 'N/A')}<br>"
        f"Score: {r['composite_score']:.1f}/100<br>"
        f"Confidence: {r.get('model_confidence', 0):.1f}%<br>"
        f"MFCC: {r.get('mfcc_similarity', 0):.1f}%<br>"
        f"Pitch: {r.get('pitch_corr', 0):.1f}<br>"
        f"Energy: {r.get('energy_corr', 0):.1f}"
        for r in segment_results
    ]
    
    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Segments", len(segment_results))
    with col2:
        st.metric("Average Score", f"{avg_score:.1f}/100")
    with col3:
        st.metric("Best Segment", f"{max_score:.1f}/100")
    with col4:
        st.metric("Consistency (Std)", f"{std_score:.1f}")
    
    st.markdown("---")
    st.subheader("📈 Performance Visualization")
    
    bar = go.Bar(x=seg_labels, y=scores, marker=dict(color=bar_color, line=dict(width=1, color="#888")),
                 customdata=hover_texts, hovertemplate="%{customdata}<extra></extra>", hoverinfo="none", name="Segment Score")
    
    shapes = [
        dict(type="line", xref="x", yref="y", x0=-1, x1=len(seg_labels), y0=avg_score, y1=avg_score,
             line=dict(width=2, dash="dash", color=avg_line_color)),
        dict(type="line", xref="x", yref="y", x0=-1, x1=len(seg_labels), y0=80, y1=80,
             line=dict(width=1, dash="dot", color=exc_line_color)),
        dict(type="line", xref="x", yref="y", x0=-1, x1=len(seg_labels), y0=60, y1=60,
             line=dict(width=1, dash="dot", color=fair_line_color)),
        dict(type="line", xref="x", yref="y", x0=-1, x1=len(seg_labels), y0=40, y1=40,
             line=dict(width=1, dash="dot", color=poor_line_color)),
    ]
    
    bar_layout = go.Layout(title="Per-Segment Performance Scores", xaxis=dict(title="Segment Number"),
                           yaxis=dict(title="Score", range=[0, 105]), shapes=shapes, hovermode="x unified", height=420,
                           template="plotly_dark", font=dict(color=text_color), plot_bgcolor=bg_color, paper_bgcolor=bg_color)
    
    bar_fig = go.Figure(data=[bar], layout=bar_layout)
    if seg_labels:
        bar_fig.add_annotation(x=seg_labels[-1], y=avg_score, text=f"Avg: {avg_score:.1f}", showarrow=False,
                               yanchor="middle", xanchor="left", font=dict(color=avg_line_color))
        bar_fig.add_annotation(x=seg_labels[-1], y=80, text="Excellent Threshold", showarrow=False,
                               yanchor="middle", xanchor="left", font=dict(color=exc_line_color))
        bar_fig.add_annotation(x=seg_labels[-1], y=60, text="Fair Threshold", showarrow=False,
                               yanchor="middle", xanchor="left", font=dict(color=fair_line_color))
        bar_fig.add_annotation(x=seg_labels[-1], y=40, text="Poor Threshold", showarrow=False,
                               yanchor="middle", xanchor="left", font=dict(color=poor_line_color))
    
    st.plotly_chart(bar_fig, use_container_width=True)
    
    metrics_data = np.array([
        [r.get('model_confidence', 0) for r in segment_results],
        [r.get('mfcc_similarity', 0) for r in segment_results],
        [r.get('pitch_corr', 0) for r in segment_results],
        [r.get('energy_corr', 0) for r in segment_results]
    ])
    metric_labels = ['Model\nConfidence', 'MFCC\nSimilarity', 'Pitch\nCorrelation', 'Energy\nCorrelation']
    
    heat = go.Heatmap(z=metrics_data, x=list(range(len(seg_labels))), y=metric_labels, colorscale=heatmap_colorscale,
                      zmin=0, zmax=100, colorbar=dict(title="Score (%)"), hovertemplate="%{y}<br>Segment %{x}<br>Value: %{z:.0f}<extra></extra>")
    
    heat_layout = go.Layout(title="Detailed Metrics Heatmap", xaxis=dict(title="Segment Number", tickmode="array",
                                                                          tickvals=list(range(len(seg_labels))), ticktext=seg_labels),
                            yaxis=dict(autorange="reversed"), height=350, template="plotly_dark")
    
    heat_fig = go.Figure(data=[heat], layout=heat_layout)
    annotations = []
    for i, metric in enumerate(metric_labels):
        for j, seg in enumerate(seg_labels):
            val = metrics_data[i, j]
            if val > 40:
                text_color = "#0E0E10" 
            else:
                text_color = "#F8F8FF" 
            annotations.append(dict(x=j, y=metric, text=f"{val:.0f}", showarrow=False, xanchor="center",
                                    yanchor="middle", font=dict(size=10, color=text_color)))
    heat_fig.update_layout(annotations=annotations)
    st.plotly_chart(heat_fig, use_container_width=True)
    st.markdown("---")
    
    st.subheader("⚠️ 5 Worst Performing Segments")
    worst_segments = sorted(segment_results, key=lambda x: x['composite_score'])[:5]
    for result_s in worst_segments:
        st.markdown(f"**SEGMENT {result_s['segment_num']} ({result_s.get('time_range','N/A')}):**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Score:** {result_s['composite_score']:.1f}/100")
            st.markdown(f"**Rating:** {result_s['rating']}")
        with col2:
            st.markdown(f"**Weaknesses:** {', '.join(result_s['weaknesses'])}")
        st.markdown(f"Model Conf: {result_s.get('model_confidence',0):.1f}% | MFCC: {result_s.get('mfcc_similarity',0):.1f}% | "
                    f"Pitch: {result_s.get('pitch_corr',0):.1f}% | Energy: {result_s.get('energy_corr',0):.1f}%")
        st.markdown("---")
    
    all_weaknesses = []
    for r in segment_results:
        if r.get('weaknesses') and r['weaknesses'] != ["None - well done!"]:
            all_weaknesses.extend(r['weaknesses'])
    
    if all_weaknesses:
        weakness_counts = Counter(all_weaknesses)
        st.subheader("🎯 Most Common Issues")
        for weakness, count in weakness_counts.most_common():
            st.markdown(f"- **{weakness}**: {count} segment(s)")
    else:
        st.success("✅ No significant issues detected!")

# ========================================
# STREAMLIT APP
# ========================================
st.set_page_config(page_title="VocaLift - AI Vocal Coach", page_icon="🎤", layout="wide")
st.markdown("""
<style>
/* Neon Glow Header Styles */

/* Full Song — Neon Green */
h2.full-song {
    color: #39FF14 !important;
    text-shadow: 0 0 10px #39FF14, 0 0 20px #39FF14, 0 0 30px #39FF14;
}

/* Segment — Bright Magenta */
h2.segment {
    color: #FF00FF !important;
    text-shadow: 0 0 10px #FF00FF, 0 0 20px #FF00FF, 0 0 30px #FF00FF;
}

/* Pitch Contour — Neon Yellow */
h2.pitch {
    color: #FFFF33 !important;
    text-shadow: 0 0 10px #FFFF33, 0 0 20px #FFFF33, 0 0 30px #FFFF33;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; width:100%; display:flex; flex-direction:column; align-items:center;">
  <h1 style="
      margin: 0;
      padding: 0;
      font-size: 3rem;
      line-height: 1.05;
      display: inline-block;
      background: linear-gradient(90deg, #39FF14, #00FFFF, #FF00FF);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      text-shadow: 0 0 10px rgba(57,255,20,0.6);
      ">
    🎵 VocaLift - Your AI Vocal Coach 🎵
  </h1>

  <p style="
      margin: 12px 0 0 0;
      padding: 0;
      font-size: 16px;
      line-height: 1.4;
      max-width: 900px;
      color: #F8F8FF;
      ">
    Upload your singing audio and a coach/reference audio to get detailed AI-powered feedback!
  </p>
</div>
""", unsafe_allow_html=True)



with st.sidebar:
    st.header("Model Configuration")
    if os.path.exists(MODEL_PATH):
        st.success("✅ Model File Found")
        model_loaded = True
    else:
        st.error(f"❌ Model file not found")
        model_loaded = False
    if st.session_state.full_song_result is not None or st.session_state.segment_results is not None or st.session_state.pitch_contour_fig is not None:
        st.header("Results:")

    if st.session_state.full_song_result is not None:
        st.sidebar.markdown("""
    <div style='margin-bottom:8px;background:#0F2E0F;border-radius:8px;padding:10px 16px;'>
        <a href="#full-analysis" style='text-decoration:none;color:#39FF14;font-weight:bold;display:flex;align-items:center;'>
            <span style="font-size:1.2em;margin-right:0.5em;">🎵</span>
            Full Song Analysis
        </a>
    </div>""", unsafe_allow_html=True)

    if st.session_state.segment_results is not None:
        st.sidebar.markdown("""
    <div style='margin-bottom:8px;background:#3A0036;border-radius:8px;padding:10px 16px;'>
        <a href="#segment-analysis" style='text-decoration:none;color:#FF2AEF;font-weight:bold;display:flex;align-items:center;'>
            <span style="font-size:1.2em;margin-right:0.5em;">🔬</span>
            Segment Analysis
        </a>
    </div>""", unsafe_allow_html=True)

    if st.session_state.pitch_contour_fig is not None:
        st.sidebar.markdown("""
    <div style='margin-bottom:8px;background:#3A3000;border-radius:8px;padding:10px 16px;'>
        <a href="#pitch-contour" style='text-decoration:none;color:#FFD700;font-weight:bold;display:flex;align-items:center;'>
            <span style="font-size:1.2em;margin-right:0.5em;">🎚️</span>
            Pitch Contour
        </a>
    </div>""", unsafe_allow_html=True)


col1, col2 = st.columns(2)
with col1:
    st.subheader("📁 Coach Audio")
    coach_file = st.file_uploader("Upload coach audio", type=['mp3', 'wav'], key="coach")
    if coach_file:
        st.audio(coach_file)
        st.success(f"✅ Loaded: {coach_file.name}")

with col2:
    st.subheader("🎙️ Your Audio")
    user_file = st.file_uploader("Upload your singing", type=['mp3', 'wav'], key="user")
    if user_file:
        st.audio(user_file)
        st.success(f"✅ Loaded: {user_file.name}")

st.markdown("---")

if coach_file and user_file and model_loaded:
    disable_buttons = st.session_state.is_processing
    
    col1, col2, col3 = st.columns(3)
    with col1:
        analyze_full = st.button("🎵 Get Full Song Analysis", type="secondary", use_container_width=True, disabled=disable_buttons)
    with col2:
        analyze_segments_button = st.button("🔬 Get Segment Analysis", type="secondary", use_container_width=True, disabled=disable_buttons)
    with col3:
        analyze_pitch = st.button("🎚️ Get Pitch Contour", type="secondary", use_container_width=True, disabled=disable_buttons)
    
    if (analyze_full or analyze_segments_button or analyze_pitch) and not st.session_state.is_processing:
        st.session_state.is_processing = True
        st.session_state.analysis_to_run = "full" if analyze_full else "segments" if analyze_segments_button else "pitch"
        st.rerun()
    
    if st.session_state.is_processing:
        st.info("Please wait, analysis is running...")
        if st.session_state.analysis_to_run == "pitch":
            with st.spinner("Generating Pitch Contours..."):
                try:
                    coach_temp, user_temp = save_uploaded_files(coach_file, user_file)
                    pitch_fig = make_pitch_contour_figure(coach_temp, user_temp, sr=SR, hop_length=512)
                    st.session_state.pitch_contour_fig = pitch_fig
                except Exception as e:
                    st.session_state.pitch_contour_fig = None
                    st.warning(f"Could not generate pitch contour plot: {e}")
        
        elif st.session_state.analysis_to_run == "full":
            with st.spinner("Processing full song..."):
                try:
                    coach_temp, user_temp = save_uploaded_files(coach_file, user_file)
                    model = load_model(MODEL_PATH, compile=False)
                    result = generate_feedback(coach_temp, user_temp, model)
                    st.session_state.full_song_result = result
                    st.session_state.analysis_audio_files = {'coach_temp': coach_temp, 'user_temp': user_temp}
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif st.session_state.analysis_to_run == "segments":
            with st.spinner("Processing segments..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                def progress_callback(current, total):
                    status_text.text(f"Processing segment {current}/{total}...")
                    progress_bar.progress(current / total)
                try:
                    coach_temp, user_temp = save_uploaded_files(coach_file, user_file)
                    model = load_model(MODEL_PATH, compile=False)
                    segment_results = analyze_segments(coach_temp, user_temp, model, progress_callback=progress_callback)
                    st.session_state.segment_results = segment_results
                    st.session_state.analysis_audio_files = {'coach_temp': coach_temp, 'user_temp': user_temp}
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    status_text.empty()
                    progress_bar.empty()
        st.session_state.is_processing = False
        st.rerun()
    st.markdown("---")
    
    if st.session_state.full_song_result is not None:
        st.markdown('<a id="full-analysis"></a>', unsafe_allow_html=True)
        show_full_song_analysis(st.session_state.full_song_result)
        st.markdown("---")
    
    if st.session_state.segment_results is not None:
        st.markdown('<a id="segment-analysis"></a>', unsafe_allow_html=True)
        st.markdown('<h2 class="segment">🔬 Segment Analysis Results</h2>', unsafe_allow_html=True)
        show_segment_analysis(st.session_state.segment_results)
        st.markdown("---")
    
    if st.session_state.pitch_contour_fig is not None:
        st.markdown('<a id="pitch-contour"></a>', unsafe_allow_html=True)
        st.markdown('<h2 class="pitch">🎚️ Pitch Contour Comparison</h2>', unsafe_allow_html=True)
        st.plotly_chart(st.session_state.pitch_contour_fig, use_container_width=True)
    
    if st.session_state.full_song_result is not None or st.session_state.segment_results is not None or st.session_state.pitch_contour_fig is not None:
        if st.button("🧹 Clear All Results"):
            st.session_state.full_song_result = None
            st.session_state.segment_results = None
            st.session_state.pitch_contour_fig = None
            st.session_state.analysis_to_run = None
            for key in ['coach_temp', 'user_temp']:
                if key in st.session_state.analysis_audio_files:
                    try:
                        os.remove(st.session_state.analysis_audio_files[key])
                    except:
                        pass
            st.session_state.analysis_audio_files = {}
            st.rerun()

elif not model_loaded:
    st.warning("⚠️ Model file not found!")
else:
    st.info("👆 Upload both audio files to start")

st.markdown("---")
st.markdown("<div style='text-align: center;'><p><strong>VocaLift</strong> - Your AI Vocal Coach | Built with Streamlit & TensorFlow</p></div>", unsafe_allow_html=True)
