
import streamlit as st
import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from collections import Counter
from tensorflow.keras.models import load_model

# Import all utility functions from vocalift_utils
from vocalift_utils import (
    SR,
    generate_feedback,
    analyze_segments,
    save_uploaded_files,
    make_pitch_contour_figure,
    compute_soft_label,
    get_gemini_feedback
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

if "gemini_feedback" not in st.session_state:  # NEW
    st.session_state.gemini_feedback = None

# ========================================
# DISPLAY FUNCTIONS
# ========================================

def show_full_song_analysis(result):
    """Display full song analysis results."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    prob_good = result.get('prob_good', 0) * 100
    overall = result.get('overall_assessment', 'Unknown')
    
    st.subheader("📈 Overall Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Confidence", f"{prob_good:.1f}%")
    with col2:
        st.metric("Overall Assessment", overall)
    
    st.markdown("---")
    st.subheader("📊 Comparison Metrics")
    
    comp = result.get('comp_feats', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MFCC Similarity", f"{comp.get('mfcc_cosine', 0):.2%}")
        st.metric("Pitch Correlation", f"{comp.get('pitch_corr', 0):.2f}")
    with col2:
        st.metric("DTW Distance", f"{comp.get('dtw_mean', 0):.1f}")
        st.metric("Energy Correlation", f"{comp.get('energy_corr', 0):.2f}")
    with col3:
        st.metric("Chroma Similarity", f"{comp.get('chroma_cosine', 0):.2%}")
        st.metric("Tempo Diff", f"{comp.get('tempo_diff', 0):.1f} BPM")
    
    st.markdown('</div>', unsafe_allow_html=True)

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
    heatmap_colorscale = [[0.0, "#0E0E10"], [0.25, "#FF2AEF"], [0.5, "#FFB114"],
                          [0.75, "#FFFF00"], [1.0, "#39FF14"]]
    
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
    st.subheader("📈 Segment Scores Over Time")
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=seg_labels, y=scores, marker_color=bar_color, text=[f"{s:.1f}" for s in scores],
        textposition='outside', hovertext=hover_texts, hoverinfo='text', name='Segment Score'
    ))
    
    fig_bar.add_trace(go.Scatter(
        x=seg_labels, y=[avg_score]*len(seg_labels), mode='lines',
        name=f'Average ({avg_score:.1f})', line=dict(color=avg_line_color, width=2, dash='dash')
    ))
    
    fig_bar.add_trace(go.Scatter(
        x=seg_labels, y=[80]*len(seg_labels), mode='lines',
        name='Excellent (80)', line=dict(color=exc_line_color, width=1, dash='dot')
    ))
    
    fig_bar.add_trace(go.Scatter(
        x=seg_labels, y=[50]*len(seg_labels), mode='lines',
        name='Fair (50)', line=dict(color=fair_line_color, width=1, dash='dot')
    ))
    
    fig_bar.add_trace(go.Scatter(
        x=seg_labels, y=[35]*len(seg_labels), mode='lines',
        name='Poor (35)', line=dict(color=poor_line_color, width=1, dash='dot')
    ))
    
    fig_bar.update_layout(
        title="Composite Score by Segment",
        xaxis_title="Segment", yaxis_title="Score (0-100)",
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=80, b=40),
        template="plotly_dark", height=450,
        plot_bgcolor=bg_color, paper_bgcolor=bg_color, font=dict(color=text_color)
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔥 Heatmap: Metric Performance")
    
    metrics_data = [
        [r.get('mfcc_similarity', 0) for r in segment_results],
        [r.get('pitch_corr', 0) for r in segment_results],
        [r.get('energy_corr', 0) for r in segment_results],
        [100 - min(r.get('dtw_distance', 0), 100) for r in segment_results]
    ]
    metric_names = ['MFCC Sim', 'Pitch Corr', 'Energy Corr', 'DTW (inverted)']
    
    hover_matrix = []
    for i, metric_name in enumerate(metric_names):
        row = []
        for j, seg_num in enumerate(segment_nums):
            val = metrics_data[i][j]
            row.append(f"<b>{metric_name}</b><br>Segment {seg_num}<br>Value: {val:.1f}")
        hover_matrix.append(row)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=metrics_data, x=seg_labels, y=metric_names,
        colorscale=heatmap_colorscale, colorbar=dict(title="Score"),
        hovertext=hover_matrix, hoverinfo='text'
    ))
    
    fig_heatmap.update_layout(
        title="Heatmap: Metrics Across Segments",
        xaxis_title="Segment", yaxis_title="Metric",
        margin=dict(l=80, r=20, t=60, b=40),
        template="plotly_dark", height=350,
        plot_bgcolor=bg_color, paper_bgcolor=bg_color, font=dict(color=text_color)
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🧩 Detailed Segment Breakdown")
    
    for seg in segment_results:
        with st.expander(f"Segment {seg['segment_num']} - {seg['rating']} ({seg['composite_score']:.1f}/100)"):
            st.markdown(f"**Time Range**: {seg.get('time_range', 'N/A')}")
            st.markdown(f"**Weaknesses**: {', '.join(seg.get('weaknesses', ['None']))}")
            
            seg_col1, seg_col2, seg_col3 = st.columns(3)
            with seg_col1:
                st.metric("Model Confidence", f"{seg.get('model_confidence', 0):.1f}%")
                st.metric("MFCC Similarity", f"{seg.get('mfcc_similarity', 0):.1f}%")
            with seg_col2:
                st.metric("Pitch Correlation", f"{seg.get('pitch_corr', 0):.1f}")
                st.metric("Energy Correlation", f"{seg.get('energy_corr', 0):.1f}")
            with seg_col3:
                st.metric("DTW Distance", f"{seg.get('dtw_distance', 0):.1f}")
                st.metric("Tempo Diff", f"{seg.get('tempo_diff', 0):.1f} BPM")

def show_gemini_feedback(gemini_result, comp_feats=None):
    """Display AI-generated coaching feedback from Gemini."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 AI Vocal Coach Feedback")
    
    if gemini_result.get('status') == 'error':
        st.error(f"❌ Error generating AI feedback: {gemini_result.get('error', 'Unknown error')}")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Check if we have parsed feedback or raw response
    if 'feedback' in gemini_result:
        feedback = gemini_result['feedback']
        
        # Display strengths
        if 'strengths' in feedback:
            st.markdown("### ✨ Your Strengths")
            st.success(feedback['strengths'])
        
        # Display metrics with feedback
        if 'metrics' in feedback:
            st.markdown("### 📊 Detailed Metric Analysis")
            
            metrics = feedback['metrics']
            
            metric_items = list(metrics.items())
            half = (len(metric_items) + 1) // 2
            
            metrics_map = {
                "mfcc_similarity": ("mfcc_cosine", "{:.2%}"),
                "chroma_similarity": ("chroma_cosine", "{:.2%}"),
                "pitch_correlation": ("pitch_corr", "{:.2f}"),
                "energy_correlation": ("energy_corr", "{:.2f}"),
                "dtw_distance": ("dtw_mean", "{:.1f}"),
                "tempo_diff": ("tempo_diff", "{:.1f} BPM"),
            }

            col1, col2 = st.columns(2)
            # First half
            with col1:
                for metric_key, metric_data in metric_items[:half]:
                    with st.container():
                        comp_key, fmt = metrics_map.get(metric_key, (None, None))
                        if comp_key and comp_feats and comp_key in comp_feats:
                            actual_value = fmt.format(comp_feats[comp_key])
                        else:
                            actual_value = "N/A"
                        st.markdown(f"**{metric_data.get('label', metric_key)}**: {actual_value}")
                        st.caption(metric_data.get('meaning', ''))
                        if metric_data.get('feedback'):
                            st.info(metric_data['feedback'])
                        st.markdown("---")

            # Second half
            with col2:
                for metric_key, metric_data in metric_items[half:]:
                    with st.container():
                        comp_key, fmt = metrics_map.get(metric_key, (None, None))
                        if comp_key and comp_feats and comp_key in comp_feats:
                            actual_value = fmt.format(comp_feats[comp_key])
                        else:
                            actual_value = "N/A"
                        st.markdown(f"**{metric_data.get('label', metric_key)}**: {actual_value}")
                        st.caption(metric_data.get('meaning', ''))
                        if metric_data.get('feedback'):
                            st.info(metric_data['feedback'])
                        st.markdown("---")

        
        # Display next steps
        if 'next_steps' in feedback:
            st.markdown("### 🎯 Practice Recommendations")
            st.warning(feedback['next_steps'])
    
    elif 'raw_response' in gemini_result:
        # Fallback: display raw response if JSON parsing failed
        st.warning("⚠️ AI response received but couldn't be fully parsed. Here's the raw feedback:")
        st.text(gemini_result['raw_response'])
        if 'parse_error' in gemini_result:
            st.caption(f"Parse error: {gemini_result['parse_error']}")
    
    else:
        st.info("No feedback data available")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========================================
# MODEL LOADING
# ========================================

@st.cache_resource
def load_vocalift_model():
    """Load the trained VocaLift model (cached)."""
    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# ========================================
# MAIN APP
# ========================================

def main():
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
    🎵 VocaLift 🎵</h1>

  <p style="
      margin: 12px auto 0 auto;
      padding: 0;
      font-size: 16px;
      line-height: 1.4;
      max-width: 900px;
      color: #F8F8FF;
      text-align: center;
      ">
    Your AI Vocal Coach, Anytime, Anywhere
  </p>
</div>
""", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Model Configuration")
        model = load_vocalift_model()

        if model is None:
            st.error("❌ Model could not be loaded. Please check the model path.")
            return
        
        st.success("✅ VocaLift model loaded successfully!")

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
            
        if (
            st.session_state.full_song_result is not None or
            st.session_state.segment_results is not None or
            st.session_state.pitch_contour_fig is not None or
            st.session_state.gemini_feedback is not None
        ):
            st.markdown(
                """
                <div style='
                    position: fixed;
                    left: 0;
                    bottom: 0;
                    width: 21rem;  /* Streamlit's sidebar default width, change if you customize sidebar width */
                    z-index: 9999;
                    background: rgba(20,24,20,0.97);
                    border-top: 2px solid #39FF14;
                    padding: 16px 16px 24px 16px;
                    text-align: center;
                    box-shadow: 0 -2px 12px #202023;
                '>
                    <span style='font-weight: bold; color: #39FF14; font-size: 1rem;'>
                        Need a fresh start?
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            # The button visually appears on top of Streamlit, but logically is at the bottom
            if st.button(
                "Clear Results",
                help="Remove all analysis results and start fresh.",
                use_container_width=True,
                key="clear_results_btn"
            ):
                st.session_state.full_song_result = None
                st.session_state.segment_results = None
                st.session_state.pitch_contour_fig = None
                st.session_state.gemini_feedback = None
                st.session_state.analysis_audio_files = {}
                st.session_state.analysis_to_run = None
                st.rerun()
    
    
    # File upload section
    st.markdown("---")
    st.header("📁 Upload Audio Files")
    
    col1, col2 = st.columns(2)
    with col1:
        coach_file = st.file_uploader(
            "🎵 Coach/Reference Audio",
            type=['mp3', 'wav', 'm4a', 'flac', 'aiff', 'ogg'],
            help="Upload the reference/coach singing audio"
        )
    with col2:
        user_file = st.file_uploader(
            "🎤 Your Singing Audio",
            type=['mp3', 'wav', 'm4a', 'flac', 'aiff', 'ogg'],
            help="Upload your singing audio to analyze"
        )
    
    if coach_file and user_file:
        st.success("✅ Both audio files uploaded!")
        
        # Audio playback
        st.markdown("---")
        st.header("🔊 Audio Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Coach Audio")
            st.audio(coach_file, format='audio/wav')
        with col2:
            st.subheader("Your Audio")
            st.audio(user_file, format='audio/wav')
        
        st.markdown("---")
        st.header("🔬 Analysis Options")
        
        analysis_type = st.radio(
            "Choose Analysis Type:",
            ["Full Song Analysis", "Segment-by-Segment Analysis", "Pitch Contour Visualization", "Complete Analysis (All)"],
            help="Select what type of analysis you want to perform"
        )
        
        analyze_button = st.button("Run Analysis", type="primary", use_container_width=True)
        
        if analyze_button:
            st.session_state.is_processing = True
            st.session_state.analysis_audio_files = {"coach": coach_file, "user": user_file}
            
            if analysis_type == "Full Song Analysis":
                st.session_state.analysis_to_run = "full"
            elif analysis_type == "Segment-by-Segment Analysis":
                st.session_state.analysis_to_run = "segments"
            elif analysis_type == "Pitch Contour Visualization":
                st.session_state.analysis_to_run = "pitch"
            else:
                st.session_state.analysis_to_run = "complete"
            
            st.rerun()
    
    # Process analysis if triggered
    if st.session_state.is_processing and st.session_state.analysis_to_run:
        coach_file = st.session_state.analysis_audio_files.get("coach")
        user_file = st.session_state.analysis_audio_files.get("user")
        
        if not coach_file or not user_file:
            st.error("Audio files not found in session state")
            st.session_state.is_processing = False
            return
        
        try:
            with st.spinner("🔄 Processing audio files..."):
                coach_path, user_path = save_uploaded_files(coach_file, user_file)
            
            if st.session_state.analysis_to_run == "full":
                with st.spinner("🧠 Running full song analysis..."):
                    result = generate_feedback(coach_path, user_path, model, sr=SR)
                    st.session_state.full_song_result = result
                    
                    # NEW: Generate Gemini feedback
                    with st.spinner("🤖 Generating AI coach feedback..."):
                        gemini_result = get_gemini_feedback(result)
                        st.session_state.gemini_feedback = gemini_result
                
                st.success("✅ Full song analysis complete!")
            
            elif st.session_state.analysis_to_run == "segments":
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing segment {current}/{total}...")
                
                with st.spinner("🧠 Running segment analysis..."):
                    segment_results = analyze_segments(
                        coach_path, user_path, model, 
                        progress_callback=progress_callback, sr=SR
                    )
                    st.session_state.segment_results = segment_results
                
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Segment analysis complete!")
            
            elif st.session_state.analysis_to_run == "pitch":
                with st.spinner("🎵 Generating pitch contour visualization..."):
                    fig = make_pitch_contour_figure(coach_path, user_path, sr=SR)
                    st.session_state.pitch_contour_fig = fig
                
                st.success("✅ Pitch contour visualization complete!")
            
            elif st.session_state.analysis_to_run == "complete":
                with st.spinner("🧠 Running full song analysis..."):
                    result = generate_feedback(coach_path, user_path, model, sr=SR)
                    st.session_state.full_song_result = result
                    
                    with st.spinner("🤖 Generating AI coach feedback..."):
                        gemini_result = get_gemini_feedback(result)
                        st.session_state.gemini_feedback = gemini_result
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing segment {current}/{total}...")
                
                with st.spinner("🧠 Running segment analysis..."):
                    segment_results = analyze_segments(
                        coach_path, user_path, model,
                        progress_callback=progress_callback, sr=SR
                    )
                    st.session_state.segment_results = segment_results
                
                progress_bar.empty()
                status_text.empty()
                
                with st.spinner("🎵 Generating pitch contour..."):
                    fig = make_pitch_contour_figure(coach_path, user_path, sr=SR)
                    st.session_state.pitch_contour_fig = fig
                
                st.success("✅ Complete analysis finished!")
            
            # Clean up temp files
            try:
                if os.path.exists(coach_path):
                    os.remove(coach_path)
                if os.path.exists(user_path):
                    os.remove(user_path)
            except Exception as e:
                st.warning(f"Could not remove temp files: {e}")
            
            st.session_state.is_processing = False
            st.session_state.analysis_to_run = None
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state.is_processing = False
            st.session_state.analysis_to_run = None
    
    # Display results
    st.markdown("---")
    
    if st.session_state.full_song_result:
        st.markdown('<a id="full-analysis"></a>', unsafe_allow_html=True)
        st.markdown('<h2 class="full-song">📊 Full Song Analysis Results</h2>', unsafe_allow_html=True)
        show_full_song_analysis(st.session_state.full_song_result)
        
        if st.session_state.gemini_feedback:
            st.markdown("---")
            show_gemini_feedback(st.session_state.gemini_feedback, st.session_state.full_song_result["comp_feats"])
    
    if st.session_state.segment_results:
        st.markdown("---")
        st.markdown('<a id="segment-analysis"></a>', unsafe_allow_html=True)
        st.markdown('<h2 class="segment">🔬 Segment Analysis Results</h2>', unsafe_allow_html=True)
        show_segment_analysis(st.session_state.segment_results)
    
    if st.session_state.pitch_contour_fig:
        st.markdown("---")
        st.markdown('<a id="pitch-contour"></a>', unsafe_allow_html=True)
        st.markdown('<h2 class="pitch">🎚️ Pitch Contour Comparison</h2>', unsafe_allow_html=True)
        st.plotly_chart(st.session_state.pitch_contour_fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #777;'>
        <p><strong>VocaLift</strong> - Your AI Vocal Coach | Built with Streamlit & TensorFlow</p>
        <p>Powered by advanced audio analysis and machine learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()