# VocaLift - AI Vocal Coach

**VocaLift** is a Streamlit-powered AI vocal coach that delivers actionable feedback and rich visualizations based on advanced audio analysis and machine learning. By comparing your singing with a reference (coach) audio, VocaLift helps users improve their pitch, timing, dynamics, and overall vocal quality — anytime, anywhere.

***

## Features

- **Full Song Analysis:**  
  Quantitative assessment of your singing compared to a coach track using metrics like pitch correlation, MFCC similarity, DTW distance, energy correlation, tempo difference, and chroma similarity.  
  Provides model confidence scores and a personalized overall assessment.

- **Segment-by-Segment Feedback:**  
  Breaks down your performance into overlapping segments, providing detailed, time-localized scoring and pinpointing weaknesses in pitch, timing, dynamics, or tempo.

- **AI-Generated Coaching (Gemini API):**  
  Leverages generative AI (Google Gemini) for tailored, structured feedback. The application automatically rotates API keys and model endpoints to handle rate limits seamlessly.

- **Pitch Contour Visualization:**  
  Plots detailed pitch contours for coach and user tracks with voiced region shading, leveraging a neon arcade-style theme for readability and aesthetics.

- **Interactive Streamlit UI:**  
  Upload audio files, select analysis types, and explore results with Plotly-powered charts and metrics. Includes customizable dark theme with neon green highlights, magenta accents, and Orbitron font for a modern look.

***

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AI-Mentorship/VocaLift.git
   cd vocalift
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   - Key Python packages:
     - `streamlit`
     - `tensorflow`
     - `librosa`
     - `numpy`
     - `plotly`
     - `scipy`
     - `scikit-learn`
     - `google-generativeai`
     - `python-dotenv`
     - `joblib`

3. **Set up Google Gemini API Keys:**
   - Place your keys in a `.env` file as:  
     ```
     GEMINI_API_KEY1=your-key-1
     GEMINI_API_KEY2=your-key-2
     . . .
     ```
***

## Usage

1. **Start the app:**
   ```bash
   streamlit run vocalift_streamlit_app.py
   ```

2. **Upload audio:**
   - `Coach/Reference` audio: Professional or reference singing track.
   - `User` audio: Your singing for analysis.

3. **Select analysis type:**
   - Full Song Analysis
   - Segment-by-Segment Analysis
   - Pitch Contour Visualization
   - Complete Analysis (all modes)

4. **Run analysis:**
   - Get instant feedback and interactive charts for overall and segment-specific performance.

5. **Explore AI feedback:**
   - Receive strengths, metric analysis, and recommended next steps powered by Gemini.

***

## File Structure

- `vocalift_streamlit_app.py` : Main Streamlit app interface and user interaction.
- `vocalift_utils.py` : Core audio feature extraction, comparison logic, Gemini API integration, and feedback generation.
- `model/vocaliftmodel_3.h5` : Trained TensorFlow model.

***

## Requirements

- Python 3.8+
- Google Gemini API access
- Supported audio formats: mp3, wav, m4a, flac, aiff, ogg

***

## Development Notes

- Modular utility functions for audio segmentation, feature extraction, alignment, and feedback.
- Neon arcade theme optimized for visual accessibility.
- Automatic temp file management and result clearing buttons in UI.
- API/model rotation logic for robust Gemini integration under heavy usage.

***

## Troubleshooting

- **Model not loading:**  
  - Ensure the `.h5` file is present and compatible with TensorFlow Keras.
- **Gemini feedback not available:**  
  - Check `.env` for valid/active API keys and internet connectivity.
- **Audio error:**  
  - Verify files are correct format and not corrupted.

***

**VocaLift — Your AI Vocal Coach, Anytime, Anywhere.**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/127719437/085084ed-4f2c-4049-bebd-ca75e28515a5/vocalift_utils.py)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/127719437/50814e01-6fa6-412d-96eb-435d167c0f08/vocalift_streamlit_app.py)
