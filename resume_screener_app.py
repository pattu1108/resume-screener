import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .hero {
        background-color: #1A5276;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 2.2rem; font-weight: 700; margin: 0; color: white; }
    .hero p  { font-size: 1rem; opacity: 0.85; margin: 0.5rem 0 0; }
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E8EDF2;
        margin-bottom: 1rem;
    }
    .card h3 { color: #1A5276; font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem; }
    .score-box {
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .score-num { font-size: 4rem; font-weight: 700; line-height: 1; }
    .score-label { font-size: 1rem; font-weight: 500; margin-top: 0.5rem; }
    .keyword-hit {
        display: inline-block;
        background: #D5F5E3;
        color: #1E8449;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 3px;
    }
    .keyword-miss {
        display: inline-block;
        background: #FADBD8;
        color: #C0392B;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 3px;
    }
    .tip-box {
        background: #EBF5FB;
        border-left: 4px solid #2471A3;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #1A5276;
    }
    .footer {
        text-align: center;
        color: #7F8C8D;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E8EDF2;
    }
</style>
""", unsafe_allow_html=True)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_match_score(jd, resume):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform([clean_text(jd), clean_text(resume)])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 1)

def extract_keywords(text, top_n=30):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=top_n)
    vectorizer.fit([clean_text(text)])
    return set(vectorizer.get_feature_names_out())

def keyword_analysis(jd, resume):
    jd_keywords = extract_keywords(jd, 40)
    resume_clean = clean_text(resume)
    hits = [kw for kw in jd_keywords if kw in resume_clean]
    misses = [kw for kw in jd_keywords if kw not in resume_clean]
    return hits[:20], misses[:15]

def score_color(score):
    if score >= 70:
        return "#1E8449", "#D5F5E3", "Strong Match!"
    if score >= 45:
        return "#B7950B", "#FEF9E7", "Moderate Match"
    return "#C0392B", "#FADBD8", "Needs Improvement"

def get_tips(score, misses):
    tips = []
    if score < 70:
        tips.append("Add more keywords from the job description directly into your resume.")
    if misses:
        top = ', '.join(misses[:5])
        tips.append("Missing keywords to add to your resume: " + top)
    if score >= 70:
        tips.append("Great match! Make sure your resume is ATS-friendly (avoid tables and columns).")
        tips.append("Quantify your achievements - add numbers like '25% improvement' or '10k records processed'.")
    tips.append("Tailor your resume summary to mirror the exact language used in the job description.")
    return tips

def gauge_chart(score):
    color, _, _ = score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': '%', 'font': {'size': 36, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#AAB7C4"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0,  45], 'color': '#FADBD8'},
                {'range': [45, 70], 'color': '#FEF9E7'},
                {'range': [70, 100], 'color': '#D5F5E3'},
            ],
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor='white'
    )
    return fig


# Hero section
st.markdown("""
<div class="hero">
  <h1>AI Resume Screener</h1>
  <p>Paste any Job Description + your Resume and get an instant AI match score with keyword insights</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card"><h3>Job Description</h3>', unsafe_allow_html=True)
    jd_text = st.text_area(
        label="Job Description",
        placeholder="Paste the full job description here...\n\nExample:\nWe are looking for a Data Scientist with Python, Machine Learning, NLP, Scikit-learn, Power BI and SQL experience...",
        height=280,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card"><h3>Your Resume</h3>', unsafe_allow_html=True)
    resume_text = st.text_area(
        label="Resume",
        placeholder="Paste your resume text here...\n\nTip: Open your resume, press Ctrl+A to select all, then Ctrl+C to copy and paste here.",
        height=280,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyse = st.button("Analyse My Resume")

# Results
if analyse:
    if not jd_text.strip() or not resume_text.strip():
        st.warning("Please paste both the Job Description and your Resume to continue.")
    else:
        with st.spinner("Analysing your resume with AI..."):
            score = get_match_score(jd_text, resume_text)
            hits, misses = keyword_analysis(jd_text, resume_text)
            tips = get_tips(score, misses)
            color, bg, label = score_color(score)

        st.markdown("---")
        st.markdown("### Your Results")

        r1, r2, r3 = st.columns([1, 1.2, 1])

        with r1:
            st.markdown(
                '<div class="score-box" style="background:' + bg + ';">'
                '<div class="score-num" style="color:' + color + ';">' + str(score) + '%</div>'
                '<div class="score-label" style="color:' + color + ';">' + label + '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with r2:
            st.plotly_chart(gauge_chart(score), use_container_width=True)

        with r3:
            st.markdown(
                '<div class="card" style="margin-top:0;">'
                '<h3>Quick Stats</h3>'
                '<p style="font-size:0.9rem; margin:4px 0;"><b style="color:#1E8449;">Keywords matched: ' + str(len(hits)) + '</b></p>'
                '<p style="font-size:0.9rem; margin:4px 0;"><b style="color:#C0392B;">Keywords missing: ' + str(len(misses)) + '</b></p>'
                '<p style="font-size:0.9rem; margin:4px 0;">JD length: ' + str(len(jd_text.split())) + ' words</p>'
                '<p style="font-size:0.9rem; margin:4px 0;">Resume length: ' + str(len(resume_text.split())) + ' words</p>'
                '</div>',
                unsafe_allow_html=True
            )

        k1, k2 = st.columns(2)
        with k1:
            st.markdown('<div class="card"><h3>Matched Keywords</h3>', unsafe_allow_html=True)
            if hits:
                st.markdown(' '.join(['<span class="keyword-hit">' + k + '</span>' for k in hits]), unsafe_allow_html=True)
            else:
                st.write("No strong keyword matches found.")
            st.markdown('</div>', unsafe_allow_html=True)

        with k2:
            st.markdown('<div class="card"><h3>Missing Keywords - Add These!</h3>', unsafe_allow_html=True)
            if misses:
                st.markdown(' '.join(['<span class="keyword-miss">' + k + '</span>' for k in misses]), unsafe_allow_html=True)
            else:
                st.write("You are covering most keywords - great job!")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>AI Suggestions to Improve Your Resume</h3>', unsafe_allow_html=True)
        for tip in tips:
            st.markdown('<div class="tip-box">-> ' + tip + '</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="footer">
            Built by Pratiksha Talewadikar &middot; Data Scientist & AI Engineer &middot;
            <a href="https://linkedin.com/in/pratiksha-talewadikar-312523228" target="_blank">LinkedIn</a> &middot;
            <a href="https://github.com/pattu1108" target="_blank">GitHub</a>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 2rem; color: #7F8C8D;">
        <p style="font-size: 1rem;">Paste your Job Description and Resume above, then click Analyse</p>
        <p style="font-size: 0.85rem;">Works with any job from LinkedIn, Naukri, or any company website</p>
    </div>
    """, unsafe_allow_html=True)
