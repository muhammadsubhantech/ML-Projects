import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime
import base64

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Subhan AI Placement Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_files():
    model = joblib.load("best_model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, encoders

model, encoders = load_files()

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

*{
font-family:'Poppins',sans-serif;
}

html,
body,
.stApp{

background:linear-gradient(
135deg,
#020617,
#0F172A,
#1E293B,
#1E3A8A,
#2563EB);

background-size:500% 500%;

animation:bgMove 18s ease infinite;

color:white;

}

@keyframes bgMove{

0%{
background-position:0% 50%;
}

50%{
background-position:100% 50%;
}

100%{
background-position:0% 50%;
}

}

.main>.block-container{

padding-top:30px;
padding-left:40px;
padding-right:40px;
padding-bottom:30px;

}

section[data-testid="stSidebar"]{

background:rgba(15,23,42,.96);

border-right:1px solid rgba(255,255,255,.08);

}

::-webkit-scrollbar{

width:10px;

}

::-webkit-scrollbar-thumb{

background:#3B82F6;
border-radius:30px;

}

::-webkit-scrollbar-track{

background:#111827;

}

.glass{

background:rgba(255,255,255,.08);

backdrop-filter:blur(20px);

border:1px solid rgba(255,255,255,.12);

border-radius:25px;

padding:25px;

box-shadow:0 10px 40px rgba(0,0,0,.35);

margin-bottom:25px;

}

.hero{

background:linear-gradient(
135deg,
rgba(37,99,235,.35),
rgba(124,58,237,.35));

border-radius:30px;

padding:45px;

backdrop-filter:blur(25px);

border:1px solid rgba(255,255,255,.10);

box-shadow:0 15px 50px rgba(0,0,0,.45);

margin-bottom:30px;

overflow:hidden;

position:relative;

}

.hero::before{

content:"";

position:absolute;

width:350px;

height:350px;

background:rgba(255,255,255,.08);

border-radius:50%;

top:-150px;

right:-120px;

}

.hero-title{

font-size:52px;

font-weight:800;

color:white;

margin-bottom:8px;

}

.hero-sub{

font-size:20px;

color:#E2E8F0;

margin-bottom:30px;

}

.badge{

display:inline-block;

padding:10px 18px;

border-radius:30px;

background:rgba(255,255,255,.12);

margin-right:10px;

margin-top:10px;

font-size:15px;

font-weight:600;

}

.metric-card{

background:rgba(255,255,255,.08);

border-radius:22px;

padding:25px;

text-align:center;

border:1px solid rgba(255,255,255,.10);

transition:.35s;

}

.metric-card:hover{

transform:translateY(-8px);

box-shadow:0 15px 35px rgba(0,0,0,.40);

}

.metric-title{

font-size:15px;

color:#CBD5E1;

}

.metric-value{

font-size:34px;

font-weight:700;

color:white;

margin-top:8px;

}

.stButton>button{

width:100%;

height:58px;

border:none;

border-radius:18px;

font-size:18px;

font-weight:700;

color:white;

background:linear-gradient(
90deg,
#2563EB,
#7C3AED);

transition:.35s;

}

.stButton>button:hover{

transform:scale(1.03);

box-shadow:0 10px 30px rgba(37,99,235,.45);

}

div[data-baseweb="select"]>div{

background:#0F172A;

color:white;

border-radius:12px;

}

.stNumberInput input{

background:#0F172A;

color:white;

}

.stSlider{

padding-top:15px;

padding-bottom:15px;

}

.footer{

text-align:center;

padding:30px;

color:#CBD5E1;

font-size:15px;

}

</style>

""",unsafe_allow_html=True)
# ==========================================================
# HERO SECTION
# ==========================================================

current_time = datetime.now().strftime("%I:%M %p")
current_date = datetime.now().strftime("%d %B %Y")

st.markdown(f"""
<div class="hero">

<div class="hero-title">
🧠 Subhan AI Placement Intelligence Platform
</div>

<div class="hero-sub">
Professional AI Powered Student Placement Prediction Dashboard
</div>

<span class="badge">🤖 AI Engine Online</span>
<span class="badge">⚡ Random Forest Active</span>
<span class="badge">📊 Analytics Ready</span>
<span class="badge">🚀 Version 3.0</span>

<br><br>

</div>

""", unsafe_allow_html=True)

# ==========================================================
# TOP DASHBOARD
# ==========================================================

left, right = st.columns([3,1])

with left:

    st.markdown(f"""
    <div class="glass">

    <h2>👋 Welcome, Subhan</h2>

    <p style="font-size:18px;color:#CBD5E1;">

    Artificial Intelligence Engineer

    <br><br>

    This dashboard uses Machine Learning to predict
    student placement probability and generate
    intelligent career recommendations.

    </p>

    </div>

    """, unsafe_allow_html=True)

with right:

    st.markdown(f"""
    <div class="glass">

    <h3 align="center">👨‍💻 Developer</h3>

    <hr>

    <b>Name</b><br>
    Subhan

    <br><br>

    <b>Department</b><br>
    AI Engineering

    <br><br>

    <b>Date</b><br>
    {current_date}

    <br><br>

    <b>Time</b><br>
    {current_time}

    <br><br>

    <b>Status</b><br>

    🟢 Online

    </div>

    """, unsafe_allow_html=True)

# ==========================================================
# KPI CARDS
# ==========================================================

st.markdown("## 📊 AI Dashboard")

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown("""
    <div class="metric-card">

    <div class="metric-title">
    AI Model
    </div>

    <div class="metric-value">
    Random Forest
    </div>

    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown("""
    <div class="metric-card">

    <div class="metric-title">
    Prediction
    </div>

    <div class="metric-value">
    Real-Time
    </div>

    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown("""
    <div class="metric-card">

    <div class="metric-title">
    Analytics
    </div>

    <div class="metric-value">
    Live
    </div>

    </div>
    """, unsafe_allow_html=True)

with c4:

    st.markdown("""
    <div class="metric-card">

    <div class="metric-title">
    AI Status
    </div>

    <div class="metric-value">
    Ready ✅
    </div>

    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR HEADER
# ==========================================================

st.sidebar.markdown("""
# 🎓 Student Profile
""")

st.sidebar.markdown("---")

st.sidebar.success("🟢 AI Engine Connected")

st.sidebar.info("""
Fill all student information carefully.

The prediction quality depends on the accuracy of the provided data.
""")

st.sidebar.markdown("---")

# ==========================================================
# QUICK STATS
# ==========================================================

st.sidebar.markdown("### 📈 Dashboard")

st.sidebar.metric(
    "Model",
    "Random Forest"
)

st.sidebar.metric(
    "Version",
    "3.0"
)

st.sidebar.metric(
    "Developer",
    "Subhan"
)

st.sidebar.metric(
    "Status",
    "Online"
)
# ==========================================================
# STUDENT INPUT DASHBOARD
# ==========================================================

st.markdown("## 🎓 Student Information")

tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Personal",
    "📚 Academic",
    "💻 Technical",
    "🚀 Experience"
])

# ==========================================================
# TAB 1 : PERSONAL
# ==========================================================

with tab1:

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "🎂 Age",
            min_value=18,
            max_value=35,
            value=21
        )

        gender = st.selectbox(
            "🚻 Gender",
            encoders["gender"].classes_.tolist()
        )

        branch = st.selectbox(
            "🎓 Branch",
            encoders["branch"].classes_.tolist()
        )

    with col2:

        college_tier = st.selectbox(
            "🏫 College Tier",
            encoders["college_tier"].classes_.tolist()
        )

        volunteer_experience = st.selectbox(
            "🤝 Volunteer Experience",
            encoders["volunteer_experience"].classes_.tolist()
        )

        sleep_hours = st.slider(
            "😴 Sleep Hours",
            3.0,
            12.0,
            7.0,
            0.5
        )

# ==========================================================
# TAB 2 : ACADEMIC
# ==========================================================

with tab2:

    c1, c2 = st.columns(2)

    with c1:

        cgpa = st.slider(
            "📈 CGPA",
            0.0,
            10.0,
            7.5,
            0.1
        )

        attendance_percentage = st.slider(
            "📅 Attendance %",
            0,
            100,
            85
        )

    with c2:

        study_hours_per_day = st.slider(
            "📖 Study Hours / Day",
            0.0,
            15.0,
            4.0,
            0.5
        )

        backlogs = st.slider(
            "❌ Backlogs",
            0,
            15,
            0
        )

# ==========================================================
# TAB 3 : TECHNICAL
# ==========================================================

with tab3:

    col1, col2 = st.columns(2)

    with col1:

        coding_skill_score = st.slider(
            "💻 Coding",
            0,
            100,
            75
        )

        aptitude_score = st.slider(
            "🧠 Aptitude",
            0,
            100,
            70
        )

        logical_reasoning_score = st.slider(
            "📊 Logical Reasoning",
            0,
            100,
            70
        )

    with col2:

        communication_skill_score = st.slider(
            "🗣 Communication",
            0,
            100,
            75
        )

        mock_interview_score = st.slider(
            "🎤 Mock Interview",
            0,
            100,
            70
        )

# ==========================================================
# TAB 4 : EXPERIENCE
# ==========================================================

with tab4:

    c1, c2 = st.columns(2)

    with c1:

        internships_count = st.slider(
            "🏢 Internships",
            0,
            10,
            2
        )

        projects_count = st.slider(
            "📂 Projects",
            0,
            20,
            5
        )

        certifications_count = st.slider(
            "📜 Certifications",
            0,
            20,
            3
        )

        hackathons_participated = st.slider(
            "🏆 Hackathons",
            0,
            20,
            2
        )

    with c2:

        github_repos = st.slider(
            "🐙 GitHub Repositories",
            0,
            100,
            8
        )

        linkedin_connections = st.slider(
            "🔗 LinkedIn Connections",
            0,
            5000,
            500
        )

        extracurricular_score = st.slider(
            "🎯 Extracurricular",
            0,
            100,
            60
        )

        leadership_score = st.slider(
            "👑 Leadership",
            0,
            100,
            60
        )

# ==========================================================
# LIVE PROFILE SCORE
# ==========================================================

st.markdown("## 📊 Live Student Analysis")

score = (
    cgpa * 10 +
    coding_skill_score +
    communication_skill_score +
    aptitude_score +
    logical_reasoning_score
) / 5

left, right = st.columns([2, 1])

with left:

    st.progress(score / 100)

    if score >= 85:
        st.success(f"🌟 Excellent Profile ({score:.1f}/100)")
    elif score >= 70:
        st.info(f"👍 Good Profile ({score:.1f}/100)")
    elif score >= 50:
        st.warning(f"⚠ Average Profile ({score:.1f}/100)")
    else:
        st.error(f"❌ Needs Improvement ({score:.1f}/100)")

with right:

    st.metric(
        "Profile Score",
        f"{score:.1f}/100"
    )

# ==========================================================
# SKILL OVERVIEW
# ==========================================================

st.markdown("## 🚀 Skill Overview")

a, b, c, d = st.columns(4)

with a:
    st.metric("💻 Coding", f"{coding_skill_score}%")

with b:
    st.metric("🧠 Aptitude", f"{aptitude_score}%")

with c:
    st.metric("🗣 Communication", f"{communication_skill_score}%")

with d:
    st.metric("👑 Leadership", f"{leadership_score}%")

st.markdown("---")

predict = st.button(
    "🚀 Predict Placement",
    use_container_width=True
)
# ==========================================================
# CREATE INPUT DATAFRAME
# ==========================================================

input_df = pd.DataFrame({

    "age":[age],
    "gender":[gender],
    "cgpa":[cgpa],
    "branch":[branch],
    "college_tier":[college_tier],
    "internships_count":[internships_count],
    "projects_count":[projects_count],
    "certifications_count":[certifications_count],
    "coding_skill_score":[coding_skill_score],
    "aptitude_score":[aptitude_score],
    "communication_skill_score":[communication_skill_score],
    "logical_reasoning_score":[logical_reasoning_score],
    "hackathons_participated":[hackathons_participated],
    "github_repos":[github_repos],
    "linkedin_connections":[linkedin_connections],
    "mock_interview_score":[mock_interview_score],
    "attendance_percentage":[attendance_percentage],
    "backlogs":[backlogs],
    "extracurricular_score":[extracurricular_score],
    "leadership_score":[leadership_score],
    "volunteer_experience":[volunteer_experience],
    "sleep_hours":[sleep_hours],
    "study_hours_per_day":[study_hours_per_day]

})

# ==========================================================
# ENCODE CATEGORICAL FEATURES
# ==========================================================

input_df["gender"] = encoders["gender"].transform(input_df["gender"])
input_df["branch"] = encoders["branch"].transform(input_df["branch"])
input_df["college_tier"] = encoders["college_tier"].transform(input_df["college_tier"])
input_df["volunteer_experience"] = encoders["volunteer_experience"].transform(input_df["volunteer_experience"])

# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    # ------------------------------
    # INPUT VALIDATION
    # ------------------------------

    warning_messages = []

    if cgpa < 5 and coding_skill_score > 90:
        warning_messages.append(
            "Very low CGPA with extremely high coding score."
        )

    if internships_count == 0 and projects_count == 0:
        warning_messages.append(
            "No internships and no projects detected."
        )

    if attendance_percentage < 40:
        warning_messages.append(
            "Attendance is extremely low."
        )

    if backlogs > 5:
        warning_messages.append(
            "Large number of backlogs."
        )

    if warning_messages:

        st.warning("### ⚠ Profile Warnings")

        for msg in warning_messages:
            st.write("•", msg)

    # ------------------------------
    # LOADING
    # ------------------------------

    progress = st.progress(0)

    status = st.empty()

    loading = [

        "📚 Reading Student Profile...",
        "🧠 Running AI Engine...",
        "📊 Calculating Placement Probability...",
        "⚡ Random Forest Prediction...",
        "🤖 Generating AI Report...",
        "✅ Prediction Completed"

    ]

    for i in range(100):

        progress.progress(i + 1)

        if i < 15:
            status.info(loading[0])

        elif i < 35:
            status.info(loading[1])

        elif i < 55:
            status.info(loading[2])

        elif i < 80:
            status.info(loading[3])

        elif i < 95:
            status.info(loading[4])

        else:
            status.success(loading[5])

        time.sleep(0.01)

    progress.empty()
    status.empty()

    # ------------------------------
    # MODEL
    # ------------------------------

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    placed_probability = probability[1] * 100

    not_placed_probability = probability[0] * 100

    confidence = np.max(probability) * 100

    # ------------------------------
    # RESULT
    # ------------------------------

    st.markdown("---")

    st.markdown("# 🎯 AI Prediction Result")

    if prediction == 1:

        st.balloons()

        st.success(
            "## 🎉 Congratulations!\n\nThe student is predicted to be **Placed**."
        )

    else:

        st.error(
            "## ❌ Placement Not Predicted"
        )

    # ------------------------------
    # METRICS
    # ------------------------------

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(

            "Placement Probability",

            f"{placed_probability:.2f}%"

        )

    with m2:

        st.metric(

            "Not Placement",

            f"{not_placed_probability:.2f}%"

        )

    with m3:

        st.metric(

            "Model Confidence",

            f"{confidence:.2f}%"

        )

    st.markdown("---")

    # ------------------------------
    # AI PROFILE ANALYSIS
    # ------------------------------

    st.markdown("## 🤖 AI Profile Analysis")

    strengths = []

    improvements = []

    if cgpa >= 8:
        strengths.append("Excellent Academic Performance")
    else:
        improvements.append("Improve CGPA")

    if coding_skill_score >= 80:
        strengths.append("Strong Coding Skills")
    else:
        improvements.append("Practice Coding Daily")

    if communication_skill_score >= 75:
        strengths.append("Excellent Communication")
    else:
        improvements.append("Improve Communication")

    if internships_count >= 2:
        strengths.append("Industry Experience")
    else:
        improvements.append("Complete More Internships")

    if projects_count >= 5:
        strengths.append("Strong Project Portfolio")
    else:
        improvements.append("Build More Projects")

    c1, c2 = st.columns(2)

    with c1:

        st.success("### 💪 Strengths")

        if strengths:
            for s in strengths:
                st.write("✅", s)
        else:
            st.write("No major strengths detected.")

    with c2:

        st.warning("### 📈 Improvements")

        if improvements:
            for s in improvements:
                st.write("🔸", s)
        else:
            st.write("Profile looks excellent.")

    # ------------------------------
    # READINESS SCORE
    # ------------------------------

    readiness = (

        cgpa * 10 +

        coding_skill_score +

        communication_skill_score +

        aptitude_score +

        logical_reasoning_score

    ) / 5

    st.markdown("---")

    st.markdown("## 🚀 Career Readiness")

    st.progress(readiness / 100)

    st.metric(

        "Readiness Score",

        f"{readiness:.1f}/100"

    )

    if readiness >= 85:

        st.success("🌟 Outstanding Career Readiness")

    elif readiness >= 70:

        st.info("👍 Good Career Readiness")

    elif readiness >= 50:

        st.warning("⚠ Average Career Readiness")

    else:

        st.error("❌ Needs Major Improvement")
    # ==========================================================
    # ADVANCED AI ANALYTICS DASHBOARD
    # ==========================================================

    st.markdown("---")
    st.markdown("# 📊 AI Analytics Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Charts",
        "🕸 Skills",
        "📋 Summary",
        "🎯 Grade"
    ])

    # ==========================================================
    # TAB 1
    # ==========================================================

    with tab1:

        col1, col2 = st.columns(2)

        with col1:

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["Placed", "Not Placed"],
                        values=[
                            placed_probability,
                            not_placed_probability
                        ],
                        hole=.72,
                        marker=dict(
                            colors=[
                                "#10B981",
                                "#EF4444"
                            ]
                        )
                    )
                ]
            )

            fig.update_layout(

                title="Placement Probability",

                paper_bgcolor="rgba(0,0,0,0)",

                font_color="white",

                height=430

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = go.Figure()

            fig.add_trace(

                go.Bar(

                    x=[
                        "Placement",
                        "Confidence"
                    ],

                    y=[
                        placed_probability,
                        confidence
                    ],

                    text=[
                        f"{placed_probability:.1f}%",
                        f"{confidence:.1f}%"
                    ],

                    textposition="outside",

                    marker_color=[
                        "#3B82F6",
                        "#7C3AED"
                    ]

                )

            )

            fig.update_layout(

                title="Prediction Analysis",

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(0,0,0,0)",

                font_color="white",

                height=430

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ==========================================================
    # TAB 2
    # ==========================================================

    with tab2:

        radar = go.Figure()

        radar.add_trace(

            go.Scatterpolar(

                r=[

                    coding_skill_score,

                    communication_skill_score,

                    aptitude_score,

                    logical_reasoning_score,

                    leadership_score,

                    extracurricular_score,

                    coding_skill_score

                ],

                theta=[

                    "Coding",

                    "Communication",

                    "Aptitude",

                    "Reasoning",

                    "Leadership",

                    "Activities",

                    "Coding"

                ],

                fill="toself",

                name="Student"

            )

        )

        radar.update_layout(

            polar=dict(

                radialaxis=dict(

                    visible=True,

                    range=[0,100]

                )

            ),

            paper_bgcolor="rgba(0,0,0,0)",

            font_color="white",

            height=600

        )

        st.plotly_chart(
            radar,
            use_container_width=True
        )

    # ==========================================================
    # TAB 3
    # ==========================================================

    with tab3:

        summary = pd.DataFrame({

            "Feature":[

                "CGPA",

                "Coding",

                "Communication",

                "Projects",

                "Internships",

                "Attendance",

                "Leadership",

                "Placement Probability"

            ],

            "Value":[

                cgpa,

                coding_skill_score,

                communication_skill_score,

                projects_count,

                internships_count,

                attendance_percentage,

                leadership_score,

                f"{placed_probability:.2f}%"

            ]

        })

        st.dataframe(

            summary,

            hide_index=True,

            use_container_width=True

        )

    # ==========================================================
    # TAB 4
    # ==========================================================

    with tab4:

        if placed_probability >= 90:

            grade = "A+"

            remark = "Outstanding"

        elif placed_probability >= 80:

            grade = "A"

            remark = "Excellent"

        elif placed_probability >= 70:

            grade = "B"

            remark = "Very Good"

        elif placed_probability >= 60:

            grade = "C"

            remark = "Good"

        else:

            grade = "D"

            remark = "Needs Improvement"

        st.metric(
            "Placement Grade",
            grade
        )

        st.success(
            f"Overall Evaluation : {remark}"
        )

    # ==========================================================
    # PREMIUM GAUGE
    # ==========================================================

    st.markdown("---")

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=placed_probability,

            number={"suffix":"%"},

            title={"text":"Placement Score"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"#2563EB"},

                "steps":[

                    {

                        "range":[0,40],

                        "color":"#EF4444"

                    },

                    {

                        "range":[40,70],

                        "color":"#FACC15"

                    },

                    {

                        "range":[70,100],

                        "color":"#22C55E"

                    }

                ]

            }

        )

    )

    gauge.update_layout(

        paper_bgcolor="rgba(0,0,0,0)",

        font_color="white",

        height=430

    )

    st.plotly_chart(

        gauge,

        use_container_width=True

    )

    # ==========================================================
    # SALARY ESTIMATION
    # ==========================================================

    st.markdown("## 💰 Estimated Starting Salary")

    if placed_probability >= 90:

        salary = "$15,000 - $25,000"

    elif placed_probability >= 80:

        salary = "$10,000 - $15,000"

    elif placed_probability >= 70:

        salary = "$7,000 - $10,000"

    elif placed_probability >= 60:

        salary = "$5,000 - $7,000"

    else:

        salary = "Focus on Skill Development"

    st.info(f"Estimated Career Outcome : **{salary}**")

    # ==========================================================
    # AI STATUS PANEL
    # ==========================================================

    st.markdown("## 🤖 AI Engine Status")

    a,b,c,d = st.columns(4)

    with a:
        st.metric("Model", "Random Forest")

    with b:
        st.metric("Prediction", "Complete")

    with c:
        st.metric("Confidence", f"{confidence:.1f}%")

    with d:
        st.metric("Developer", "Subhan AI")
        # ==========================================================
    # AI CAREER ADVISOR
    # ==========================================================

    st.markdown("---")
    st.markdown("# 🤖 AI Career Advisor")

    career_score = 0

    career_score += cgpa * 5
    career_score += coding_skill_score * 0.30
    career_score += communication_skill_score * 0.15
    career_score += aptitude_score * 0.15
    career_score += logical_reasoning_score * 0.10
    career_score += leadership_score * 0.10
    career_score += extracurricular_score * 0.05
    career_score += internships_count * 2
    career_score += projects_count

    career_score = min(100, career_score)

    st.progress(career_score/100)

    st.metric(
        "Career Score",
        f"{career_score:.1f}/100"
    )

    # ==========================================================
    # AI LEVEL
    # ==========================================================

    if career_score >= 90:

        level = "🌟 Elite Candidate"

    elif career_score >= 80:

        level = "🚀 Excellent Candidate"

    elif career_score >= 70:

        level = "🔥 Strong Candidate"

    elif career_score >= 60:

        level = "👍 Good Candidate"

    else:

        level = "📚 Beginner"

    st.success(level)

    # ==========================================================
    # COMPANY RECOMMENDATION
    # ==========================================================

    st.markdown("## 🏢 Recommended Companies")

    companies = []

    if coding_skill_score >= 90 and cgpa >= 8.5:

        companies += [
            "Google",
            "Microsoft",
            "Amazon",
            "Meta",
            "Apple"
        ]

    elif coding_skill_score >= 80:

        companies += [
            "IBM",
            "Oracle",
            "Intel",
            "Dell",
            "Cisco"
        ]

    elif coding_skill_score >= 70:

        companies += [
            "TCS",
            "Infosys",
            "Capgemini",
            "Wipro",
            "Accenture"
        ]

    else:

        companies += [
            "Startups",
            "Local Software Houses",
            "Internship Programs"
        ]

    for company in companies:

        st.write("✅", company)

    # ==========================================================
    # AI IMPROVEMENT PLAN
    # ==========================================================

    st.markdown("## 📈 Personalized Improvement Plan")

    roadmap = []

    if cgpa < 8:

        roadmap.append("Improve CGPA above 8.0")

    if coding_skill_score < 85:

        roadmap.append("Solve 300+ LeetCode problems")

    if communication_skill_score < 80:

        roadmap.append("Practice public speaking weekly")

    if internships_count < 2:

        roadmap.append("Complete at least 2 internships")

    if github_repos < 15:

        roadmap.append("Upload more GitHub projects")

    if certifications_count < 5:

        roadmap.append("Earn AI / ML Certifications")

    if projects_count < 6:

        roadmap.append("Build real-world projects")

    if leadership_score < 70:

        roadmap.append("Lead a university project")

    if not roadmap:

        st.success("🎉 Your profile is already very strong.")

    else:

        for i,item in enumerate(roadmap,1):

            st.write(f"{i}. {item}")

    # ==========================================================
    # INTERVIEW SUCCESS CHANCE
    # ==========================================================

    st.markdown("## 🎤 Interview Prediction")

    interview_score = (

        communication_skill_score*0.40 +

        aptitude_score*0.20 +

        logical_reasoning_score*0.20 +

        mock_interview_score*0.20

    )

    st.progress(interview_score/100)

    st.metric(
        "Interview Success",
        f"{interview_score:.1f}%"
    )

    if interview_score >= 85:

        st.success("High Interview Success Probability")

    elif interview_score >=70:

        st.info("Good Interview Performance Expected")

    else:

        st.warning("Practice Mock Interviews")

    # ==========================================================
    # AI FINAL DECISION
    # ==========================================================

    st.markdown("## 🧠 Final AI Verdict")

    if placed_probability >= 90:

        verdict = """
    Excellent profile.

    Ready for top product-based companies.

    Keep building advanced AI projects.
    """

    elif placed_probability >= 75:

        verdict = """
    Strong placement chances.

    Improve coding and interview skills
    to maximize opportunities.
    """

    elif placed_probability >=60:

        verdict = """
    Moderate placement chances.

    Focus on projects,
    internships,
    and aptitude preparation.
    """

    else:

        verdict = """
    Placement chances are currently low.

    Build skills,
    improve CGPA,
    gain experience,
    and practice interviews.
    """

    st.info(verdict)
    # ==========================================================
    # AI RESUME SCORE
    # ==========================================================

    st.markdown("---")
    st.markdown("# 📄 AI Resume Analyzer")

    resume_score = 0

    resume_score += min(cgpa * 8, 30)
    resume_score += min(projects_count * 3, 15)
    resume_score += min(internships_count * 5, 20)
    resume_score += min(certifications_count * 2, 10)
    resume_score += min(github_repos * 0.5, 10)
    resume_score += min(linkedin_connections / 100, 5)
    resume_score += communication_skill_score * 0.10
    resume_score += leadership_score * 0.10

    resume_score = min(100, resume_score)

    st.progress(resume_score / 100)

    st.metric(
        "Resume Score",
        f"{resume_score:.1f}/100"
    )

    if resume_score >= 90:

        st.success("🌟 World-Class Resume")

    elif resume_score >= 80:

        st.success("🚀 Excellent Resume")

    elif resume_score >= 70:

        st.info("👍 Good Resume")

    elif resume_score >= 60:

        st.warning("Average Resume")

    else:

        st.error("Needs Improvement")

    # ==========================================================
    # DREAM COMPANY MATCH
    # ==========================================================

    st.markdown("## 🏢 Dream Company Match")

    dream = []

    if coding_skill_score >= 90:

        dream.extend([
            "Google",
            "Microsoft",
            "Meta",
            "Amazon",
            "Apple",
            "Netflix"
        ])

    elif coding_skill_score >= 80:

        dream.extend([
            "Oracle",
            "IBM",
            "Intel",
            "Cisco",
            "Dell"
        ])

    else:

        dream.extend([
            "Software Houses",
            "Growing Startups",
            "Internship Programs"
        ])

    cols = st.columns(3)

    for i, company in enumerate(dream):

        cols[i % 3].success(company)

    # ==========================================================
    # WHAT IF SIMULATOR
    # ==========================================================

    st.markdown("---")
    st.markdown("# 📈 What If Simulator")

    new_cgpa = st.slider(

        "Suppose your CGPA becomes",

        float(cgpa),

        10.0,

        float(cgpa),

        0.1,

        key="future_cgpa"

    )

    difference = (new_cgpa - cgpa) * 6

    future_probability = min(

        100,

        placed_probability + difference

    )

    col1,col2 = st.columns(2)

    with col1:

        st.metric(

            "Current Probability",

            f"{placed_probability:.1f}%"

        )

    with col2:

        st.metric(

            "Future Probability",

            f"{future_probability:.1f}%",

            delta=f"+{future_probability-placed_probability:.1f}%"

        )

    # ==========================================================
    # FEATURE IMPORTANCE
    # ==========================================================

    st.markdown("---")
    st.markdown("# 🧠 Feature Importance")

    importance = pd.DataFrame({

        "Feature":[

            "CGPA",

            "Coding",

            "Communication",

            "Projects",

            "Internships",

            "Aptitude",

            "Reasoning",

            "Leadership"

        ],

        "Importance":[

            cgpa*10,

            coding_skill_score,

            communication_skill_score,

            projects_count*10,

            internships_count*15,

            aptitude_score,

            logical_reasoning_score,

            leadership_score

        ]

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    fig = px.bar(

        importance,

        x="Importance",

        y="Feature",

        orientation="h",

        text="Importance"

    )

    fig.update_layout(

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font_color="white",

        height=500

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ==========================================================
    # ACHIEVEMENT BADGES
    # ==========================================================

    st.markdown("---")
    st.markdown("# 🏅 Achievement Badges")

    badges=[]

    if cgpa>=8.5:
        badges.append("🎓 Academic Star")

    if coding_skill_score>=90:
        badges.append("💻 Coding Master")

    if communication_skill_score>=85:
        badges.append("🎤 Communication Expert")

    if internships_count>=3:
        badges.append("🏢 Industry Ready")

    if github_repos>=20:
        badges.append("🐙 GitHub Champion")

    if leadership_score>=80:
        badges.append("👑 Team Leader")

    if projects_count>=8:
        badges.append("📂 Project Expert")

    if certifications_count>=6:
        badges.append("📜 Certified Professional")

    if len(badges)==0:

        st.info("No badges unlocked yet.")

    else:

        cols=st.columns(4)

        for i,b in enumerate(badges):

            cols[i%4].success(b)

    # ==========================================================
    # AI FINAL REPORT
    # ==========================================================

    st.markdown("---")
    st.markdown("# 🤖 AI Executive Summary")

    summary=f"""
    Student Profile Score : {career_score:.1f}/100

    Resume Score : {resume_score:.1f}/100

    Placement Probability : {placed_probability:.1f}%

    Confidence : {confidence:.1f}%

    Recommended Companies :

    {', '.join(dream)}

    Overall Grade : {grade}

    Developer :

    Subhan
    Artificial Intelligence Engineer

    Generated by
    Subhan AI Placement Intelligence Platform
    """

    st.text_area(

        "Executive Report",

        summary,

        height=300

    )

    # ==========================================================
    # VERIFIED DEVELOPER FOOTER
    # ==========================================================

    st.markdown("""
    <hr>

    <div style="text-align:center;padding:20px;">

    <h3 style="color:#60A5FA;">
    🧠 Subhan AI
    </h3>

    <p style="font-size:18px;">
    Developed with ❤️ by
    <b>Subhan</b>
    </p>

    <p>
    Artificial Intelligence Engineer
    </p>

    <p style="color:gray;">
    © 2026 Subhan AI Placement Intelligence Platform
    </p>

    </div>

    """,unsafe_allow_html=True)
    # ==========================================================
    # PLACEMENT READINESS CERTIFICATE
    # ==========================================================

    st.markdown("---")
    st.markdown("# 🏆 AI Placement Readiness Certificate")

    certificate = f"""
    ═══════════════════════════════════════════════

                SUBHAN AI

    AI Placement Intelligence Platform

    ═══════════════════════════════════════════════

    Certificate of Placement Readiness

    This certificate is awarded to the student
    after successful AI evaluation.

    Placement Probability :
    {placed_probability:.2f} %

    Career Score :
    {career_score:.2f}/100

    Resume Score :
    {resume_score:.2f}/100

    Placement Grade :
    {grade}

    AI Confidence :
    {confidence:.2f} %

    Generated :
    {current_date}

    Verified By :

    Subhan
    Artificial Intelligence Engineer

    ═══════════════════════════════════════════════
    """

    st.code(certificate)

    # ==========================================================
    # DOWNLOAD REPORT
    # ==========================================================

    report = f"""

    SUBHAN AI

    Placement Intelligence Report

    ====================================

    Placement Probability :
    {placed_probability:.2f}

    Confidence :
    {confidence:.2f}

    Career Score :
    {career_score:.2f}

    Resume Score :
    {resume_score:.2f}

    Grade :
    {grade}

    Recommended Companies :

    {",".join(dream)}

    Strengths

    {",".join(strengths)}

    Improvements

    {",".join(improvements)}

    Generated by

    Subhan AI
    """

    st.download_button(

        "📥 Download AI Report",

        report,

        file_name="Subhan_AI_Report.txt",

        mime="text/plain",

        use_container_width=True

    )

    # ==========================================================
    # PERFORMANCE DASHBOARD
    # ==========================================================

    st.markdown("---")
    st.markdown("# 🚀 Performance Dashboard")

    col1,col2,col3,col4=st.columns(4)

    with col1:

        st.metric(

            "Coding",

            f"{coding_skill_score}%"

        )

    with col2:

        st.metric(

            "Communication",

            f"{communication_skill_score}%"

        )

    with col3:

        st.metric(

            "Leadership",

            f"{leadership_score}%"

        )

    with col4:

        st.metric(

            "Aptitude",

            f"{aptitude_score}%"

        )

    # ==========================================================
    # SKILL PROGRESS
    # ==========================================================

    st.markdown("## 📈 Skill Progress")

    skills={

    "Coding":coding_skill_score,

    "Communication":communication_skill_score,

    "Aptitude":aptitude_score,

    "Reasoning":logical_reasoning_score,

    "Leadership":leadership_score,

    "Activities":extracurricular_score

    }

    for skill,value in skills.items():

        st.write(f"### {skill}")

        st.progress(value/100)

    # ==========================================================
    # PROFILE COMPLETENESS
    # ==========================================================

    st.markdown("---")
    st.markdown("# 📋 Profile Completeness")

    completed=100

    if internships_count==0:
        completed-=10

    if github_repos==0:
        completed-=10

    if certifications_count==0:
        completed-=10

    if projects_count==0:
        completed-=15

    if leadership_score<20:
        completed-=5

    st.progress(completed/100)

    st.metric(

        "Profile Completion",

        f"{completed}%"

    )

    # ==========================================================
    # AI MOTIVATION
    # ==========================================================

    quotes=[

    "Success comes from consistent learning.",

    "Every project increases your value.",

    "Never stop improving your coding skills.",

    "AI engineers build the future.",

    "Your GitHub is your digital portfolio.",

    "Dream big, work harder."

    ]

    st.markdown("---")

    st.info(np.random.choice(quotes))

    # ==========================================================
    # VERIFIED DEVELOPER CARD
    # ==========================================================

    st.markdown("""

    <div style="

    background:linear-gradient(135deg,#2563EB,#7C3AED);

    padding:35px;

    border-radius:25px;

    margin-top:40px;

    text-align:center;

    box-shadow:0 15px 40px rgba(0,0,0,.35);

    ">

    <h1 style="color:white;">

    🧠 SUBHAN AI

    </h1>

    <h3 style="color:white;">

    Artificial Intelligence Engineer

    </h3>

    <p style="color:white;font-size:18px;">

    Machine Learning

    •

    Deep Learning

    •

    Data Science

    •

    Python

    </p>

    <br>

    <h2 style="color:#FACC15;">

    Verified Developer

    ✅

    </h2>

    <br>

    <p style="color:white;">

    © 2026

    Subhan AI Placement Intelligence Platform

    </p>

    </div>

    """,unsafe_allow_html=True)
    # ==========================================================
    # ADMIN ANALYTICS DASHBOARD
    # ==========================================================

    st.markdown("---")
    st.markdown("# 📊 AI Admin Dashboard")

    dashboard1, dashboard2, dashboard3, dashboard4 = st.columns(4)

    dashboard1.metric(
        "Prediction",
        "Placed" if prediction==1 else "Not Placed"
    )

    dashboard2.metric(
        "Probability",
        f"{placed_probability:.1f}%"
    )

    dashboard3.metric(
        "Confidence",
        f"{confidence:.1f}%"
    )

    dashboard4.metric(
        "Generated",
        datetime.now().strftime("%H:%M:%S")
    )

    st.markdown("---")

    # ==========================================================
    # PROFILE HEALTH
    # ==========================================================

    st.markdown("## ❤️ Profile Health")

    health = 100

    if cgpa < 7:
        health -= 15

    if coding_skill_score < 70:
        health -= 15

    if communication_skill_score < 70:
        health -= 10

    if internships_count == 0:
        health -= 10

    if projects_count < 3:
        health -= 10

    if certifications_count < 2:
        health -= 10

    health = max(0, health)

    st.progress(health / 100)

    if health >= 90:
        st.success("Excellent Profile Health")

    elif health >= 75:
        st.info("Good Profile Health")

    elif health >= 60:
        st.warning("Average Profile Health")

    else:
        st.error("Poor Profile Health")

    # ==========================================================
    # TOP SKILLS
    # ==========================================================

    st.markdown("## 🏆 Top Skills")

    skills = {
        "Coding": coding_skill_score,
        "Communication": communication_skill_score,
        "Aptitude": aptitude_score,
        "Reasoning": logical_reasoning_score,
        "Leadership": leadership_score,
        "Activities": extracurricular_score
    }

    ranking = sorted(
        skills.items(),
        key=lambda x: x[1],
        reverse=True
    )

    rank_df = pd.DataFrame(
        ranking,
        columns=["Skill","Score"]
    )

    st.dataframe(
        rank_df,
        hide_index=True,
        use_container_width=True
    )

    # ==========================================================
    # AI DECISION LEVEL
    # ==========================================================

    st.markdown("## 🤖 Decision Quality")

    if confidence >= 98:

        quality = "★★★★★"

    elif confidence >= 95:

        quality = "★★★★☆"

    elif confidence >= 90:

        quality = "★★★☆☆"

    else:

        quality = "★★☆☆☆"

    st.metric(
        "Prediction Quality",
        quality
    )

    # ==========================================================
    # SESSION SUMMARY
    # ==========================================================

    st.markdown("## 📋 Session Summary")

    st.success(f"""
    Student Age : {age}

    Branch : {branch}

    CGPA : {cgpa}

    Projects : {projects_count}

    Internships : {internships_count}

    Placement Probability : {placed_probability:.2f}%

    Confidence : {confidence:.2f}%
    """)
    # ==========================================================
    # AI PLACEMENT INTELLIGENCE CENTER
    # ==========================================================

    st.markdown("---")
    st.markdown("# 🧠 AI Placement Intelligence Center")

    # ----------------------------------------------------------
    # AI RISK LEVEL
    # ----------------------------------------------------------

    if placed_probability >= 90:

        risk = "🟢 Very Low Risk"

    elif placed_probability >= 75:

        risk = "🟢 Low Risk"

    elif placed_probability >= 60:

        risk = "🟡 Medium Risk"

    elif placed_probability >= 40:

        risk = "🟠 High Risk"

    else:

        risk = "🔴 Very High Risk"

    # ----------------------------------------------------------
    # EMPLOYABILITY
    # ----------------------------------------------------------

    employability = (

        coding_skill_score*0.30+

        communication_skill_score*0.20+

        aptitude_score*0.15+

        logical_reasoning_score*0.15+

        leadership_score*0.10+

        extracurricular_score*0.10

    )

    left,right=st.columns(2)

    with left:

        st.metric(

            "Employability Index",

            f"{employability:.1f}/100"

        )

    with right:

        st.metric(

            "Placement Risk",

            risk

        )

    # ----------------------------------------------------------
    # PROFILE STRENGTH
    # ----------------------------------------------------------

    profile_strength=(

    cgpa*10+

    coding_skill_score+

    communication_skill_score+

    projects_count*4+

    internships_count*5

    )/5

    st.progress(profile_strength/100)

    # ----------------------------------------------------------
    # AI DECISION ENGINE
    # ----------------------------------------------------------

    st.markdown("## 🤖 AI Decision Engine")

    if placed_probability>=90:

        decision="""
    Excellent profile detected.

    The student has a very high probability
    of securing placement.

    Recommendation:

    Apply directly to Product Based Companies.
    """

    elif placed_probability>=75:

        decision="""
    Very Competitive Profile.

    Continue improving DSA,
    Projects,
    Communication.

    High placement chances.
    """

    elif placed_probability>=60:

        decision="""
    Average Profile.

    Needs better internships,
    Projects,
    Coding Practice.

    Placement possible with preparation.
    """

    else:

        decision="""
    Weak Profile.

    Focus on:

    CGPA

    Projects

    Internships

    Coding

    Mock Interviews

    before placement season.
    """

    st.success(decision)

    # ----------------------------------------------------------
    # AI READINESS MATRIX
    # ----------------------------------------------------------

    st.markdown("## 📊 Readiness Matrix")

    matrix=pd.DataFrame({

    "Category":[

    "Academics",

    "Coding",

    "Communication",

    "Experience",

    "Leadership"

    ],

    "Score":[

    cgpa*10,

    coding_skill_score,

    communication_skill_score,

    (projects_count*10),

    leadership_score

    ]

    })

    fig=px.bar(

    matrix,

    x="Category",

    y="Score",

    text="Score"

    )

    fig.update_layout(

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    font_color="white",

    height=450

    )

    st.plotly_chart(fig,use_container_width=True)

    # ----------------------------------------------------------
    # NEXT TARGETS
    # ----------------------------------------------------------

    st.markdown("## 🎯 Next Targets")

    targets=[]

    if cgpa<8.5:
        targets.append("Increase CGPA to 8.5+")

    if coding_skill_score<90:
        targets.append("Reach Coding Score 90+")

    if internships_count<3:
        targets.append("Complete 3 Internships")

    if github_repos<25:
        targets.append("Publish 25 GitHub Projects")

    if certifications_count<8:
        targets.append("Earn 8 Certifications")

    if communication_skill_score<85:
        targets.append("Improve Communication")

    for i,target in enumerate(targets,1):

        st.write(f"{i}. {target}")

    # ----------------------------------------------------------
    # AI SCOREBOARD
    # ----------------------------------------------------------

    st.markdown("---")
    st.markdown("# 🏆 Final AI Scoreboard")

    scoreboard=pd.DataFrame({

    "Metric":[

    "Placement Probability",

    "Career Score",

    "Resume Score",

    "Interview Score",

    "Confidence",

    "Profile Health"

    ],

    "Score":[

    f"{placed_probability:.1f}%",

    f"{career_score:.1f}",

    f"{resume_score:.1f}",

    f"{interview_score:.1f}",

    f"{confidence:.1f}%",

    f"{health}%"

    ]

    })

    st.dataframe(

    scoreboard,

    hide_index=True,

    use_container_width=True

    )

    st.balloons()