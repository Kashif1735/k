import joblib
import pandas as pd
import streamlit as st

# # -------------------- CUSTOM CSS --------------------
# st.markdown("""
#     <style>
#     /* 1. GLOBAL THEME RESET */
#     :root {
#         --primary-color: #1E3A8A;
#         --background-color: #F8F9FA;
#         --secondary-background-color: #FFFFFF;
#         --text-color: #1E293B;
#         --font: 'Inter', sans-serif;
#     }

#     .stApp {
#         background-color: var(--background-color);
#         color: var(--text-color);
#     }

#     /* 2. FORCE LABEL VISIBILITY */
#     /* This targets the actual text inside the labels */
#     div[data-testid="stWidgetLabel"] p, 
#     .stSlider label, 
#     .stSelectbox label, 
#     .stNumberInput label {
#         color: #1E3A8A !important;
#         font-weight: 700 !important;
#         font-size: 1.1rem !important;
#         opacity: 1 !important;
#     }

#     /* 3. SIDEBAR FIX */
#     [data-testid="stSidebar"] {
#         background-color: #FFFFFF !important;
#         border-right: 1px solid #E2E8F0;
#     }
    
#     [data-testid="stSidebar"] * {
#         color: #1E293B !important;
#     }

#     /* 4. INPUT FIELD STYLING */
#     /* Makes the input boxes look cleaner and prevents "ghosting" */
#     .stSelectbox div[data-baseweb="select"], 
#     .stNumberInput div[data-baseweb="input"] {
#         background-color: #FFFFFF !important;
#         border-radius: 8px !important;
#         border: 1px solid #CBD5E1 !important;
#     }

#     /* 5. BUTTON GRADIENT */
#     .stButton>button {
#         background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%) !important;
#         color: white !important;
#         font-weight: 600 !important;
#         border: none !important;
#         padding: 0.6rem 2rem !important;
#         border-radius: 10px !important;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
#     </style>
#     """, unsafe_allow_html=True)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
import os
print(f"Current working directory: {os.getcwd()}")
model = joblib.load("rf_model.joblib")
model_features = joblib.load("model_columns.joblib")

# # -------------------- LOAD DATA --------------------
df = pd.read_csv("cleaned_df.csv")

# -------------------- HEADER --------------------
st.markdown("""
    <h1 style="text-align: center; font-weight: 800; color: #1E3A8A; margin-bottom: 0px;">
        🏠 PrimeEstate
    </h1>
    """, unsafe_allow_html=True)

st.markdown("""
    <p style="text-align: center; font-weight: 700; font-size: 20px; color: #475569; margin-top: 0px;">
        Premium Real Estate ML Estimator
    </p>
    """, unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("🏠 App Info")
    st.image("house_logo.png", width=200)
    st.markdown("""
    ### Instructions:
    - Select location
    - Enter property details
    - Click predict

    Built using Machine Learning
    """)


# -------------------- LOAD DATA --------------------
df_raw = pd.read_csv("cleaned_df.csv")

# Extract locations from model
model_features = joblib.load("model_columns.joblib")
locations = [col.replace("location_", "") 
             for col in model_features if col.startswith("location_")]

# -------------------- INPUT --------------------
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("📍 Location", sorted(locations))
    sqft = st.number_input("📐 Total Square Feet", min_value=300)

with col2:
    bath = st.selectbox("🛁 Bathrooms", sorted(df_raw["bath"].unique()))
    bhk = st.selectbox("🏠 BHK", sorted(df_raw["bhk"].unique()))

# -------------------- PREPARE INPUT --------------------
def prepare_input():
    # Create dictionary with all features = 0
    input_dict = {col: 0 for col in model_features}

    # Fill numerical values
    input_dict['total_sqft'] = sqft
    input_dict['bath'] = bath
    input_dict['bhk'] = bhk

    # Set selected location = 1
    loc_col = f"location_{location}"
    if loc_col in input_dict:
        input_dict[loc_col] = 1

    return pd.DataFrame([input_dict])

# -------------------- PREDICTION --------------------
if st.button("💰 Predict Price"):
    input_df = prepare_input()

    prediction = model.predict(input_df)
    price = float(f"{prediction[0]:.2f}")

    # st.markdown(f"""
    #     <div class="result-card">
    #         <p style="margin:0; font-size: 1.2rem; opacity: 0.9;">Estimated Market Value</p>
    #         <h1 style="margin:0; color: white;">₹ {price*100000:,.0f}</h1>
    #     </div>
    # """, unsafe_allow_html=True)

    # st.success(f"🏡 Estimated Price: ₹ {price*100000:,.0f}")

    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            color: white;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            margin-bottom: 25px;
        ">
            <p style="margin: 0; font-size: 1.1rem; opacity: 0.9; color: white !important;">
                Estimated Market Value
            </p>
            <h1 style="margin: 0; font-size: 3rem; font-weight: 800; color: white !important;">
                ₹ {price*100000:,.0f}
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # ---------------- PRICE INSIGHT ----------------
    st.subheader("💡 Price Insight")

    # Filter similar properties (NO location)
    similar_props = df[
        (df["total_sqft"].between(sqft * 0.8, sqft * 1.2)) &
        (df["bhk"] == bhk) &
        (df["bath"].between(bath - 1, bath + 1))
    ]

    if len(similar_props) > 5:
        avg_price = similar_props["price"].mean()
        diff_percent = ((price - avg_price) / avg_price) * 100

        if diff_percent < -10:
            st.success(f"🟢 Underpriced by {abs(diff_percent):.1f}% compared to similar homes")
        elif -10 <= diff_percent <= 10:
            st.info("🟡 Fairly priced (close to market average)")
        else:
            st.error(f"🔴 Overpriced by {diff_percent:.1f}% compared to similar homes")

        st.caption(f"📊 Based on {len(similar_props)} similar properties")
    else:
        st.warning("⚠️ Not enough similar data to generate insight")


    # -------------------- VISUALIZATION --------------------
    import matplotlib.pyplot as plt

    if len(similar_props) > 5:
        st.subheader("📊 Market Comparison")

        fig, ax = plt.subplots(figsize=(5,3))  # 👈 FIX

        ax.hist(similar_props["price"], bins=15)
        ax.axvline(price, linestyle='dashed')

        ax.set_title("Your Price vs Similar Properties")
        ax.set_xlabel("Price (Lakhs)")
        ax.set_ylabel("Frequency")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<center>Developed as part of Machine Learning (Data Science) Project | CS Engineering</center>",
    unsafe_allow_html=True
)