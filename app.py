import streamlit as st
import joblib

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

model = joblib.load("fake_news_model.pkl")

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article or headline below and check whether it is Real or Fake."
)

news_text = st.text_area(
    "Paste News Content Here",
    height=200
)

if st.button("Check News"):

    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:

        prediction = model.predict([news_text])[0]

        if prediction == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")

st.markdown("---")
st.write("Developed by Preethi")