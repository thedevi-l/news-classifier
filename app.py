import streamlit as st
import matplotlib.pyplot as plt
from model import train_model

st.set_page_config(page_title="Fake News Classifier", layout="centered")

st.title("📰 Fake News Classifier")
st.markdown("Введите текст новости, и модель определит — **фейковая** она или **настоящая**.")

with st.spinner("🔄 Обучение модели..."):
    model, vectorizer, accuracy = train_model()

user_input = st.text_area("✏️ Введите текст новости:")

if st.button("Проверить"):
    if user_input.strip() == "":
        st.warning("Пожалуйста, введите текст новости.")
    else:
        input_vec = vectorizer.transform([user_input])
        proba = model.predict_proba(input_vec)[0]
        label = model.predict(input_vec)[0]
        confidence = max(proba)

        st.success(f"✅ Результат: **{label}** (уверенность: {confidence:.2f})")

st.markdown("### 📊 Точность модели")
fig, ax = plt.subplots()
ax.bar(["Model Accuracy"], [accuracy], color="skyblue")
ax.set_ylim([0, 1])
ax.set_ylabel("Точность")
ax.set_title("Logistic Regression Accuracy")
st.pyplot(fig)
