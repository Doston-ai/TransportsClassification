import streamlit as st
from fastai.vision.all import *
import plotly.express as px
from PIL import Image

model = load_learner('transport_model1.pkl')

file = st.file_uploader("Rasm yuklang", type=["png", "jpg", "jpeg"])

if file:
    img = PILImage.create(file)
    st.image(img.to_thumb(256,256), caption="Yuklangan rasm")
    pred, pred_idx, probs = model.predict(img)
    st.success(f"Taxmin: {pred}")
    st.write(f"Ishonchlilik: {probs[pred_idx]:.3f}")
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig, use_container_width=True)
    st.balloons()
    
    
    st.success("Rasm muvaffaqiyatli yuklandi!")
else:
    st.info("Iltimos, rasm tanlang!")

