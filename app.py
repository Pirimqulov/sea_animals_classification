import streamlit as st 
from fastai.vision.all import *
import pathlib 
import plotly.express as px
import platform
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
# title
st.title('Dengiz hayvonlarini va qushlarni klassifikatsiya qiluvchi model')
st.text('Bunda biz asosan dengiz hayvonlari(marine mammal, shellfish, fish) va qushlarni (bird) taniy oladigan dastur qilmoqchi edik.')
# rasmni joylash
file = st.file_uploader('Rasm yuklash', type=['png','jpeg','gif','svg'])
if file : 
    st.image(file)
    # PIL convert
    img = PILImage.create(file)

    ### Mana shu yeriga filter qo'yish kerak. Ya'ni agar bu rasm boshqa rasmlar sinfiga kirsa Siz xato rasm tashladingiz desin
      #demak g'oya shundya bo'ladi : demak birinchi ustoz aytganday bitta 2 ta papka yaratib  bittasiga meni klasslarimga tegishli rasm (100 ta), ikkinchisiga umuman boshqa narsalarning rasmi (100 ta ) 
      # keyin model o'qitish va u javob qaytarganda if oladi agar boshqa sinfga tegishli bo'lsa shu yozuv chiqadi : "Siz kiritgan rasm modelning klassifikatsiya qiluvchi sinfiga kirmaydi"
      # agar bizning sinfga kirsa pasdagi narsalar bajariladi  
    # model
    model = load_learner('sea_animals.pkl')
    model11 = load_learner('filter1.pkl')
    pred11, pred_id11, probs11 = model11.predict(img)
    # st.success(f"Tdawdawadw guruh : {pred11}")
    
    if pred11=="Sea_animals" :
      #predictions
      pred, pred_id, probs = model.predict(img)
      st.success(f"To'g'ri kelgan hayvon turi : {pred}")
      st.info(f'Ehtimollik : {probs[pred_id]*100:.1f}%')
      # plotting
      fig = px.bar(x=probs*100, y=model.dls.vocab)
      st.plotly_chart(fig)
    else:
      st.warning('Yuklangan rasm dengiz hayvoni yoki qush emas. Iltimos shu hayvon va qushlarga tegishli rasm kiriting !')

