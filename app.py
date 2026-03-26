import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # <-- เพิ่มบรรทัดนี้เพื่อเรียกใช้กราฟ Radar

st.set_page_config(
    page_title='⚽ FIFA 22 Player Value Predictor',
    page_icon='⚽',
    layout='wide'
)

@st.cache_resource
def load_model():
    model = joblib.load('player_value_model.pkl')
    with open('model_metadata.json') as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()

st.title('⚽ FIFA 22 Player Value Predictor')
st.markdown("""
Predict a football player market value (EUR) based on their attributes.  
Enter the player stats below and click **Predict**.
""")
st.info('📌 Disclaimer: ผลลัพธ์ทำนายจากโมเดล ML ที่เรียนรู้จากข้อมูล FIFA 22 ไม่ใช่ราคาตลาดจริง ควรใช้เป็นข้อมูลอ้างอิงเบื้องต้นเท่านั้น')

st.subheader('🎮 Player Attributes')
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('**📊 Overall Stats**')
    overall = st.slider('Overall Rating', 40, 99, 75, help='คะแนนรวมความสามารถทั้งหมด (1-99)')
    potential = st.slider('Potential', 40, 99, 80, help='ศักยภาพสูงสุดที่สามารถพัฒนาได้')
    age = st.slider('Age', 16, 45, 24, help='อายุผู้เล่น')
    international_reputation = st.selectbox('International Reputation', [1,2,3,4,5],
        index=0, help='ชื่อเสียงระดับนานาชาติ 1=ต่ำสุด, 5=สูงสุด')
    wage_eur = st.number_input('Weekly Wage (EUR)', 500, 500000, 10000, step=500,
        help='เงินเดือนต่อสัปดาห์เป็นยูโร')

with col2:
    st.markdown('**⚡ Physical and Skills**')
    is_gk = st.checkbox('Is Goalkeeper (GK)?', value=False)
    preferred_foot = st.radio('Preferred Foot', ['Right','Left'], horizontal=True)
    weak_foot = st.slider('Weak Foot Stars', 1, 5, 3, help='ความสามารถเท้าอ่อน 1-5 ดาว')
    skill_moves = st.slider('Skill Moves Stars', 1, 5, 3, help='ความสามารถลูกเล่น 1-5 ดาว')
    height_cm = st.slider('Height (cm)', 155, 205, 180)
    weight_kg = st.slider('Weight (kg)', 50, 110, 75)

with col3:
    st.markdown('**🏃 Performance Stats**')
    if is_gk:
        st.info('GK mode: ค่า outfield stats ตั้งเป็น 0 อัตโนมัติ')
        pace=shooting=passing=dribbling=defending=physic=0
        ac=af=aha=asp=av=mr=mv=mc=0
    else:
        pace = st.slider('Pace', 20, 99, 70)
        shooting = st.slider('Shooting', 20, 99, 65)
        passing = st.slider('Passing', 20, 99, 70)
        dribbling = st.slider('Dribbling', 20, 99, 72)
        defending = st.slider('Defending', 20, 99, 50)
        physic = st.slider('Physic', 20, 99, 70)
        ac = st.slider('Attacking Crossing', 20, 99, 65)
        af = st.slider('Attacking Finishing', 20, 99, 65)
        aha = st.slider('Heading Accuracy', 20, 99, 60)
        asp = st.slider('Short Passing', 20, 99, 70)
        av = st.slider('Volleys', 20, 99, 60)
        mr = st.slider('Movement Reactions', 20, 99, 68)
        mv = st.slider('Mentality Vision', 20, 99, 68)
        mc = st.slider('Mentality Composure', 20, 99, 68)

if potential < overall:
    st.warning('⚠️ Potential ควรมากกว่าหรือเท่ากับ Overall Rating')

if st.button('🔮 Predict Market Value', use_container_width=True, type='primary'):
    input_data = pd.DataFrame([{
        'overall': overall, 'potential': potential, 'age': age,
        'height_cm': height_cm, 'weight_kg': weight_kg,
        'pace': pace, 'shooting': shooting, 'passing': passing,
        'dribbling': dribbling, 'defending': defending, 'physic': physic,
        'attacking_crossing': ac, 'attacking_finishing': af,
        'attacking_heading_accuracy': aha, 'attacking_short_passing': asp,
        'attacking_volleys': av, 'movement_reactions': mr,
        'mentality_vision': mv, 'mentality_composure': mc,
        'weak_foot': weak_foot, 'skill_moves': skill_moves,
        'international_reputation': international_reputation,
        'wage_eur': wage_eur, 'is_gk': int(is_gk),
        'preferred_foot': preferred_foot
    }])

    log_pred = model.predict(input_data)[0]
    pred_value = np.expm1(log_pred)

    st.success(f'### 💰 Predicted Market Value: €{pred_value:,.0f}')
    st.markdown(f'*คิดเป็น: €{pred_value/1e6:.2f} Million EUR*')

    low = pred_value * 0.85
    high = pred_value * 1.15
    st.info(f'📊 ช่วงประมาณการ (±15%): €{low:,.0f} — €{high:,.0f}')

    st.markdown("---")
    st.markdown("### 🕸️ Player Stats Overview")
    categories = ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physic']

    r_values = [pace, shooting, passing, dribbling, defending, physic]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=r_values,
        theta=categories,
        fill='toself',
        name='Player Stats',
        line=dict(color='#42A5F5')
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown("---")
 

    with st.expander('📈 Model Performance Metrics'):
        m = meta['test_metrics']
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('R² Score', f"{m['r2']:.4f}")
        c2.metric('MAE', f"€{m['mae']:,.0f}")
        c3.metric('RMSE', f"€{m['rmse']:,.0f}")
        c4.metric('MAPE', f"{m['mape']:.1f}%")

with st.expander('📊 Feature Importance (Gradient Boosting) — Bonus'):
    feat_names = meta['numeric_features']
    try:
        imps = model.named_steps['model'].feature_importances_[:len(feat_names)]
        fi_df = pd.DataFrame({'feature': feat_names, 'importance': imps})
        fi_df = fi_df.nlargest(10, 'importance').sort_values('importance')
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(fi_df['feature'], fi_df['importance'], color='#42A5F5')
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importances')
        st.pyplot(fig)
    except Exception as e:
        st.write(f'ไม่สามารถแสดง feature importance: {e}')
