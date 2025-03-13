import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import tensorflow as tf

# สร้าง Navbar ด้านบน
page = option_menu(
    menu_title=None,  # ซ่อนชื่อเมนู
    options=["Machine Learning", "Neural Network", "Machine Learning Demo", "Neural Network Model"],
    icons=["mushroom", "clipboard", "mushroom", "clipboard"],  # ไอคอนที่ใช้
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"  # แสดง Navbar ด้านบน
)

mapping = {
    "cap-shape": {"Bell": 1, "Conical": 2, "Convex": 3, "Flat": 4, "Sunken": 5, "Spherical": 6, "Others": 7},
    "cap-surface": {"Fibrous": 1, "Grooves": 2, "Scaly": 3, "Smooth": 4, "Dry": 5, "Shiny": 6, "Leathery": 7, "Silky": 8, "Sticky": 9, "Wrinkled": 10, "Fleshy": 11},
    "cap-color": {"Brown": 1, "Buff": 2, "Gray": 3, "Green": 4, "Pink": 5, "Purple": 6, "Red": 7, "White": 8, "Yellow": 9, "Blue": 10, "Orange": 11, "Black": 12},
    "does-bruise-or-bleed": {"Yes": 1, "No": 0},
    "gill-attachment": {"Adnate": 1, "Adnexed": 2, "Decurrent": 3, "Free": 4, "Sinuate": 5, "Pores": 6, "None": 7, "Unknown": 8},
    "gill-color": {"None": 0, "Brown": 1, "Buff": 2, "Gray": 3, "Green": 4, "Pink": 5, "Purple": 6, "Red": 7, "White": 8, "Yellow": 9, "Blue": 10, "Orange": 11, "Black": 12},
    "stem-color": {"None": 0, "Brown": 1, "Buff": 2, "Gray": 3, "Green": 4, "Pink": 5, "Purple": 6, "Red": 7, "White": 8, "Yellow": 9, "Blue": 10, "Orange": 11, "Black": 12},
    "Has-ring": {"Yes": 1, "No": 0},
    "ring-type": {"Cobwebby": 1, "Evanescent": 2, "Flaring": 3, "Grooved": 4, "Large": 5, "Pendant": 6, "Sheathing": 7, "Zone": 8, "Scaly": 9, "Movable": 10, "None": 0, "Unknown": 11},
    "habitat": {"Grasses": 1, "Leaves": 2, "Meadows": 3, "Paths": 4, "Heaths": 5, "Urban": 6, "Waste": 7, "Woods": 8},
    "season": {"Spring": 1, "Summer": 2, "Autumn": 3, "Winter": 4}
}

type_mapping = {
    "CASH-IN": 0,
    "CASH-OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}

# แสดงเนื้อหาตามที่เลือก
if page == "Machine Learning":
    st.title("🖥️ Machine Learning 🖥️")
    st.write("🛠️ maintenance 🛠️")
    
elif page == "Neural Network":
    st.title("🧠 Neural Network 🧠")
    st.write("🛠️ maintenance 🛠️")
    
elif page == "Machine Learning Demo":
    st.title("🍄 Machine Learning Demo🍄 ")
    st.write("กรอกข้อมูลเพื่อทำนายเห็ดพิษจากลักษณะของเห็ด")
    
    # รับค่าจากผู้ใช้
    cap_diameter = st.number_input("Cap Diameter (cm)", min_value=0.1, max_value=100.0, step=0.1)
    stem_height = st.number_input("Stem Height (cm)", min_value=0.1, max_value=100.0, step=0.1)
    stem_width = st.number_input("Stem Width (mm)", min_value=0.1, max_value=100.0, step=0.1)
    cap_shape = st.selectbox("Cap Shape", options=list(mapping["cap-shape"].keys()))
    cap_surface = st.selectbox("Cap Surface", options=list(mapping["cap-surface"].keys()))
    cap_color = st.selectbox("Cap Color", options=list(mapping["cap-color"].keys()))
    does_bruise_or_bleed = st.selectbox("Does Bruise or Bleed", options=list(mapping["does-bruise-or-bleed"].keys()))
    gill_attachment = st.selectbox("Gill Attachment", options=list(mapping["gill-attachment"].keys()))
    gill_color = st.selectbox("Gill Color", options=list(mapping["gill-color"].keys()))
    stem_color = st.selectbox("Stem Color", options=list(mapping["stem-color"].keys()))
    Has_ring = st.selectbox("Has Ring", options=list(mapping["Has-ring"].keys()))
    ring_type = st.selectbox("Ring Type", options=list(mapping["ring-type"].keys()))
    habitat = st.selectbox("Habitat", options=list(mapping["habitat"].keys()))
    season = st.selectbox("Season", options=list(mapping["season"].keys()))
    
    model_choice = st.selectbox("เลือกโมเดล", options=["KNN", "Random Forest"], index=0)
    
    if st.button("Predict"):
        model_path = "Model/mushroom_model_logreg.pkl" if model_choice == "KNN" else "Model/mushroom_model_RandomForest.pkl"
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        
        input_data = np.array([[cap_diameter, mapping["cap-shape"][cap_shape], mapping["cap-surface"][cap_surface], 
                                mapping["cap-color"][cap_color], mapping["does-bruise-or-bleed"][does_bruise_or_bleed], 
                                mapping["gill-attachment"][gill_attachment], mapping["gill-color"][gill_color], 
                                stem_height, stem_width, mapping["stem-color"][stem_color], 
                                mapping["Has-ring"][Has_ring], mapping["ring-type"][ring_type], 
                                mapping["habitat"][habitat], mapping["season"][season]]])
        
        prediction = model.predict(input_data)
        result = "🍄 เป็นพิษ" if prediction[0] == 1 else "✅ ปลอดภัย"
        st.success(f"ผลลัพธ์: {result}")

    
elif page == "Neural Network Model":
    st.title("🤖 Neural Network Model 🤖")
    st.write("กรอกข้อมูลเพื่อทำนายการฉ้อโกงทางการเงิน")
    
    step = st.number_input("Step", min_value=0, step=1)
    type_ = st.selectbox("Transaction Type", options=list(type_mapping.values()), format_func=lambda x: list(type_mapping.keys())[list(type_mapping.values()).index(x)])
    amount = st.number_input("Amount", min_value=0.0, step=0.01)
    oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, step=0.01)
    newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, step=0.01)
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, step=0.01)
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, step=0.01)
    
    if st.button("Predict"):
        model = tf.keras.models.load_model("Model/fraud_detection_model.keras")
        input_data = np.array([[step, type_, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])
        prediction = model.predict(input_data)
        result = "🚨 Fraud Detected" if prediction[0] > 0.5 else "✅ No Fraud"
        st.success(f"ผลลัพธ์: {result}")


