import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd

# Code ML
MLcode1 = "import pandas as pd\ndf = pd.read_csv('../Dataset/mushroom_overload.csv')\ndf"

MLcode2 = "df = df.drop(columns=['gill-spacing','stem-root','stem-surface','veil-type','veil-color','spore-print-color'])\ndf"

MLcode3 = "df.isnull().sum()"

MLcode4 = "df.duplicated().sum()"

MLcode5 = "df=df.dropna()\ndf.drop_duplicates(inplace=True)\ndf"

MLcode6 = """mapping = {
    "cap-shape": {
        "b": 1, "c": 2, "x": 3, "f": 4, "s": 5, "p": 6, "o": 7
    },
    "cap-surface": {
        "i": 1, "g": 2, "y": 3, "s": 4, "d": 5, "h": 6, "l": 7, "k": 8, "t": 9, "w": 10, "e": 11
    },
    "cap-color": {
        "n": 1, "b": 2, "g": 3, "r": 4, "p": 5, "u": 6, "e": 7, "w": 8, "y": 9, "l": 10, "o": 11, "k": 12
    },
    "does-bruise-or-bleed": {
        "t": 1, "f": 0
    },
    "gill-attachment": {
        "a": 1, "x": 2, "d": 3, "e": 4, "s": 5, "p": 6, "f": 7, "?": 8
    },
    "gill-color": {
        "f": 0, "n": 1, "b": 2, "g": 3, "r": 4, "p": 5, "u": 6, "e": 7, "w": 8, "y": 9, "l": 10, "o": 11, "k": 12
    },
    "stem-color": {
        "f": 0, "n": 1, "b": 2, "g": 3, "r": 4, "p": 5, "u": 6, "e": 7, "w": 8, "y": 9, "l": 10, "o": 11, "k": 12
    },
    "has-ring": {
        "t": 1, "f": 0
    },
    "ring-type": {
        "c": 1, "e": 2, "r": 3, "g": 4, "l": 5, "p": 6, "s": 7, "z": 8, "y": 9, "m": 10, "f": 0, "?": 11
    },
    "habitat": {
        "g": 1, "l": 2, "m": 3, "p": 4, "h": 5, "u": 6, "w": 7, "d": 8
    },
    "season": {"s": 1, "u": 2, "a": 3, "w": 4},
    "class": {"e": 1, "p": 0}
}

df.replace(mapping, inplace=True)
df"
"""

MLcode7 = "from sklearn.model_selection import train_test_split\n\nx = df.drop(columns=['class'])\ny = df['class']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"

MLcode8 = """from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(random_state=42)

# Train the model on the training data
model_rf.fit(x_train, y_train)

# Make predictions on test data
y_pred_rf = model_rf.predict(x_test)

# Confusion matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix (Random Forest):")
print(conf_matrix_rf)

# Accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy (Random Forest):", accuracy_rf)

# Precision
precision_rf = precision_score(y_test, y_pred_rf)
print("Precision (Random Forest):", precision_rf)

# Recall
recall_rf = recall_score(y_test, y_pred_rf)
print("Recall (Random Forest):", recall_rf)

# F1 Score
f1_rf = f1_score(y_test, y_pred_rf)
print("F1 Score (Random Forest):", f1_rf)"
"""

MLcode9 = """from sklearn.linear_model import LogisticRegression

model_logreg = LogisticRegression()

# Train the model on the training data
model_logreg.fit(x_train, y_train)

# Make predictions on test data
y_pred_logreg = model_logreg.predict(x_test)

# Confusion matrix
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
print("Confusion Matrix (Logistic Regression):")
print(conf_matrix_logreg)

# Accuracy
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print("Accuracy:", accuracy_logreg)

# Precision
precision_logreg = precision_score(y_test, y_pred_logreg)
print("Precision:", precision_logreg)

# Recall
recall_logreg = recall_score(y_test, y_pred_logreg)
print("Recall (LogReg):", recall_logreg)

# F1 Score
f1_logreg = f1_score(y_test, y_pred_logreg)
print("F1 Score (LogReg):", f1_logreg)
"""

# Code NN
NNcode1 = "import tensorflow as tf"

# สร้าง Navbar ด้านบน
page = option_menu(
    menu_title=None,  # ซ่อนชื่อเมนู
    options=["Machine Learning", "Neural Network", "Machine Learning Demo", "Neural Network Model"],
    icons=["cast", "cast", "cast", "cast"],  # ไอคอนที่ใช้
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

df = pd.read_csv("Dataset/mushroom_overload.csv", nrows=20)

# แสดงเนื้อหาตามที่เลือก
if page == "Machine Learning":
    st.header("🗂️ Dataset ที่ใช้")
    st.subheader("🍄Mushroom Overload| 6.7M Rows")
    st.markdown("โหลดจาก Kaggle: https://www.kaggle.com/datasets/bwandowando/mushroom-overload/data")
    st.subheader("📊 ตัวอย่างข้อมูลจาก Dataset")
    st.dataframe(df.head(20))
    st.subheader("📄 Features")
    features_info = """
    - **cap-diameter (m)**: float number in cm  
    - **cap-shape (n)**: bell=b, conical=c, convex=x, flat=f, sunken=s, spherical=p, others=o  
    - **cap-surface (n)**: fibrous=i, grooves=g, scaly=y, smooth=s, dry=d, shiny=h, leathery=l, silky=k, sticky=t, wrinkled=w, fleshy=e  
    - **cap-color (n)**: brown=n, buff=b, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y, blue=l, orange=o, black=k  
    - **does-bruise-bleed (n)**: bruises-or-bleeding=t, no=f  
    - **gill-attachment (n)**: adnate=a, adnexed=x, decurrent=d, free=e, sinuate=s, pores=p, none=f, unknown=?  
    - **gill-spacing (n)**: close=c, distant=d, none=f  
    - **gill-color (n)**: see cap-color + none=f  
    - **stem-height (m)**: float number in cm  
    - **stem-width (m)**: float number in mm  
    - **stem-root (n)**: bulbous=b, swollen=s, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r  
    - **stem-surface (n)**: see cap-surface + none=f  
    - **stem-color (n)**: see cap-color + none=f  
    - **veil-type (n)**: partial=p, universal=u  
    - **veil-color (n)**: see cap-color + none=f  
    - **has-ring (n)**: ring=t, none=f  
    - **ring-type (n)**: cobwebby=c, evanescent=e, flaring=r, grooved=g, large=l, pendant=p, sheathing=s, zone=z, scaly=y, movable=m, none=f, unknown=?  
    - **spore-print-color (n)**: see cap-color  
    - **habitat (n)**: grasses=g, leaves=l, meadows=m, paths=p, heaths=h, urban=u, waste=w, woods=d  
    - **season (n)**: spring=s, summer=u, autumn=a, winter=w  
    - **class (n)**: e=edible, p=poisonous  
    """
    
    st.markdown(features_info)
    st.header("📝 การเตรียมข้อมูล")
    st.text("เริ่มจากนำเข้าข้อมูล DataSet ที่โหลดเข้ามาเก็บไว้ในตัวแปร df")
    st.code(MLcode1, language="python")
    st.image("Streamlit/image/ML/ML1.png")
    
    st.text("")
    st.text("จากนั้น drop ข้อมูลที่ไม่จำเป็นในการจำแนกเห็ดพิษออกไป")
    st.code(MLcode2, language="python")
    st.image("Streamlit/image/ML/ML2.png")
    
    st.text("")
    st.text("ต่อมาก็ทำการเช็คดูว่ามีค่าที่เป็น Null และมีข้อมูลที่ซ้ำกันไหม")
    st.code(MLcode3, language="python")
    st.image("Streamlit/image/ML/ML3.png")
    st.code(MLcode4, language="python")
    st.image("Streamlit/image/ML/ML4.png")
    
    st.text("")
    st.text("จะเห็นได้ว่ามีข้อมูลที่เป็น Null และมีข้อมูลที่ซ้ำกันอยู่จากนั้นทำการ Drop แถวที่เป็น Null และแถวที่เป็นข้อมูลซ้ำทิ้งออกไปได้เพราะเรามีข้อมูลจำนวนมากที่ไม่เป็น Null และ ไม่ซ้ำกัน")
    st.code(MLcode5, language="python")
    st.image("Streamlit/image/ML/ML5.png")
    
    st.text("")
    st.text("จากนั้นก็แปลงข้อมูลให้เป็นตัวเลข เพื่อที่จะทำให้ง่ายต่อการนำไปเทรนโมเดล")
    st.code(MLcode6, language="python")
    st.image("Streamlit/image/ML/ML6.png")
    
    st.text("")
    st.text("จากนั้นทำการแบ่งข้อมูลออกเป็นส่วนที่ใช้ฝึก 80% และใช้ทดสอบ 20% โดยให้คอลัมน์ที่เป็นผลลัพธ์ในการทำนายที่เราต้องการเก็บไว้ใน y ซึ่งก็คือ class ที่เป็นตัวบอกสถานะว่าเป็นเห็ดพิษหรือไม่ และที่เหลือไว้ใน x")
    st.code(MLcode7, language="python")
    
    st.text("โมเดลตัวแรกจะเลือกใช้ Random Forest ซึ่งเป็นอัลกอริธึมของ Machine Learning ที่ใช้สำหรับ Classification ซึ่งวิธีการของ Random Forest จะสร้าง Dicision Trees ออกมาหลายๆต้นออกมาแล้วจะนำผลลัพธ์ที่ได้จาก Dicision Trees ทั้งหมดมาเช็คดูว่าผลลัพธ์ที่ออกมาส่วนใหญ่เป็นอะไร แล้วจะให้ผลลัพธ์ส่วนใหญ่นั้นเป็นคำตอบออกมา หลังจากที่ได้โมเดลออกมาแล้วก็ลองทำการทดสอบโมเดลเพื่อดูค่า Confusion Matrix เพื่อแสดงจำนวน ค่าทำนายที่ถูกต้องและผิดพลาด | Accuracy ใช้วัดค่าความแม่นยำของโมเดล | Precision ใช้วัดความแม่นยำของการทำนายผล | Recall ใช้วัดค่าความถูกต้อง | F1 Score ใช้วัดค่าความสมดุลระหว่าง Precision และ Recall")
    st.code(MLcode8, language="python")
    st.image("Streamlit/image/ML/ML7.png")
    
    st.text("")
    st.text("โมเดลที่สองจะเลือกใช้ Logistic Regression เป็นอัลกอริธึมของ Machine Learning ที่ใช้สำหรับ Classification เหมือนกัน วิธีการของ Logistic Regression จะใช้สมการเส้นตรง แล้วนำค่าที่ได้ไปผ่านฟังก์ชัน Sigmoid เพื่อแปลงผลลัพธ์เป็นค่าความน่าจะเป็น แล้วตัดสินว่าอยู่ในกลุ่มไหน หลังจากที่ได้โมเดลออกมาแล้วก็ลองทำการทดสอบโมเดลเพื่อดูค่า Confusion Matrix, Accuracy, Precision, Recall, และ F1 Score เหมือนกับโมเดลแรก")
    st.code(MLcode9, language="python")
    st.image("Streamlit/image/ML/ML8.png")

    st.text("")
    st.text("หลังจากการเทรนโมเดลทั้งสองแบบจะเห็นได้ว่าอัลกอริธึมแบบ Random Forest จะความแม่นยำมากกว่า Logistic Regression อยู่มากเพราะมีการใช้ Dicision trees จำนวนมากมาใช้โหวตตัดสินใจ")
    
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
    
    model_choice = st.selectbox("เลือกโมเดล", options=["Logistic Regression", "Random Forest"], index=0)
    
    if st.button("Predict"):
        model_path = "Model/mushroom_model_logreg.pkl" if model_choice == "Logistic Regression" else "Model/mushroom_model_RandomForest.pkl"
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
    st.title("🤖 Neural Network Demo 🤖")
    st.write("อัปโหลดไฟล์รูปภาพเพื่อเช็คว่าเป็น หมา หรือ แมว")
    
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


