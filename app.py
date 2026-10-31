import json
import numpy as np
import streamlit as st
import tensorflow as tf

st.title("Pakistan Sign Language Recognition")
st.write("Hath ke keypoints ki JSON file upload karen, model Urdu harf batayega.")

@st.cache_resource
def load_all():
    model = tf.keras.models.load_model("psl_model.keras")
    with open("class_names.json", encoding="utf-8") as fp:
        names = json.load(fp)
    return model, names

model, class_names = load_all()

def prepare(d):
    p = d["people"][0]
    feat = (p["pose_keypoints_2d"]
            + p["hand_left_keypoints_2d"]
            + p["hand_right_keypoints_2d"])
    if len(feat) != 201:
        return None
    pts = np.array(feat, dtype="float32").reshape(67, 3)
    conf = pts[:, 2]
    xy = pts[:, :2]
    xy = xy - xy[1]
    shoulder = np.linalg.norm(xy[2] - xy[5])
    if shoulder < 1e-6:
        shoulder = 1.0
    xy = xy / shoulder
    xy = xy * (conf[:, None] > 0)
    out = np.concatenate([xy, conf[:, None]], axis=1).astype("float32")
    return out[None, :, :]

uploaded = st.file_uploader("JSON file upload karen", type=["json"])

if uploaded is not None:
    try:
        data = json.load(uploaded)
        x = prepare(data)
        if x is None:
            st.error("Ye file sahi format mein nahi hai (201 keypoint numbers chahiye).")
        else:
            probs = model.predict(x)[0]
            top = np.argsort(probs)[::-1][:3]
            st.success(f"Predicted harf: {class_names[top[0]]}  ({probs[top[0]] * 100:.1f}%)")
            st.write("Top 3 andaze:")
            for i in top:
                st.write(f"{class_names[i]} : {probs[i] * 100:.1f}%")
    except Exception as e:
        st.error(f"File parhne mein masla: {e}")
