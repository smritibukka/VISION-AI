import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
from collections import Counter

from detector import detect_objects


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="VISION-AI",
    page_icon="👁️",
    layout="wide"
)


# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown(
    """
    <style>

    /* Main page */

    .main {
        padding-top: 1rem;
    }


    /* Main title */

    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }


    /* Subtitle */

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }


    /* Feature cards */

    .feature-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        text-align: center;
        margin-bottom: 1rem;
    }


    /* Section headers */

    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }


    /* Footer */

    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        font-size: 0.9rem;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">👁️ VISION-AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Object Detection System'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="feature-card">
        Upload an image or capture one using your camera.
        VISION-AI detects objects, counts them, displays
        confidence scores, and draws bounding boxes.
    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Settings")

    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )

    st.write(
        f"Minimum confidence: "
        f"**{threshold * 100:.0f}%**"
    )

    st.divider()

    st.header("ℹ️ About")

    st.write(
        """
        **VISION-AI** is an AI-powered
        computer vision application.

        **Technology**

        • Python
        • Streamlit
        • TensorFlow
        • TensorFlow Hub
        • Pillow
        • Pandas

        **Model**

        SSD MobileNet V2
        """
    )


# ==========================================
# INPUT METHOD
# ==========================================

st.markdown(
    '<div class="section-title">📥 Choose Input</div>',
    unsafe_allow_html=True
)

input_method = st.radio(
    "Select how you want to provide an image:",
    [
        "🖼️ Upload Image",
        "📷 Camera"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


# ==========================================
# IMAGE VARIABLE
# ==========================================

image = None


# ==========================================
# UPLOAD IMAGE
# ==========================================

if input_method == "🖼️ Upload Image":

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception:

            st.error(
                "❌ The uploaded image could not be opened. "
                "Please try a different image."
            )


# ==========================================
# CAMERA
# ==========================================

else:

    camera_image = st.camera_input(
        "Take a picture"
    )

    if camera_image is not None:

        try:

            image = Image.open(
                camera_image
            ).convert("RGB")

        except Exception:

            st.error(
                "❌ The camera image could not be processed. "
                "Please try again."
            )


# ==========================================
# IMAGE PROCESSING
# ==========================================

if image is not None:

    st.divider()


    # ======================================
    # INPUT IMAGE
    # ======================================

    st.markdown(
        '<div class="section-title">🖼️ Input Image</div>',
        unsafe_allow_html=True
    )

    st.image(
        image,
        use_container_width=True
    )


    # ======================================
    # DETECT BUTTON
    # ======================================

    detect_button = st.button(
        "🔍 Detect Objects",
        use_container_width=True
    )


    if detect_button:

        # ==================================
        # RUN AI DETECTION
        # ==================================

        with st.spinner(
            "🤖 AI is analyzing the image..."
        ):

            try:

                detections = detect_objects(
                    image,
                    threshold=threshold
                )

            except Exception as e:

                st.error(
                    "❌ Something went wrong "
                    "while analyzing the image."
                )

                st.exception(e)

                detections = None


        # ==================================
        # DETECTION ERROR
        # ==================================

        if detections is None:

            st.warning(
                "Please try another image or "
                "try running the detection again."
            )


        # ==================================
        # OBJECTS FOUND
        # ==================================

        elif detections:

            # --------------------------------
            # CREATE ANNOTATED IMAGE
            # --------------------------------

            annotated_image = image.copy()

            draw = ImageDraw.Draw(
                annotated_image
            )

            image_width, image_height = (
                image.size
            )


            # --------------------------------
            # DRAW DETECTIONS
            # --------------------------------

            for detection in detections:

                name = detection["name"]

                confidence = (
                    detection["score"] * 100
                )

                top, left, bottom, right = (
                    detection["box"]
                )


                # Convert normalized coordinates
                # to pixel coordinates

                left = int(
                    left * image_width
                )

                right = int(
                    right * image_width
                )

                top = int(
                    top * image_height
                )

                bottom = int(
                    bottom * image_height
                )


                # Draw bounding box

                draw.rectangle(
                    [
                        left,
                        top,
                        right,
                        bottom
                    ],
                    outline="red",
                    width=3
                )


                # Create label

                label = (
                    f"{name.title()} "
                    f"{confidence:.1f}%"
                )


                # Draw label

                draw.text(
                    (
                        left,
                        max(0, top - 20)
                    ),
                    label,
                    fill="red"
                )


            # ==================================
            # SUCCESS MESSAGE
            # ==================================

            st.success(
                f"Detection completed! "
                f"{len(detections)} object(s) detected."
            )


            # ==================================
            # DETECTION STATISTICS
            # ==================================

            st.markdown(
                '<div class="section-title">'
                '📊 Detection Summary'
                '</div>',
                unsafe_allow_html=True
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Objects Detected",
                    len(detections)
                )


            with col2:

                highest_confidence = max(
                    detection["score"]
                    for detection in detections
                ) * 100

                st.metric(
                    "Highest Confidence",
                    f"{highest_confidence:.1f}%"
                )


            with col3:

                unique_objects = len(
                    set(
                        detection["name"]
                        for detection in detections
                    )
                )

                st.metric(
                    "Unique Objects",
                    unique_objects
                )


            # ==================================
            # OBJECT COUNT
            # ==================================

            st.markdown(
                '<div class="section-title">'
                '🔢 Object Count'
                '</div>',
                unsafe_allow_html=True
            )


            object_names = [
                detection["name"].title()
                for detection in detections
            ]


            object_counts = Counter(
                object_names
            )


            count_data = []


            for object_name, count in (
                object_counts.items()
            ):

                count_data.append(
                    {
                        "Object": object_name,
                        "Count": count
                    }
                )


            count_df = pd.DataFrame(
                count_data
            )


            st.dataframe(
                count_df,
                use_container_width=True,
                hide_index=True
            )


            # ==================================
            # DETECTION RESULT
            # ==================================

            st.markdown(
                '<div class="section-title">'
                '🎯 Detection Result'
                '</div>',
                unsafe_allow_html=True
            )


            st.image(
                annotated_image,
                caption="Detected Objects",
                use_container_width=True
            )


            # ==================================
            # RESULTS TABLE
            # ==================================

            st.markdown(
                '<div class="section-title">'
                '📋 Detected Objects'
                '</div>',
                unsafe_allow_html=True
            )


            data = []


            for detection in detections:

                data.append(
                    {
                        "Object":
                            detection["name"].title(),

                        "Confidence":
                            f"{detection['score'] * 100:.1f}%"
                    }
                )


            df = pd.DataFrame(
                data
            )


            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )


            # ==================================
            # DOWNLOAD RESULTS
            # ==================================

            csv_data = df.to_csv(
                index=False
            )


            st.download_button(
                label="📥 Download Detection Results",
                data=csv_data,
                file_name="vision_ai_results.csv",
                mime="text/csv",
                use_container_width=True
            )


        # ==================================
        # NO OBJECTS FOUND
        # ==================================

        else:

            st.warning(
                "⚠️ No objects were detected "
                "above the selected confidence threshold."
            )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    '<div class="footer">'
    '👁️ VISION-AI • AI Object Detection System'
    '</div>',
    unsafe_allow_html=True
)