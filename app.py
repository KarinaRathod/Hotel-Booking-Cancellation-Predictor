import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# --- 1. Page Configuration & CSS ---
st.set_page_config(page_title="Hotel Booking Predictor", page_icon="🏨", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    /* Style the form submit button to stand out */
    [data-testid="stFormSubmitButton"]>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3em; 
        background-color: #ff4b4b; 
        color: white; 
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    [data-testid="stFormSubmitButton"]>button:hover {
        background-color: #ff3333;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Helper Functions ---
@st.cache_resource
def load_assets():
    """Loads the machine learning model and encoders."""
    try:
        model = pickle.load(open("model.pkl", "rb"))
        encoders = pickle.load(open("encoders.pkl", "rb"))
        return model, encoders
    except FileNotFoundError:
        return None, None

def create_gauge_chart(probability):
    """Generates a Plotly gauge chart for the risk probability."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        number = {'suffix': "%", 'font': {'size': 40}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Cancellation Risk", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#ff4b4b" if probability > 0.5 else "#28a745"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "rgba(40, 167, 69, 0.2)"},   # Green zone
                {'range': [30, 70], 'color': "rgba(255, 193, 7, 0.2)"},  # Yellow zone
                {'range': [70, 100], 'color': "rgba(220, 53, 69, 0.2)"}  # Red zone
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- 3. Main Application Logic ---
def main():
    model, encoders = load_assets()

    if model is None:
        st.error("⚠️ Error: 'model.pkl' or 'encoders.pkl' not found. Please upload them to the project folder.")
        st.stop()

    # Header
    st.title("🏨 Hotel Booking Cancellation Predictor")
    st.markdown("Fill in the guest and stay details below to evaluate the likelihood of a cancellation.")
    
    # Form wrapping prevents the app from rerunning on every single input change
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👥 Guest Profile")
            adults = st.number_input("Adults", 1, 5, 2)
            children = st.number_input("Children", 0, 5, 0)
            repeated = st.selectbox("Repeated Guest", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            p_c = st.number_input("Prev. Cancellations", 0, 10, 0)
            p_not_c = st.number_input("Prev. Not Cancelled", 0, 20, 0)

        with col2:
            st.subheader("📅 Stay Details")
            weekend = st.slider("Weekend Nights", 0, 10, 1)
            week = st.slider("Week Nights", 0, 10, 2)
            lead_time = st.number_input("Lead Time (Days)", 0, 500, 50)
            month = st.selectbox("Reservation Month", range(1, 13), index=5)
            day = st.selectbox("Reservation Day", range(1, 32), index=14)

        with col3:
            st.subheader("🛎️ Room & Plan")
            market = st.selectbox("Market Segment", ["Online", "Offline", "Corporate", "Aviation", "Complementary"])
            room = st.selectbox("Room Type", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5"])
            meal = st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
            price = st.number_input("Average Price ($)", 0.0, 500.0, 100.0)
            parking = st.radio("Car Parking Space?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
            requests = st.slider("Special Requests", 0, 5, 0)

        st.divider()
        # The submit button triggers the prediction
        submitted = st.form_submit_button("Analyze Booking Risk")

    # --- 4. Prediction Execution ---
    if submitted:
        input_df = pd.DataFrame({
            "number of adults": [adults], "number of children": [children],
            "number of weekend nights": [weekend], "number of week nights": [week],
            "type of meal": [meal], "car parking space": [parking],
            "room type": [room], "lead time": [lead_time],
            "market segment type": [market], "repeated": [repeated],
            "P-C": [p_c], "P-not-C": [p_not_c],
            "average price": [price], "special requests": [requests],
            "reservation_month": [month], "reservation_day": [day]
        })

        # Apply Encoders
        for col, encoder in encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col])

        # Get Predictions
        prediction = model.predict(input_df)
        
        # UI for Results
        st.subheader("📊 Analysis Results")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.write("") # Spacer
            st.write("") # Spacer
            if prediction[0] == 1:
                st.error("### 🚩 Status: High Risk of Cancellation")
                st.write("Consider reaching out to the guest with a special offer or requesting a partial deposit to secure the booking.")
            else:
                st.success("### ✅ Status: Likely to Confirm")
                st.write("This booking looks solid. Standard confirmation procedures apply.")
                
        with res_col2:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0][1]
                # Display the Plotly Gauge Chart
                st.plotly_chart(create_gauge_chart(prob), use_container_width=True)
            else:
                st.info("Note: Confidence score visualization is not available for this model type.")

if __name__ == "__main__":
    main()