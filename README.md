
# 🏨 Hotel Booking Cancellation Predictor

A Streamlit web application that predicts the likelihood of a hotel guest canceling their reservation. This tool is designed to help hotel managers identify high-risk bookings and take proactive measures to optimize occupancy.

## ✨ Features
- **Interactive UI:** Clean, dashboard-style interface using Streamlit.
- **Real-time Prediction:** Uses a pre-trained Machine Learning model to evaluate booking parameters.
- **Risk Visualization:** Displays a dynamic gauge chart (via Plotly) to show the exact probability/confidence of a cancellation.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Data Manipulation:** Pandas
- **Machine Learning:** Scikit-Learn (Pickle)
- **Visualization:** Plotly

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/hotel-booking-predictor.git](https://github.com/yourusername/hotel-booking-predictor.git)
   cd hotel-booking-predictor

```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```


3. **Install dependencies**
```bash
pip install -r requirements.txt

```


4. **Run the Streamlit app**
```bash
streamlit run app.py

```



## 📁 Project Structure

* `app.py`: The main Streamlit application script.
* `model.pkl`: The pre-trained machine learning model.
* `encoders.pkl`: The saved label encoders for processing categorical inputs.
* `requirements.txt`: List of Python dependencies.



