# 🧼 LWM Insights

LWM Insights is a Streamlit-based web application that enables washing machine manufacturers and retailers to understand and visualize market trends and sales predictions. This tool provides descriptive, predictive, and prescriptive analytics for sales data, including features such as brand, product model, size, capacity, and efficiency, among others.

![App Preview](./images/preview.png)   <!-- Insert app preview image path -->

## Features
* Filters to refine and customize data views
* Descriptive analytics with interactive plots to visualize the sales count by brand, product features comparisons, etc.
* Predictive analytics with linear regression models to forecast unit sales

## Installation

Follow these steps to run this app locally:

1. Clone this repository to your local machine.
```shell
git clone https://github.com/yourusername/lwm-insights.git
```

2. Navigate to the project directory.
```shell
cd lwm-insights
```

3. Install the required dependencies.
```shell
pip install -r requirements.txt
```

4. Run the Streamlit app.
```shell
streamlit run app.py
```

The app should now be accessible at localhost:8501 in your web browser.

## Data
The app uses a synthetic dataset of washing machine sales records generated with the Faker and NumPy libraries. The dataset includes attributes such as brand, model, size, efficiency, delivery time, shipping costs, units sold, and customer sentiment.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
