import pandas as pd
import numpy as np

num_records = 100000

brands = ['Brand1', 'Brand2', 'Brand3', 'Brand4', 'Brand5']
models = [f'Model{i}' for i in range(1, 11)]
top_load = [True, False]
size = ['Small', 'Medium', 'Large']
capacity = np.random.normal(500, 50, num_records).tolist()
efficiency = np.random.normal(90, 10, num_records).tolist()
seller_ratings = np.random.uniform(1, 5, num_records).tolist()
extended_warranty = [True, False]
avg_delivery_time = np.random.uniform(1, 7, num_records).tolist()
avg_shipping_costs = np.random.uniform(50, 150, num_records).tolist()
units_sold = np.random.poisson(1000, num_records).tolist()
cart_additions = np.random.poisson(5000, num_records).tolist()
customer_sentiment_indicator = np.random.uniform(1, 5, num_records).tolist()
year = ['2021', '2022', '2023']

df = pd.DataFrame({
    'Brand': [np.random.choice(brands) for _ in range(num_records)],
    'Product Model': [np.random.choice(models) for _ in range(num_records)],
    'Top Load or Not': [np.random.choice(top_load) for _ in range(num_records)],
    'Size': [np.random.choice(size) for _ in range(num_records)],
    'Capacity': capacity,
    'Efficiency': efficiency,
    'Seller Ratings': seller_ratings,
    'Extended Warranty': [np.random.choice(extended_warranty) for _ in range(num_records)],
    'Avg. Delivery Time': avg_delivery_time,
    'Avg Shipping Costs': avg_shipping_costs,
    'Units Sold': units_sold,
    'Number of Times Item was placed in an online shopping cart': cart_additions,
    'Customer Sentiment Indicator': customer_sentiment_indicator,
    'Year': [np.random.choice(year) for _ in range(num_records)]
})

df.to_csv('data.csv', index=False)
