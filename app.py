from flask import Flask, render_template, send_file
import pandas as pd
from analysis import perform_analysis, export_cleaned_data

app = Flask(__name__)

@app.route('/')
def index():
    # Assuming you have the sales, products, stores, and customers DataFrames
    sales = pd.read_csv('transactions.csv')
    products = pd.read_csv('products.csv')
    stores = pd.read_csv('stores.csv')
    customers = pd.read_csv('customers.csv')
    
    results, cleaned_sales = perform_analysis(sales, products, stores, customers)
    
    return render_template('results.html', results=results)

@app.route('/export')
def export():
    # Assuming you have the sales DataFrame loaded previously in perform_analysis
    sales = pd.read_csv('transactions.csv')
    products = pd.read_csv('products.csv')
    stores = pd.read_csv('stores.csv')
    customers = pd.read_csv('customers.csv')
    
    _, cleaned_sales = perform_analysis(sales, products, stores, customers)
    
    file_path = export_cleaned_data(cleaned_sales)
    
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)


