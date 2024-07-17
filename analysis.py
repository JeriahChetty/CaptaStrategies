import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans

def perform_analysis(sales, products, stores, customers):
    # Data Preparation
    sales.fillna({'QuantitySold': 0, 'SalePrice': 0}, inplace=True)
    products.fillna({'CostPrice': products['CostPrice'].median()}, inplace=True)
    stores.fillna({'StoreLocation': 'Unknown', 'StoreManager': 'Unknown'}, inplace=True)
    customers.fillna({'Age': customers['Age'].median(), 'Gender': 'Unknown', 'MembershipStatus': 'regular'}, inplace=True)

    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['QuantitySold'] = sales['QuantitySold'].astype(int)
    sales['SalePrice'] = sales['SalePrice'].astype(float)
    products['CostPrice'] = products['CostPrice'].astype(float)
    customers['Age'] = customers['Age'].astype(int)

    sales = sales.merge(products, on='ProductID', how='left')
    sales = sales.merge(stores, on='StoreID', how='left')
    sales = sales.merge(customers, on='CustomerID', how='left')

    # EDA
    desc_stats, sales_by_category, sales_by_store, sales_by_customer_membership, sales_by_customer_age, sales_by_customer_gender, purchase_frequency_by_age_gender = perform_eda(sales)

    # Generate customer behavior table
    customer_behavior = sales.groupby('CustomerID').agg(
        TotalSpend=('SalePrice', 'sum'),
        PurchaseFrequency=('TransactionID', 'nunique')
    ).reset_index()
    
    # Create age groups
    age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
    age_labels = ['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '66+']
    customers['AgeGroup'] = pd.cut(customers['Age'], bins=age_bins, labels=age_labels, right=False)

    # Merge customer_behavior with customers to get AgeGroup and Gender
    customer_behavior = customer_behavior.merge(customers[['CustomerID', 'AgeGroup', 'Gender']], on='CustomerID', how='left')

    # Group by AgeGroup and Gender
    customer_behavior_grouped = customer_behavior.groupby(['AgeGroup', 'Gender']).agg(
        TotalSpend=('TotalSpend', 'sum'),
        PurchaseFrequency=('PurchaseFrequency', 'sum')
    ).reset_index()

    # Check for required columns before KMeans
    if 'TotalSpend' in customer_behavior.columns and 'PurchaseFrequency' in customer_behavior.columns:
        customer_behavior['TotalSpend'] = customer_behavior['TotalSpend'].fillna(0)
        customer_behavior['PurchaseFrequency'] = customer_behavior['PurchaseFrequency'].fillna(0)

        if customer_behavior[['TotalSpend', 'PurchaseFrequency']].isnull().any().any():
            raise ValueError("NaN values found in customer_behavior dataframe after filling NaNs.")

        kmeans = KMeans(n_clusters=3)
        customer_behavior['CustomerSegment'] = kmeans.fit_predict(customer_behavior[['TotalSpend', 'PurchaseFrequency']])

        customer_behavior = customer_behavior.reset_index()

        if isinstance(customer_behavior.columns, pd.MultiIndex):
            customer_behavior.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in customer_behavior.columns.values]

        customers = customers.merge(customer_behavior[['CustomerID', 'CustomerSegment']], on='CustomerID', how='left')
        sales = sales.merge(customers[['CustomerID', 'CustomerSegment']], on='CustomerID', how='left')
    else:
        raise ValueError("Required columns 'TotalSpend' and/or 'PurchaseFrequency' not found in customer_behavior dataframe.")

    promotions_impact = sales[sales['SalePrice'] < sales['CostPrice']].groupby('StoreLocation')['SalePrice'].sum()

    basket = (sales.groupby(['TransactionID', 'ProductName'])['QuantitySold']
              .sum().unstack().reset_index().fillna(0)
              .set_index('TransactionID'))
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    frequent_items = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)

    # Calculations for best_stores and top_products
    best_stores = sales.groupby('StoreID')['SalePrice'].sum().sort_values(ascending=False).head(5)
    top_products = sales.groupby('ProductName')['QuantitySold'].sum().sort_values(ascending=False).head(5)

    # Check if ProductCategory exists before calculating top_categories
    if 'ProductCategory' in sales.columns:
        top_categories = sales.groupby('ProductCategory')['SalePrice'].sum().sort_values(ascending=False).head(5)
    else:
        top_categories = pd.Series([])  # Empty series if ProductCategory is missing

    # Filter out non-numeric columns for mean calculation
    numeric_columns = customer_behavior.select_dtypes(include='number').columns.tolist()
    marketing_strategies = customer_behavior.groupby('CustomerSegment')[numeric_columns].mean().to_dict()

    recommendations = {
        'improve_stores': best_stores.index.tolist(),
        'marketing_strategies': marketing_strategies,
        'cross_selling_opportunities': rules[['antecedents', 'consequents', 'lift']].head(10).to_dict(),
        'inventory_management': top_products.index.tolist(),
    }

    # Add merged and cleaned data to results
    merged_cleaned_data = sales.head(100).to_html(classes='table table-striped', index=False)

    results = {
        'desc_stats': desc_stats,
        'monthly_sales_plot': 'static/monthly_sales.png',
        'sales_by_category': sales_by_category.to_dict(),
        'sales_by_store': sales_by_store.to_dict(),
        'sales_by_customer_membership': sales_by_customer_membership.to_dict(),
        'sales_by_customer_age': sales_by_customer_age.to_dict(),
        'sales_by_customer_gender': sales_by_customer_gender.to_dict(),
        'top_products': top_products.to_dict(),
        'top_categories': top_categories.to_dict(),
        'best_stores': best_stores.to_dict(),
        'customer_behavior': customer_behavior_grouped.to_dict(),  # Change to grouped data
        'promotions_impact': promotions_impact.to_dict(),
        'recommendations': recommendations,
        'customer_behavior_table': customer_behavior_grouped.to_html(classes='table table-striped', index=False),  # Change to HTML table
        'purchase_frequency_by_age_gender': purchase_frequency_by_age_gender.to_html(classes='table table-striped', index=True),
        'merged_cleaned_data': merged_cleaned_data
    }

    return results, sales

def perform_eda(sales):
    desc_stats = sales.describe()
    
    # Select only numeric columns for groupby().sum() operations
    numeric_columns = sales.select_dtypes(include=[np.number]).columns
    
    sales_by_category = sales.groupby('Category')[numeric_columns].sum()['SalePrice'].sort_values(ascending=False)
    sales_by_store = sales.groupby('StoreLocation')[numeric_columns].sum()['SalePrice'].sort_values(ascending=False)
    sales_by_customer_membership = sales.groupby('MembershipStatus')[numeric_columns].sum()['SalePrice']
    sales_by_customer_age = sales.groupby('Age')[numeric_columns].sum()['SalePrice'].sort_values(ascending=False)
    sales_by_customer_gender = sales.groupby('Gender')[numeric_columns].sum()['SalePrice']
    
    # New analysis for purchase frequency by age group and gender
    sales['AgeGroup'] = pd.cut(sales['Age'], bins=[0, 18, 25, 35, 45, 55, 65, 100], labels=['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '66+'])
    purchase_frequency_by_age_gender = sales.groupby(['AgeGroup', 'Gender']).size().unstack()

    return desc_stats, sales_by_category, sales_by_store, sales_by_customer_membership, sales_by_customer_age, sales_by_customer_gender, purchase_frequency_by_age_gender



def generate_customer_behavior_table(sales, customers):
    # Calculate 'TotalSpend' and 'PurchaseFrequency'
    customer_behavior = sales.groupby('CustomerID').agg(TotalSpend=('SalePrice', 'sum'), PurchaseFrequency=('SalePrice', 'count')).reset_index()

    # Verify if the columns exist
    if 'TotalSpend' not in customer_behavior.columns or 'PurchaseFrequency' not in customer_behavior.columns:
        raise ValueError("Required columns 'TotalSpend' and/or 'PurchaseFrequency' not found in customer_behavior dataframe after aggregation.")

    customer_behavior_table = customer_behavior.to_html()
    return customer_behavior, customer_behavior_table

def export_cleaned_data(sales):
    cleaned_data_path = 'cleaned_data.csv'
    sales.to_csv(cleaned_data_path, index=False)
    return cleaned_data_path







