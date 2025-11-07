"""
Instagram Content Performance Analysis
A comprehensive EDA project for optimizing social media content strategy

Author: Sakshi  
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("📊 INSTAGRAM CONTENT PERFORMANCE ANALYSIS")
print("="*80)

# ============================================================================
# QUESTION 1: What does the dataset look like?
# ============================================================================
print("\n🔍 QUESTION 1: Dataset Structure")
print("-"*80)

df = pd.read_csv('data/Instagram-data.csv')

print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print("\n\nData Types:")
print(df.dtypes)

print("\n\nFirst 3 Rows:")
print(df.head(3))

# ============================================================================
# QUESTION 2: Basic Statistics
# ============================================================================
print("\n\n🔍 QUESTION 2: Basic Statistics")
print("-"*80)

total_posts = len(df)
avg_impressions = df['Impressions'].mean()
max_likes = df['Likes'].max()
min_likes = df['Likes'].min()
avg_likes = df['Likes'].mean()

print(f"\n📌 Total Posts: {total_posts}")
print(f"\n📊 Impressions:")
print(f"   • Average: {avg_impressions:,.2f}")
print(f"   • Total: {df['Impressions'].sum():,}")
print(f"   • Maximum: {df['Impressions'].max():,}")
print(f"   • Minimum: {df['Impressions'].min():,}")

print(f"\n❤️  Likes:")
print(f"   • Maximum: {max_likes}")
print(f"   • Minimum: {min_likes}")
print(f"   • Average: {avg_likes:.2f}")

print(f"\n💬 Other Metrics (Average):")
print(f"   • Comments: {df['Comments'].mean():.2f}")
print(f"   • Shares: {df['Shares'].mean():.2f}")
print(f"   • Saves: {df['Saves'].mean():.2f}")
print(f"   • Profile Visits: {df['Profile Visits'].mean():.2f}")
print(f"   • Follows: {df['Follows'].mean():.2f}")

# ============================================================================
# QUESTION 3: Missing Values
# ============================================================================
print("\n\n🔍 QUESTION 3: Missing Values Analysis")
print("-"*80)

missing_count = df.isnull().sum()
missing_data = pd.DataFrame({
    'Column': missing_count.index,
    'Missing_Count': missing_count.values,
    'Percentage': (missing_count.values / len(df) * 100).round(2)
})

missing_data = missing_data[missing_data['Missing_Count'] > 0]

if len(missing_data) > 0:
    print("\n⚠️  Columns with Missing Values:")
    print(missing_data.to_string(index=False))
    print(f"\nTotal rows affected: {df.isnull().any(axis=1).sum()}")
else:
    print("\n✅ No missing values found! Dataset is clean!")

# ============================================================================
# QUESTION 4: Engagement Analysis
# ============================================================================
print("\n\n🔍 QUESTION 4: Post Performance & Engagement")
print("-"*80)

# Calculate engagement metrics
# Calculate engagement metrics
df['Engagement_Rate'] = ((df['Likes'] + df['Comments'] + df['Shares'] + df['Saves']) / df['Impressions']) * 100
df['Like_Rate'] = (df['Likes'] / df['Impressions']) * 100
df['Save_Rate'] = (df['Saves'] / df['Impressions']) * 100
df['Comment_Rate'] = (df['Comments'] / df['Impressions']) * 100
df['Share_Rate'] = (df['Shares'] / df['Impressions']) * 100

print(f"\n📈 Average Engagement Rate: {df['Engagement_Rate'].mean():.2f}%")
print(f"📈 Average Like Rate: {df['Like_Rate'].mean():.2f}%")
print(f"📈 Average Save Rate: {df['Save_Rate'].mean():.2f}%")

print("\n🏆 Top 5 Posts by Engagement Rate:")
top_posts = df.nlargest(5, 'Engagement_Rate')[['Date', 'Impressions', 'Likes', 'Engagement_Rate']]
print(top_posts.to_string(index=False))

# ============================================================================
# QUESTION 5: Engagement Ratios (Likes & Saves to Impressions)
# ============================================================================
print("\n\n🔍 QUESTION 5: Engagement Ratios Analysis")
print("-"*80)

print(f"\n📊 Like to Impression Ratio: {df['Like_Rate'].mean():.2f}%")
print(f"   (On average, {df['Like_Rate'].mean():.2f}% of people who see a post, like it)")

print(f"\n💾 Save to Impression Ratio: {df['Save_Rate'].mean():.2f}%")
print(f"   (On average, {df['Save_Rate'].mean():.2f}% of people who see a post, save it)")

print(f"\n💡 Insight: Every 100 impressions generate:")
print(f"   • {df['Like_Rate'].mean():.1f} likes")
print(f"   • {df['Save_Rate'].mean():.1f} saves")
print(f"   • {df['Comment_Rate'].mean():.1f} comments")
print(f"   • {df['Share_Rate'].mean():.1f} shares")

# ============================================================================
# QUESTION 6: Temporal Analysis (Date-wise Performance)
# ============================================================================
print("\n\n🔍 QUESTION 6: Date-wise Performance Analysis")
print("-"*80)

# Convert date
df['Date'] = pd.to_datetime(df['Date'])
df['Day_of_Week'] = df['Date'].dt.day_name()
df['Month'] = df['Date'].dt.month_name()

# Best performing days
print("\n📅 Top 10 Best Performing Days (by Impressions):")
daily_perf = df.groupby('Date').agg({
    'Impressions': 'sum',
    'Likes': 'sum'
}).sort_values('Impressions', ascending=False).head(10)
print(daily_perf)

# Day of week analysis
print("\n\n📆 Performance by Day of Week:")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_perf = df.groupby('Day_of_Week').agg({
    'Impressions': 'mean',
    'Likes': 'mean'
}).round(2)
dow_perf = dow_perf.reindex([d for d in day_order if d in dow_perf.index])
print(dow_perf)

best_day = dow_perf['Impressions'].idxmax()
print(f"\n🏆 Best Day to Post: {best_day} ({dow_perf.loc[best_day, 'Impressions']:,.0f} avg impressions)")

# ============================================================================
# QUESTION 7: Outlier Detection
# ============================================================================
print("\n\n🔍 QUESTION 7: Outlier Detection (Unusual Performance)")
print("-"*80)

mean_imp = df['Impressions'].mean()
std_imp = df['Impressions'].std()

upper_threshold = mean_imp + (2 * std_imp)
lower_threshold = mean_imp - (2 * std_imp)

high_performers = df[df['Impressions'] > upper_threshold]
low_performers = df[df['Impressions'] < lower_threshold]

print(f"\n📊 Average Impressions: {mean_imp:,.2f}")
print(f"📊 Standard Deviation: {std_imp:,.2f}")

print(f"\n🔥 High Performers (>{upper_threshold:,.0f} impressions): {len(high_performers)} posts")
print(f"❄️  Low Performers (<{lower_threshold:,.0f} impressions): {len(low_performers)} posts")

if len(high_performers) > 0:
    print(f"\n🌟 Top 5 High-Performing Posts:")
    print(high_performers.nlargest(5, 'Impressions')[['Date', 'Impressions', 'Likes', 'Saves']].to_string(index=False))

# ============================================================================
# BONUS: Traffic Source Analysis
# ============================================================================
print("\n\n🎁 BONUS ANALYSIS: Traffic Sources")
print("-"*80)

sources = ['From Home', 'From Hashtags', 'From Explore', 'From Other']
print("\n📍 Impressions by Source:")
for source in sources:
    total = df[source].sum()
    percentage = (total / df['Impressions'].sum()) * 100
    print(f"   • {source}: {total:,} ({percentage:.1f}%)")

# ============================================================================
# Create Visualizations
# ============================================================================
print("\n\n📊 Creating Visualizations...")
print("-"*80)

fig = plt.figure(figsize=(20, 12))

# 1. Impressions over time
ax1 = plt.subplot(3, 3, 1)
df.plot(x='Date', y='Impressions', ax=ax1, color='#2E86AB', linewidth=2, legend=False)
ax1.set_title('Impressions Over Time', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Impressions')
plt.xticks(rotation=45)

# 2. Engagement metrics
ax2 = plt.subplot(3, 3, 2)
metrics = df[['Likes', 'Comments', 'Shares', 'Saves']].mean()
ax2.bar(metrics.index, metrics.values, color=['#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
ax2.set_title('Average Engagement Metrics', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Count')
plt.xticks(rotation=45)

# 3. Engagement rate distribution
ax3 = plt.subplot(3, 3, 3)
ax3.hist(df['Engagement_Rate'], bins=30, color='#2E86AB', edgecolor='black', alpha=0.7)
ax3.set_title('Engagement Rate Distribution', fontsize=12, fontweight='bold')
ax3.set_xlabel('Engagement Rate (%)')
ax3.set_ylabel('Frequency')

# 4. Traffic sources pie chart
ax4 = plt.subplot(3, 3, 4)
source_data = df[sources].sum()
ax4.pie(source_data, labels=source_data.index, autopct='%1.1f%%', 
        colors=['#2E86AB', '#A23B72', '#F18F01', '#6A994E'])
ax4.set_title('Impressions by Source', fontsize=12, fontweight='bold')

# 5. Likes vs Impressions
ax5 = plt.subplot(3, 3, 5)
ax5.scatter(df['Impressions'], df['Likes'], alpha=0.5, color='#2E86AB')
ax5.set_title('Likes vs Impressions', fontsize=12, fontweight='bold')
ax5.set_xlabel('Impressions')
ax5.set_ylabel('Likes')

# 6. Day of week performance
ax6 = plt.subplot(3, 3, 6)
dow_data = df.groupby('Day_of_Week')['Impressions'].mean()
dow_data = dow_data.reindex([d for d in day_order if d in dow_data.index])
dow_data.plot(kind='bar', ax=ax6, color='#A23B72')
ax6.set_title('Avg Impressions by Day', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)

# 7. Profile Visits vs Follows
ax7 = plt.subplot(3, 3, 7)
ax7.scatter(df['Profile Visits'], df['Follows'], alpha=0.5, color='#F18F01')
ax7.set_title('Profile Visits vs Follows', fontsize=12, fontweight='bold')
ax7.set_xlabel('Profile Visits')
ax7.set_ylabel('Follows')

# 8. Top 10 posts
ax8 = plt.subplot(3, 3, 8)
top10 = df.nlargest(10, 'Impressions')[['Date', 'Impressions']].set_index('Date')
top10.plot(kind='barh', ax=ax8, color='#6A994E', legend=False)
ax8.set_title('Top 10 Posts', fontsize=12, fontweight='bold')

# 9. Correlation heatmap
ax9 = plt.subplot(3, 3, 9)
corr_cols = ['Impressions', 'Likes', 'Comments', 'Shares', 'Saves', 'Profile Visits', 'Follows']
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax9, square=True)
ax9.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/instagram_dashboard.png', dpi=300, bbox_inches='tight')
print("✅ Dashboard saved: visualizations/instagram_dashboard.png")

# Save processed data
df.to_csv('data/instagram_data_processed.csv', index=False)
print("✅ Processed data saved: data/instagram_data_processed.csv")

print("\n" + "="*80)
print("✨ ANALYSIS COMPLETE!")
print("="*80)
print("\n📁 Files created:")
print("   • visualizations/instagram_dashboard.png")
print("   • data/instagram_data_processed.csv")