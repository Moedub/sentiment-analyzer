# simple_sentiment_analyzer.py
# Simplified version using scikit-learn (faster to run, no TensorFlow needed)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset (expandable)
reviews = [
    # Positive reviews
    ("This product is amazing! Best purchase ever.", 1),
    ("Absolutely love it! Exceeded expectations.", 1),
    ("Good value for the price. Happy with it.", 1),
    ("Fantastic! Everything I hoped for.", 1),
    ("Highly recommend! Great product.", 1),
    ("Excellent quality and fast shipping!", 1),
    ("Perfect for what I needed. Five stars!", 1),
    ("Exceeded my expectations in every way!", 1),
    ("Best decision I've made! Love it!", 1),
    ("Great features and easy to use.", 1),
    ("Outstanding quality! Will buy again.", 1),
    ("Impressed with the durability and design.", 1),
    ("Works perfectly! No complaints at all.", 1),
    ("Solid build quality and great value.", 1),
    ("My favorite purchase this year!", 1),
    ("Reliable and well-made product.", 1),
    ("Can't believe how good this is!", 1),
    ("Exactly what I was looking for!", 1),
    ("Delighted with this purchase!", 1),
    ("Great product at a fair price.", 1),
    ("Very satisfied with the quality.", 1),
    ("Superb! Works like a charm.", 1),
    ("Highly durable and functional.", 1),
    ("Best in its category!", 1),
    ("Incredible value for money.", 1),
    
    # Negative reviews
    ("Terrible quality. Complete waste of money.", 0),
    ("Worst experience. Would not recommend.", 0),
    ("Broke after one day. Very disappointing.", 0),
    ("Poor customer service and low quality.", 0),
    ("Awful. Save your money and buy elsewhere.", 0),
    ("Not worth the money. Very disappointed.", 0),
    ("Horrible experience from start to finish.", 0),
    ("Cheaply made and broke immediately.", 0),
    ("Total rip-off. Avoid at all costs.", 0),
    ("Defective product. Asked for refund.", 0),
    ("Waste of time. Doesn't work as advertised.", 0),
    ("Arrived damaged and support was useless.", 0),
    ("Overpriced garbage. Very upset.", 0),
    ("Complete disappointment. Returning it.", 0),
    ("Stopped working after two weeks.", 0),
    ("False advertising. Nothing like described.", 0),
    ("Regret buying this. Total waste.", 0),
    ("Poor quality materials throughout.", 0),
    ("Absolutely horrible product.", 0),
    ("Worst purchase I've ever made.", 0),
    ("Completely useless and poorly designed.", 0),
    ("Failed within days of purchase.", 0),
    ("Terrible build quality.", 0),
    ("Don't waste your money on this.", 0),
    ("Extremely disappointed with quality.", 0),
]

# Separate texts and labels
texts = [r[0] for r in reviews]
labels = [r[1] for r in reviews]

print("=== Simple Sentiment Analyzer ===\n")
print(f"Dataset size: {len(reviews)} reviews\n")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}\n")

# Vectorize text (convert to numbers)
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("Training model...")
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== Results ===")
print(f"Accuracy: {accuracy*100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title(f'Confusion Matrix (Accuracy: {accuracy*100:.1f}%)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'confusion_matrix.png'")

# Test on new reviews
print("\n=== Testing New Reviews ===")
test_reviews = [
    "This is absolutely wonderful! I love it!",
    "Terrible product. Very disappointed.",
    "It's okay, nothing special.",
    "Best thing I've ever bought!",
    "Waste of time and money."
]

for review in test_reviews:
    vec = vectorizer.transform([review])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = max(proba) * 100
    
    print(f"\nReview: {review}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f}%)")

print("\n=== Done! ===")
print("Model trained successfully. Check 'confusion_matrix.png' for results.")