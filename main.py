import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from wordcloud import WordCloud

# ==================================
# CREATE OUTPUT FOLDER
# ==================================

os.makedirs("outputs", exist_ok=True)

# ==================================
# LOAD DATASET
# ==================================

df = pd.read_csv("data/all_tickets_processed_improved_v3.csv")

print("Dataset Loaded Successfully")

print("\nShape:", df.shape)

print("\nCategories:\n")
print(df["Topic_group"].value_counts())

# ==================================
# FEATURES AND TARGET
# ==================================

X = df["Document"]
y = df["Topic_group"]

# ==================================
# TF-IDF
# ==================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_vectorized = vectorizer.fit_transform(X)

# ==================================
# TRAIN TEST SPLIT
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==================================
# MODEL
# ==================================

model = LinearSVC()

model.fit(X_train, y_train)

# ==================================
# PREDICTIONS
# ==================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==================================
# CATEGORY DISTRIBUTION
# ==================================

plt.figure(figsize=(10,6))

df["Topic_group"].value_counts().plot(
    kind="bar"
)

plt.title("Ticket Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "outputs/category_distribution.png"
)

plt.close()

# ==================================
# WORD CLOUD
# ==================================

text = " ".join(
    df["Document"].astype(str)
)

custom_stopwords = {
    "please", "kindly", "regards", "hello", "dear",
    "thanks", "thank", "pm", "am", "hi", "good",
    "morning", "afternoon", "evening", "sent",
    "attached", "provide", "request"
}

wordcloud = WordCloud(
    width=1400,
    height=700,
    background_color="white",
    stopwords=custom_stopwords,
    max_words=150,
    collocations=False
).generate(text)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/wordcloud.png"
)

plt.close()

# ==================================
# CONFUSION MATRIX
# ==================================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

fig, ax = plt.subplots(figsize=(12,10))
disp.plot(ax=ax, xticks_rotation=45)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.close()

# ==================================
# SAMPLE PREDICTION
# ==================================

sample_ticket = [
    "please reset my password and unlock my account"
]

sample_vector = vectorizer.transform(
    sample_ticket
)

prediction = model.predict(
    sample_vector
)

print("\nSample Ticket:")
print(sample_ticket[0])

print("\nPredicted Category:")
print(prediction[0])

# ==================================
# REPORT
# ==================================

with open(
    "outputs/report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SUPPORT TICKET CLASSIFICATION REPORT\n\n"
    )

    f.write(
        f"Accuracy: {round(accuracy*100,2)}%\n\n"
    )

    f.write(
        classification_report(
            y_test,
            y_pred
        )
    )

print("\nFiles Generated Successfully!")

print("category_distribution.png")
print("wordcloud.png")
print("confusion_matrix.png")
print("report.txt")