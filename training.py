import pandas as pd
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support, r2_score
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv('dataset.csv')

# normalization
df["estimasi_deadline"] = df["estimasi_deadline"].str.replace(" hari", "").astype(int)
df["estimasi_budget"] = df["estimasi_budget"].astype(int)
df["ui/ux"] = df["ui/ux"].map({"Ya": 1, "Tidak": 0})
df["hosting"] = df["hosting"].map({"Ya": 1, "Tidak": 0})

# Vectorizer
vectorizer = TfidfVectorizer(stop_words=stopwords.words('indonesian'))
X = vectorizer.fit_transform(df["deskripsi"])

# Pisahkan label untuk klasifikasi dan regresi
y_class = df[["tujuan", "platform", "industri", "ui/ux", "hosting"]].astype(str)
y_reg = df[["estimasi_deadline", "estimasi_budget"]]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Klasifikasi
clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

print("\nLaporan Precision, Recall, F1-Score, dan Support untuk Klasifikasi:")
for i, col in enumerate(y_class.columns):
    precision, recall, f1, support = precision_recall_fscore_support(y_test_c.iloc[:, i], y_pred_c[:, i], average='weighted')
    print(f"\n{col}:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"Support: {support}")


# Regresi
reg = MultiOutputRegressor(RandomForestRegressor())
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)

print("\nR2 Score untuk Regresi:")
for i, col in enumerate(y_reg.columns):
    r2 = r2_score(y_test_r.iloc[:, i], y_pred_r[:, i])
    print(f"{col}: {r2:.2f}")


joblib.dump(vectorizer, 'model/vectorizer.pkl')
joblib.dump(clf, 'model/klasifikasi.pkl')
joblib.dump(reg, 'model/regresi.pkl')