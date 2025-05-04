import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import re
import joblib
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and preprocessing tools
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

stop_words = set(stopwords.words('english'))

# Clean text function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# GUI App
class SentimentApp:
    def __init__(self, master):
        self.master = master
        master.title("📊 Sentiment Prediction Tool")
        master.geometry("500x350")

        self.label = tk.Label(master, text="Upload your reviews CSV", font=('Arial', 12))
        self.label.pack(pady=10)

        self.upload_btn = tk.Button(master, text="Upload CSV", command=self.upload_csv)
        self.upload_btn.pack(pady=5)

        self.result_label = tk.Label(master, text="", font=('Arial', 11), justify="left")
        self.result_label.pack(pady=10)

        self.graph_btn = tk.Button(master, text="Show Charts", command=self.show_charts)
        self.graph_btn.pack(pady=5)
        self.graph_btn.config(state='disabled')

        self.save_btn = tk.Button(master, text="Save Predictions", command=self.save_results)
        self.save_btn.pack(pady=5)
        self.save_btn.config(state='disabled')

        self.data = None

    def upload_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return

        try:
            self.data = pd.read_csv(filepath)
            self.data['Review_Header'] = self.data['Review_Header'].fillna('')
            self.data['Review_text'] = self.data['Review_text'].fillna('')
            self.data['Full_Review'] = self.data['Review_Header'] + " " + self.data['Review_text']
            self.data['Cleaned_Review'] = self.data['Full_Review'].apply(clean_text)

            X_new = vectorizer.transform(self.data['Cleaned_Review'])
            predictions = model.predict(X_new)
            predicted_labels = label_encoder.inverse_transform(predictions)

            self.data['Predicted_Sentiment'] = predicted_labels

            counts = self.data['Predicted_Sentiment'].value_counts().to_dict()
            positive = counts.get('Positive', 0)
            neutral = counts.get('Neutral', 0)
            negative = counts.get('Negative', 0)

            result_text = f"✔ Total Predictions: {len(predicted_labels)}\n\n"
            result_text += f"🟢 Positive: {positive}\n"
            result_text += f"🟡 Neutral: {neutral}\n"
            result_text += f"🔴 Negative: {negative}\n"

            self.result_label.config(text=result_text)
            self.save_btn.config(state='normal')
            self.graph_btn.config(state='normal')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file:\n{e}")

    def save_results(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")])
        if save_path:
            self.data.to_csv(save_path, index=False)
            messagebox.showinfo("Success", "Predictions saved successfully!")

    def show_charts(self):
        if self.data is not None:
            plt.figure(figsize=(6, 4))
            sns.countplot(x='Predicted_Sentiment', data=self.data,
                          order=['Positive', 'Neutral', 'Negative'], palette='Set2')
            plt.title("Sentiment Distribution")
            plt.xlabel("Sentiment")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()

            # Pie chart
            sentiment_counts = self.data['Predicted_Sentiment'].value_counts()
            plt.figure(figsize=(5, 5))
            plt.pie(sentiment_counts, labels=sentiment_counts.index,
                    autopct='%1.1f%%', colors=['green', 'gold', 'red'], startangle=140)
            plt.title("Sentiment Pie Chart")
            plt.tight_layout()
            plt.show()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()
