import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from tkinter import ttk
from cleaning import run_pipeline  # Import your ML pipeline
from PIL import Image, ImageTk  # To handle image backgrounds

class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PhishShield - Phishing Detection Tool")
        self.root.geometry("1000x600")
        self.root.resizable(False, False)

        # === Background image ===
        try:
            bg_image = Image.open("Fraud-Detection-using-Machine-Learning.jpg")  # Your background image filename
            bg_image = bg_image.resize((1000, 600), Image.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            bg_label = tk.Label(self.root, image=self.bg_photo)
            bg_label.place(relwidth=1, relheight=1)
        except Exception as e:
            print(f"Background image load error: {e}")

        # === Main container frame ===
        self.main_frame = tk.Frame(root, bg="#ffffff", bd=2)
        self.main_frame.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.9)

        # === Left Panel (Info) ===
        self.left_frame = tk.Frame(self.main_frame, bg="#e3f2fd")
        self.left_frame.place(relx=0, rely=0, relwidth=0.45, relheight=1)

        heading = tk.Label(self.left_frame, text="PhishShield", font=("Helvetica", 30, "bold"), bg="#e3f2fd", fg="#301934")
        heading.pack(pady=20)

        desc_text = (
            "PhishShield is an efficient mechanism for machine learning based phishing attack detection where users can upload a dataset of website features. It runs powerful algorithms like XGBoost, Random Forest, SVM, Logistic Regression, Naive Bayes, CNN and BiLSTM to sniff out phishing attacks. PhishShield also uses ensemble learning to combine these models for improved detection performance. With smart metrics like accuracy, precision, recall, and F1-score, it tells you which model best catches phishing!"
        )

        desc = tk.Label(self.left_frame, text=desc_text, wraplength=320, justify="left", font=("Arial", 11), bg="#e3f2fd", fg="#333")
        desc.pack(padx=20, pady=10)
                # === Image below the description ===
        try:
            self.left_image = Image.open("Fraud-Detection-using-Machine-Learning.jpg")  # Replace with your image name
            self.left_image = self.left_image.resize((300, 180), Image.LANCZOS)
            self.left_photo = ImageTk.PhotoImage(self.left_image)

            image_label = tk.Label(self.left_frame, image=self.left_photo, bg="#e3f2fd")
            image_label.pack(pady=20)
        except Exception as e:
            print(f"Image load error: {e}")


        # === Right Panel (Actions) ===
        self.right_frame = tk.Frame(self.main_frame, bg="#f1f8e9")
        self.right_frame.place(relx=0.45, rely=0, relwidth=0.55, relheight=1)

        # Buttons
        self.upload_btn = tk.Button(self.right_frame, text="Upload Dataset (CSV)", command=self.load_csv,
                                    bg="#301934", fg="white", font=("Arial", 12,"bold"), width=25, height=2)
        self.upload_btn.pack(pady=40)

        self.run_btn = tk.Button(self.right_frame, text="RUN TO FIND", command=self.run_ml,
                                 bg="#301934", fg="white", font=("Arial", 12, "bold"), width=25, height=2)
        self.run_btn.pack(pady=10)

        # Output box
        self.output = tk.Text(self.right_frame, height=12, width=50, font=("Courier", 10), wrap=tk.WORD)
        self.output.pack(pady=20)

        # Optional progress bar
        self.progress = ttk.Progressbar(self.right_frame, orient="horizontal", length=400, mode="indeterminate")
        self.progress.pack(pady=10)

        self.df = None

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            try:
                self.df = pd.read_csv(path)
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def run_ml(self):
        if self.df is None:
            messagebox.showwarning("No file", "Please upload a dataset first.")
            return

        try:
            self.progress.start()  # Start progress bar animation (optional)
            results = run_pipeline(self.df)  # Get results from your ML pipeline
            self.progress.stop()

            self.output.delete("1.0", tk.END)  # Clear previous output
            if results is None:
                self.output.insert(tk.END, "No results returned from pipeline.")
                return

            # Loop through the models and display metrics in the output box
            for model, metrics in results.items():
                self.output.insert(tk.END, f"{model}:\n")
                for metric, val in metrics.items():
                    if metric != "Confusion Matrix":
                        # If it's not a confusion matrix, format as float
                        self.output.insert(tk.END, f"  {metric}: {val:.4f}\n")
                    else:
                        # Skip printing the confusion matrix in the output box
                        self.output.insert(tk.END, f"  {metric}: (confusion matrix shown in terminal)\n")
                self.output.insert(tk.END, "\n")

        except Exception as e:
            self.output.insert(tk.END, f"Error: {str(e)}\n")
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()
