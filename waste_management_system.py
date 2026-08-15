
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import ttk, messagebox

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================================
# 1. LOAD DATA
# ==========================================================

data = pd.read_csv("Smart_Bin.csv")


# ==========================================================
# 2. PREPROCESSING
# ==========================================================

# Missing values
numeric_columns = data.select_dtypes(
    include=np.number
).columns

for col in numeric_columns:
    data[col] = data[col].fillna(
        data[col].median()
    )

# Remove duplicates
data = data.drop_duplicates()


# ==========================================================
# 3. ENCODING
# ==========================================================

container_encoder = LabelEncoder()
recycle_encoder = LabelEncoder()
class_encoder = LabelEncoder()

data["Container Type"] = container_encoder.fit_transform(
    data["Container Type"]
)

data["Recyclable fraction"] = recycle_encoder.fit_transform(
    data["Recyclable fraction"]
)

data["Class"] = class_encoder.fit_transform(
    data["Class"]
)


# ==========================================================
# 4. X AND Y
# ==========================================================

X = data.drop("Class", axis=1)
y = data["Class"]


# ==========================================================
# 5. TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================================
# 6. SCALING
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================================
# 7. KNN
# ==========================================================

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)


# ==========================================================
# 8. DECISION TREE
# ==========================================================

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)


# ==========================================================
# 9. XGBOOST
# ==========================================================

xgb = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)


# ==========================================================
# 10. ACCURACY
# ==========================================================

knn_acc = accuracy_score(
    y_test,
    knn_pred
)

dt_acc = accuracy_score(
    y_test,
    dt_pred
)

xgb_acc = accuracy_score(
    y_test,
    xgb_pred
)


accuracies = {
    "KNN": knn_acc,
    "Decision Tree": dt_acc,
    "XGBoost": xgb_acc
}


# ==========================================================
# 11. BEST MODEL
# ==========================================================

best_model_name = max(
    accuracies,
    key=accuracies.get
)


# ==========================================================
# 12. MODEL EVALUATION
# ==========================================================

def get_metrics(y_true, prediction):

    accuracy = accuracy_score(
        y_true,
        prediction
    )

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    return accuracy, precision, recall, f1


knn_metrics = get_metrics(
    y_test,
    knn_pred
)

dt_metrics = get_metrics(
    y_test,
    dt_pred
)

xgb_metrics = get_metrics(
    y_test,
    xgb_pred
)


# ==========================================================
# 13. TKINTER FUNCTIONS
# ==========================================================

def predict_bin():

    try:

        FL_B = float(fl_b_entry.get())
        FL_A = float(fl_a_entry.get())
        VS = float(vs_entry.get())
        FL_B_3 = float(fl_b3_entry.get())
        FL_A_3 = float(fl_a3_entry.get())
        FL_B_12 = float(fl_b12_entry.get())
        FL_A_12 = float(fl_a12_entry.get())

        container = container_combo.get()
        recyclable = recycle_combo.get()

        if container == "" or recyclable == "":
            messagebox.showwarning(
                "Missing Input",
                "Please select Container Type and Recyclable Fraction."
            )
            return

        # Encode categorical values
        container_value = container_encoder.transform(
            [container]
        )[0]

        recycle_value = recycle_encoder.transform(
            [recyclable]
        )[0]

        # New record
        new_bin = pd.DataFrame([{
            "FL_B": FL_B,
            "FL_A": FL_A,
            "VS": VS,
            "FL_B_3": FL_B_3,
            "FL_A_3": FL_A_3,
            "FL_B_12": FL_B_12,
            "FL_A_12": FL_A_12,
            "Container Type": container_value,
            "Recyclable fraction": recycle_value
        }])

        # Scaling
        new_bin_scaled = scaler.transform(
            new_bin
        )

        # Prediction
        prediction = xgb.predict(
            new_bin_scaled
        )

        # Convert 0/1 to original label
        prediction_label = class_encoder.inverse_transform(
            prediction
        )[0]


        # ==================================================
        # DISPLAY RESULT
        # ==================================================

        result_label.config(
            text=prediction_label
        )

        if prediction_label == "Emptying":

            result_label.config(
                text="🗑 EMPTYING",
                foreground="#e74c3c"
            )

            action_label.config(
                text="⚠ Bin should be emptied.",
                foreground="#e74c3c"
            )

        else:

            result_label.config(
                text="✓ NON EMPTYING",
                foreground="#27ae60"
            )

            action_label.config(
                text="✓ Bin does not need emptying yet.",
                foreground="#27ae60"
            )

    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numerical values."
        )


# ==========================================================
# CLEAR FUNCTION
# ==========================================================

def clear_fields():

    entries = [
        fl_b_entry,
        fl_a_entry,
        vs_entry,
        fl_b3_entry,
        fl_a3_entry,
        fl_b12_entry,
        fl_a12_entry
    ]

    for entry in entries:
        entry.delete(0, tk.END)

    container_combo.set("")
    recycle_combo.set("")

    result_label.config(
        text="Prediction",
        foreground="#34495e"
    )

    action_label.config(
        text="Enter bin information",
        foreground="#555555"
    )


# ==========================================================
# SHOW MODEL COMPARISON
# ==========================================================

def show_models():

    model_window = tk.Toplevel(root)

    model_window.title(
        "Model Performance"
    )

    model_window.geometry(
        "750x450"
    )

    model_window.configure(
        bg="#f4f6f7"
    )

    title = tk.Label(
        model_window,
        text="ML Model Comparison",
        font=("Arial", 20, "bold"),
        bg="#f4f6f7",
        fg="#2c3e50"
    )

    title.pack(pady=20)


    # Table
    columns = (
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    )

    table = ttk.Treeview(
        model_window,
        columns=columns,
        show="headings",
        height=5
    )

    for col in columns:

        table.heading(
            col,
            text=col
        )

        table.column(
            col,
            width=130,
            anchor="center"
        )

    table.pack(
        padx=20,
        pady=20
    )


    table.insert(
        "",
        "end",
        values=(
            "KNN",
            f"{knn_metrics[0]*100:.2f}%",
            f"{knn_metrics[1]*100:.2f}%",
            f"{knn_metrics[2]*100:.2f}%",
            f"{knn_metrics[3]*100:.2f}%"
        )
    )

    table.insert(
        "",
        "end",
        values=(
            "Decision Tree",
            f"{dt_metrics[0]*100:.2f}%",
            f"{dt_metrics[1]*100:.2f}%",
            f"{dt_metrics[2]*100:.2f}%",
            f"{dt_metrics[3]*100:.2f}%"
        )
    )

    table.insert(
        "",
        "end",
        values=(
            "XGBoost",
            f"{xgb_metrics[0]*100:.2f}%",
            f"{xgb_metrics[1]*100:.2f}%",
            f"{xgb_metrics[2]*100:.2f}%",
            f"{xgb_metrics[3]*100:.2f}%"
        )
    )


    best_label = tk.Label(
        model_window,
        text=f"🏆 Best Algorithm: {best_model_name}",
        font=("Arial", 16, "bold"),
        bg="#f4f6f7",
        fg="#27ae60"
    )

    best_label.pack(
        pady=20
    )


# ==========================================================
# SHOW GRAPH
# ==========================================================

def show_graph():

    models = list(accuracies.keys())
    values = list(accuracies.values())

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        models,
        values
    )

    plt.xlabel(
        "Algorithm"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "ML Algorithm Accuracy Comparison"
    )

    plt.ylim(
        0,
        1
    )

    for i, value in enumerate(values):

        plt.text(
            i,
            value + 0.02,
            f"{value*100:.2f}%",
            ha="center"
        )

    plt.tight_layout()

    plt.show()


# ==========================================================
# 14. MAIN TKINTER WINDOW
# ==========================================================

root = tk.Tk()

root.title(
    "Smart Waste Management System"
)

root.geometry(
    "1000x720"
)

root.configure(
    bg="#ecf0f1"
)


# ==========================================================
# HEADER
# ==========================================================

header = tk.Frame(
    root,
    bg="#2c3e50",
    height=90
)

header.pack(
    fill="x"
)

title = tk.Label(
    header,
    text="🗑 Smart Waste Management System",
    font=("Arial", 24, "bold"),
    bg="#2c3e50",
    fg="white"
)

title.pack(
    pady=15
)

subtitle = tk.Label(
    header,
    text="Machine Learning Based Smart Bin Emptying Prediction",
    font=("Arial", 11),
    bg="#2c3e50",
    fg="#d5dbdb"
)

subtitle.pack()


# ==========================================================
# MAIN FRAME
# ==========================================================

main_frame = tk.Frame(
    root,
    bg="#ecf0f1"
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=30,
    pady=20
)


# ==========================================================
# INPUT FRAME
# ==========================================================

input_frame = tk.LabelFrame(
    main_frame,
    text="  Smart Bin Information  ",
    font=("Arial", 14, "bold"),
    bg="white",
    fg="#2c3e50",
    padx=20,
    pady=20
)

input_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=10
)


# ==========================================================
# ENTRY CREATION
# ==========================================================

def create_input(
    parent,
    text,
    row,
    default=""
):

    label = tk.Label(
        parent,
        text=text,
        font=("Arial", 11),
        bg="white",
        fg="#34495e"
    )

    label.grid(
        row=row,
        column=0,
        sticky="w",
        pady=7
    )

    entry = ttk.Entry(
        parent,
        width=25
    )

    entry.grid(
        row=row,
        column=1,
        pady=7,
        padx=10
    )

    if default != "":
        entry.insert(
            0,
            default
        )

    return entry


fl_b_entry = create_input(
    input_frame,
    "FL_B:",
    0,
    "85.3"
)

fl_a_entry = create_input(
    input_frame,
    "FL_A:",
    1,
    "34.4"
)

vs_entry = create_input(
    input_frame,
    "VS:",
    2,
    "6"
)

fl_b3_entry = create_input(
    input_frame,
    "FL_B_3:",
    3,
    "75.9"
)

fl_a3_entry = create_input(
    input_frame,
    "FL_A_3:",
    4,
    "54.7"
)

fl_b12_entry = create_input(
    input_frame,
    "FL_B_12:",
    5,
    "83.3"
)

fl_a12_entry = create_input(
    input_frame,
    "FL_A_12:",
    6,
    "39.8"
)


# ==========================================================
# COMBOBOXES
# ==========================================================

tk.Label(
    input_frame,
    text="Container Type:",
    font=("Arial", 11),
    bg="white",
    fg="#34495e"
).grid(
    row=7,
    column=0,
    sticky="w",
    pady=7
)


container_combo = ttk.Combobox(
    input_frame,
    values=list(
        container_encoder.classes_
    ),
    width=22,
    state="readonly"
)

container_combo.grid(
    row=7,
    column=1,
    pady=7,
    padx=10
)


tk.Label(
    input_frame,
    text="Recyclable Fraction:",
    font=("Arial", 11),
    bg="white",
    fg="#34495e"
).grid(
    row=8,
    column=0,
    sticky="w",
    pady=7
)


recycle_combo = ttk.Combobox(
    input_frame,
    values=list(
        recycle_encoder.classes_
    ),
    width=22,
    state="readonly"
)

recycle_combo.grid(
    row=8,
    column=1,
    pady=7,
    padx=10
)


# ==========================================================
# BUTTONS
# ==========================================================

predict_button = tk.Button(
    input_frame,
    text="🔍 Predict Bin Status",
    command=predict_bin,
    font=("Arial", 12, "bold"),
    bg="#27ae60",
    fg="white",
    activebackground="#229954",
    activeforeground="white",
    padx=15,
    pady=8,
    relief="flat",
    cursor="hand2"
)

predict_button.grid(
    row=9,
    column=0,
    columnspan=2,
    pady=20
)


clear_button = tk.Button(
    input_frame,
    text="Clear",
    command=clear_fields,
    font=("Arial", 11),
    bg="#95a5a6",
    fg="white",
    padx=20,
    pady=5,
    relief="flat",
    cursor="hand2"
)

clear_button.grid(
    row=10,
    column=0,
    columnspan=2
)


# ==========================================================
# RESULT FRAME
# ==========================================================

result_frame = tk.LabelFrame(
    main_frame,
    text="  Prediction Result  ",
    font=("Arial", 14, "bold"),
    bg="white",
    fg="#2c3e50",
    padx=20,
    pady=20
)

result_frame.pack(
    side="right",
    fill="both",
    expand=True,
    padx=10
)


result_label = tk.Label(
    result_frame,
    text="Prediction",
    font=("Arial", 25, "bold"),
    bg="white",
    fg="#34495e"
)

result_label.pack(
    pady=60
)


action_label = tk.Label(
    result_frame,
    text="Enter bin information",
    font=("Arial", 13),
    bg="white",
    fg="#555555",
    wraplength=300
)

action_label.pack(
    pady=10
)


# ==========================================================
# MODEL INFORMATION
# ==========================================================

best_model_display = tk.Label(
    result_frame,
    text=f"Best Model\n{best_model_name}\n\nAccuracy\n{accuracies[best_model_name]*100:.2f}%",
    font=("Arial", 15, "bold"),
    bg="white",
    fg="#2980b9"
)

best_model_display.pack(
    pady=40
)


# ==========================================================
# BOTTOM BUTTONS
# ==========================================================

bottom_frame = tk.Frame(
    root,
    bg="#ecf0f1"
)

bottom_frame.pack(
    pady=15
)


model_button = tk.Button(
    bottom_frame,
    text="📊 Model Comparison",
    command=show_models,
    font=("Arial", 11, "bold"),
    bg="#3498db",
    fg="white",
    padx=15,
    pady=8,
    relief="flat",
    cursor="hand2"
)

model_button.pack(
    side="left",
    padx=10
)


graph_button = tk.Button(
    bottom_frame,
    text="📈 Accuracy Graph",
    command=show_graph,
    font=("Arial", 11, "bold"),
    bg="#8e44ad",
    fg="white",
    padx=15,
    pady=8,
    relief="flat",
    cursor="hand2"
)

graph_button.pack(
    side="left",
    padx=10
)


# ==========================================================
# START APPLICATION
# ==========================================================

root.mainloop()
