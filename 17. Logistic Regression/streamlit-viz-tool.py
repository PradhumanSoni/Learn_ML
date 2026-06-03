import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        X, y = make_blobs(
            n_samples=200,
            n_features=2,
            centers=2,
            random_state=6
        )

    else:  # Multiclass
        X, y = make_blobs(
            n_samples=200,
            n_features=2,
            centers=3,
            random_state=2
        )

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow")
    return X, y


def draw_meshgrid(X):
    a = np.arange(
        start=X[:, 0].min() - 1,
        stop=X[:, 0].max() + 1,
        step=0.01
    )

    b = np.arange(
        start=X[:, 1].min() - 1,
        stop=X[:, 1].max() + 1,
        step=0.01
    )

    XX, YY = np.meshgrid(a, b)

    input_array = np.c_[XX.ravel(), YY.ravel()]

    return XX, YY, input_array


plt.style.use("fivethirtyeight")

st.sidebar.title("Logistic Regression Classifier")

# Dataset Selection
dataset = st.sidebar.selectbox(
    "Select Dataset",
    ("Binary", "Multiclass")
)

# Penalty Selection
penalty = st.sidebar.selectbox(
    "Regularization",
    ("l2", "l1", "elasticnet", None)
)

# Hyperparameters
c_input = st.sidebar.number_input(
    "C",
    min_value=0.01,
    value=1.0,
    step=0.1
)

solver = st.sidebar.selectbox(
    "Solver",
    ("newton-cg", "lbfgs", "liblinear", "sag", "saga")
)

max_iter = st.sidebar.number_input(
    "Max Iterations",
    min_value=50,
    value=100,
    step=50
)

# Only used for ElasticNet
l1_ratio = st.sidebar.slider(
    "L1 Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Initial Plot
fig, ax = plt.subplots(figsize=(6, 4))

X, y = load_initial_graph(dataset, ax)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

plot_placeholder = st.pyplot(fig)

if st.sidebar.button("Run Algorithm"):

    plot_placeholder.empty()

    # Build Logistic Regression parameters
    params = {
        "penalty": penalty,
        "C": c_input,
        "solver": solver,
        "max_iter": int(max_iter)
    }

    # ElasticNet requires l1_ratio
    if penalty == "elasticnet":
        params["l1_ratio"] = l1_ratio

    try:
        clf = LogisticRegression(**params)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        XX, YY, input_array = draw_meshgrid(X)

        labels = clf.predict(input_array)

        fig2, ax2 = plt.subplots(figsize=(6, 4))

        ax2.contourf(
            XX,
            YY,
            labels.reshape(XX.shape),
            alpha=0.4,
            cmap="rainbow"
        )

        ax2.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap="rainbow",
            edgecolors="k"
        )

        ax2.set_xlabel("Feature 1")
        ax2.set_ylabel("Feature 2")
        ax2.set_title("Decision Boundary")

        st.pyplot(fig2)

        st.subheader(
            f"Accuracy: {accuracy_score(y_test, y_pred):.2f}"
        )

    except ValueError as e:
        st.error(f"Invalid parameter combination:\n\n{e}")