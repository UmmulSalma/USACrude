from pathlib import Path
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hybrid Forecasting App", page_icon="📈", layout="wide")

FEATURE_COLUMNS = ["Production", "Lag1", "Lag2", "oil price"]
BILSTM_ROWS = 3
TIDE_ROWS = 8
MAX_INPUT = 99.99
MIN_INPUT = 0.00
DECIMALS = 2

REQUIRED_FILES = {
    "Bi-LSTM + SVR": [
        "bilstm_model.keras",
        "svr_model.pkl",
        "scaler_features.pkl",
        "scaler_target.pkl",
        "hybrid_meta.pkl",
    ],
    "TiDE + SVR": [
        "tide_final.pt",
        "tide_svr_corrector.pkl",
        "tide_target_scaler.pkl",
        "tide_cov_scaler.pkl",
    ],
}


def resolve_model_dir() -> Path:
    base_dir = Path(__file__).parent
    candidates = [base_dir / "models", base_dir]
    for candidate in candidates:
        if any((candidate / filename).exists() for filename in sum(REQUIRED_FILES.values(), [])):
            return candidate
    return base_dir / "models"


MODEL_DIR = resolve_model_dir()


@st.cache_data
def build_default_df(rows: int, row_prefix: str) -> pd.DataFrame:
    index = [f"{row_prefix} {i}" for i in range(1, rows + 1)]
    return pd.DataFrame(
        {column: [0.00] * rows for column in FEATURE_COLUMNS},
        index=index,
    )


@st.cache_data
def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


@st.cache_data
def get_missing_files(model_name: str, model_dir: str) -> list[str]:
    base = Path(model_dir)
    return [filename for filename in REQUIRED_FILES[model_name] if not (base / filename).exists()]


@st.cache_resource
def load_bilstm_bundle(model_dir: str):
    from tensorflow.keras.models import load_model

    model_path = Path(model_dir)
    bilstm_model = load_model(model_path / "bilstm_model.keras", compile=False)
    svr_model = joblib.load(model_path / "svr_model.pkl")
    scaler_features = joblib.load(model_path / "scaler_features.pkl")
    scaler_target = joblib.load(model_path / "scaler_target.pkl")
    meta = joblib.load(model_path / "hybrid_meta.pkl")
    return bilstm_model, svr_model, scaler_features, scaler_target, meta


@st.cache_resource
def load_tide_bundle(model_dir: str):
    from darts.models import TiDEModel

    model_path = Path(model_dir)
    tide_model = TiDEModel.load(str(model_path / "tide_final.pt"), map_location="cpu")
    tide_model.to_cpu()
    svr_model = joblib.load(model_path / "tide_svr_corrector.pkl")
    scaler_target = joblib.load(model_path / "tide_target_scaler.pkl")
    scaler_covs = joblib.load(model_path / "tide_cov_scaler.pkl")
    return tide_model, svr_model, scaler_target, scaler_covs


def init_editor_state(state_key: str, rows: int, row_prefix: str) -> None:
    if state_key not in st.session_state:
        st.session_state[state_key] = build_default_df(rows, row_prefix)


def reset_editor_state(state_key: str, editor_key: str, rows: int, row_prefix: str) -> None:
    st.session_state[state_key] = build_default_df(rows, row_prefix)
    st.session_state.pop(editor_key, None)


def render_editor(df: pd.DataFrame, editor_key: str) -> pd.DataFrame:
    return st.data_editor(
        df,
        key=editor_key,
        num_rows="fixed",
        use_container_width=True,
        hide_index=False,
        column_config={
            column: st.column_config.NumberColumn(
                column,
                help=f"Enter {column} from {MIN_INPUT:.2f} to {MAX_INPUT:.2f}",
                min_value=MIN_INPUT,
                max_value=MAX_INPUT,
                step=0.01,
                format="%.2f",
                required=True,
            )
            for column in FEATURE_COLUMNS
        },
    )


def validate_input(df: pd.DataFrame, expected_rows: int) -> pd.DataFrame:
    if df.shape[0] != expected_rows:
        raise ValueError(f"Expected exactly {expected_rows} rows, but got {df.shape[0]}.")

    work_df = df.copy()
    work_df = work_df.reindex(columns=FEATURE_COLUMNS)

    for column in FEATURE_COLUMNS:
        work_df[column] = pd.to_numeric(work_df[column], errors="coerce")

    if work_df[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Every cell must contain a valid number.")

    if ((work_df[FEATURE_COLUMNS] < MIN_INPUT) | (work_df[FEATURE_COLUMNS] > MAX_INPUT)).any().any():
        raise ValueError(f"All values must stay between {MIN_INPUT:.2f} and {MAX_INPUT:.2f}.")

    return work_df.round(DECIMALS)


def predict_bilstm_hybrid(input_df: pd.DataFrame) -> dict:
    bilstm_model, svr_model, scaler_features, scaler_target, meta = load_bilstm_bundle(str(MODEL_DIR))

    time_steps = int(meta.get("TIME_STEPS", BILSTM_ROWS))
    feature_columns = list(meta.get("feature_columns", FEATURE_COLUMNS))

    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(f"Metadata columns do not match the app columns. Found: {feature_columns}")

    if len(input_df) != time_steps:
        raise ValueError(f"Bi-LSTM model requires exactly {time_steps} rows.")

    window = input_df[feature_columns].to_numpy(dtype=np.float32)
    scaled_window = scaler_features.transform(window)

    x_input = scaled_window.reshape(1, time_steps, len(feature_columns))
    base_scaled = bilstm_model.predict(x_input, verbose=0).reshape(-1, 1)

    svr_input = np.hstack([scaled_window[-1, :].reshape(1, -1), base_scaled.reshape(1, -1)])
    correction_scaled = svr_model.predict(svr_input).reshape(-1, 1)
    hybrid_scaled = base_scaled + correction_scaled

    base_prediction = float(scaler_target.inverse_transform(base_scaled)[0, 0])
    hybrid_prediction = float(scaler_target.inverse_transform(hybrid_scaled)[0, 0])

    return {
        "base_prediction": base_prediction,
        "hybrid_prediction": hybrid_prediction,
        "correction": hybrid_prediction - base_prediction,
    }


def predict_tide_hybrid(input_df: pd.DataFrame, next_oil_price: float) -> dict:
    from darts import TimeSeries

    tide_model, svr_model, scaler_target, scaler_covs = load_tide_bundle(str(MODEL_DIR))

    if len(input_df) != TIDE_ROWS:
        raise ValueError(f"TiDE model requires exactly {TIDE_ROWS} rows.")

    work_df = input_df.copy()
    work_df["Time"] = pd.date_range(start="2000-01-01", periods=len(work_df), freq="MS")

    target_ts = TimeSeries.from_dataframe(work_df, time_col="Time", value_cols="Production")
    cov_ts = TimeSeries.from_dataframe(work_df, time_col="Time", value_cols=["Lag1", "Lag2", "oil price"])

    target_scaled = scaler_target.transform(target_ts)
    cov_scaled = scaler_covs.transform(cov_ts)

    pred_scaled = tide_model.predict(n=1, series=target_scaled, past_covariates=cov_scaled)
    base_prediction = float(scaler_target.inverse_transform(pred_scaled).values().flatten()[0])

    next_lag1 = float(work_df["Production"].iloc[-1])
    next_lag2 = float(work_df["Production"].iloc[-2])

    svr_features = np.array([[next_lag1, next_lag2, float(next_oil_price), base_prediction]], dtype=np.float32)
    correction = float(svr_model.predict(svr_features)[0])

    return {
        "base_prediction": base_prediction,
        "hybrid_prediction": base_prediction + correction,
        "correction": correction,
        "next_lag1": next_lag1,
        "next_lag2": next_lag2,
    }


def render_results(results: dict, base_label: str) -> None:
    left, middle, right = st.columns(3)
    left.metric("Hybrid prediction", f"{results['hybrid_prediction']:.6f}")
    middle.metric(base_label, f"{results['base_prediction']:.6f}")
    right.metric("SVR correction", f"{results['correction']:+.6f}")


st.title("Hybrid Forecasting App")
st.caption(
    "Choose either Bi-LSTM + SVR or TiDE + SVR, then enter the same history length used during training."
)

with st.sidebar:
    st.header("Model selection")
    selected_model = st.radio(
        "Choose a prediction model",
        ["Bi-LSTM + SVR", "TiDE + SVR"],
        help="Both models make a one-step-ahead prediction.",
    )

    st.markdown("---")
    st.subheader("Input rules")
    st.write("• Values must be between 0.00 and 99.99")
    st.write("• Up/down arrows and manual typing are both allowed")
    st.write("• Enter rows from oldest to newest")
    st.write(f"• Columns: {', '.join(FEATURE_COLUMNS)}")

    st.markdown("---")
    st.subheader("Model files")
    st.code(str(MODEL_DIR), language="text")

missing_files = get_missing_files(selected_model, str(MODEL_DIR))
if missing_files:
    st.error(
        "Missing files for the selected model: " + ", ".join(missing_files) +
        ". Put the files inside a `models/` folder in your GitHub repo, or alongside `app.py`."
    )
    st.stop()

intro_left, intro_right = st.columns([1.4, 1])
with intro_left:
    if selected_model == "Bi-LSTM + SVR":
        st.subheader("Bi-LSTM + SVR")
        st.info(
            "Enter exactly 3 rows. The app uses the final row together with the Bi-LSTM output to build the SVR correction."
        )
    else:
        st.subheader("TiDE + SVR")
        st.info(
            "Enter exactly 8 historical rows. For the SVR correction, also provide the next-step oil price below."
        )
with intro_right:
    st.markdown("**Tips**")
    st.write("- Keep the row order from oldest to newest.")
    st.write("- Lag1 and Lag2 should match your training data format.")
    st.write("- Use the reset button anytime to start from zeros.")

if selected_model == "Bi-LSTM + SVR":
    state_key = "bilstm_df"
    editor_key = "bilstm_editor"
    row_prefix = "Row"
    init_editor_state(state_key, BILSTM_ROWS, row_prefix)

    action_left, action_right = st.columns([1, 1])
    with action_left:
        st.download_button(
            label="Download 3-row CSV template",
            data=dataframe_to_csv(build_default_df(BILSTM_ROWS, row_prefix)),
            file_name="bilstm_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with action_right:
        if st.button("Reset Bi-LSTM table", use_container_width=True):
            reset_editor_state(state_key, editor_key, BILSTM_ROWS, row_prefix)
            st.rerun()

    with st.form("bilstm_form"):
        edited_df = render_editor(st.session_state[state_key], editor_key)
        submitted = st.form_submit_button("Predict with Bi-LSTM + SVR", type="primary", use_container_width=True)

    st.session_state[state_key] = edited_df

    if submitted:
        try:
            clean_df = validate_input(edited_df, BILSTM_ROWS)
            with st.spinner("Running Bi-LSTM + SVR prediction..."):
                results = predict_bilstm_hybrid(clean_df)
            render_results(results, "Bi-LSTM base")
            st.success("Prediction completed successfully.")
            st.dataframe(clean_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
else:
    state_key = "tide_df"
    editor_key = "tide_editor"
    row_prefix = "Month"
    init_editor_state(state_key, TIDE_ROWS, row_prefix)

    top_left, top_middle, top_right = st.columns([1, 1, 1.1])
    with top_left:
        st.download_button(
            label="Download 8-row CSV template",
            data=dataframe_to_csv(build_default_df(TIDE_ROWS, row_prefix)),
            file_name="tide_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with top_middle:
        if st.button("Reset TiDE table", use_container_width=True):
            reset_editor_state(state_key, editor_key, TIDE_ROWS, row_prefix)
            st.rerun()
    with top_right:
        next_oil_price = st.number_input(
            "Next-step oil price",
            min_value=MIN_INPUT,
            max_value=MAX_INPUT,
            value=0.00,
            step=0.01,
            format="%.2f",
            help="Used by the SVR correction at the forecast step.",
        )

    with st.form("tide_form"):
        edited_df = render_editor(st.session_state[state_key], editor_key)
        submitted = st.form_submit_button("Predict with TiDE + SVR", type="primary", use_container_width=True)

    st.session_state[state_key] = edited_df

    if submitted:
        try:
            clean_df = validate_input(edited_df, TIDE_ROWS)
            with st.spinner("Running TiDE + SVR prediction..."):
                results = predict_tide_hybrid(clean_df, next_oil_price)
            render_results(results, "TiDE base")
            st.caption(
                f"Auto-derived next-step lags for SVR: Lag1={results['next_lag1']:.6f}, "
                f"Lag2={results['next_lag2']:.6f}"
            )
            st.success("Prediction completed successfully.")
            st.dataframe(clean_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with st.expander("Repository layout"):
    st.code(
        """your-repo/
├─ app.py
├─ requirements.txt
└─ models/
   ├─ bilstm_model.keras
   ├─ svr_model.pkl
   ├─ scaler_features.pkl
   ├─ scaler_target.pkl
   ├─ hybrid_meta.pkl
   ├─ tide_final.pt
   ├─ tide_svr_corrector.pkl
   ├─ tide_target_scaler.pkl
   └─ tide_cov_scaler.pkl""",
        language="text",
    )
