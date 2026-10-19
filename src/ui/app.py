"""Streamlit UI for the scoring API: http://localhost:8501

Run with `uv run python -m streamlit run src/ui/app.py`; `-m` puts the project root on the
import path so `src.ui` resolves.
"""

import streamlit as st

from src.ui.api_client import ApiClient, ApiError, Health, TrainResult

st.set_page_config(page_title="MLOps Fraud Score", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_client() -> ApiClient:
    """One HTTP client for all sessions, so connections are reused."""
    return ApiClient()


def refresh_health(client: ApiClient) -> None:
    """Stores the health, or the error, in the session. Streamlit reruns the script on every
    click; keeping it in the session means one request per page load, not one per rerun."""
    try:
        st.session_state.health = client.health()
        st.session_state.health_error = None
    except ApiError as error:
        st.session_state.health = None
        st.session_state.health_error = str(error)


def render_header(health: Health | None, error: str | None) -> None:
    title, status = st.columns([3, 2], vertical_alignment="center")
    title.title("MLOps Fraud Score")
    with status:
        if health is None:
            st.badge("API unreachable", icon=":material/error:", color="red")
            st.caption(error)
        elif health["model_loaded"]:
            st.badge("API online · model loaded", icon=":material/check_circle:", color="green")
        else:
            st.badge("API online · no model", icon=":material/warning:", color="orange")
            st.caption("Run a training to register and serve a champion model.")


def render_training(client: ApiClient, api_online: bool) -> None:
    st.subheader("Training")
    st.write("Runs prepare → train → evaluate. The new model is served only if it passes the quality gate.")
    if st.button("Run training", type="primary", disabled=not api_online):
        with st.spinner("Training, this can take a minute…"):
            try:
                st.session_state.train_result = client.train()
                st.session_state.train_error = None
            except ApiError as error:
                st.session_state.train_result = None
                st.session_state.train_error = str(error)
        # Promotion can change whether a model is loaded.
        refresh_health(client)
        st.rerun()

    if st.session_state.get("train_error"):
        st.error(f"Training failed: {st.session_state.train_error}")
    result: TrainResult | None = st.session_state.get("train_result")
    if result:
        render_train_result(result)


def render_train_result(result: TrainResult) -> None:
    if result["promoted"]:
        st.success(f"Model version {result['model_version']} passed the gate and is now served.")
    else:
        st.warning(f"Model version {result['model_version']} was registered but did not pass the gate.")
    columns = st.columns(len(result["metrics"]))
    for column, (name, value) in zip(columns, sorted(result["metrics"].items()), strict=True):
        column.metric(name.replace("_", " ").upper(), f"{value:.4f}")
    st.caption(f"MLflow run `{result['run_id']}`")


client = get_client()
if "health" not in st.session_state:
    refresh_health(client)
render_header(st.session_state.health, st.session_state.health_error)
st.divider()
render_training(client, api_online=st.session_state.health is not None)
