import plotly.graph_objects as go

from pinn.visualization import TrainingDashboard, export_animation


def test_dashboard_records_losses(tmp_path):
    cfg = object()
    dash = TrainingDashboard(cfg, update_frequency=1)
    for step in range(3):
        dash.update_callback(step, {"loss": float(step)})
    assert list(dash.history.steps) == [0, 1, 2]
    assert list(dash.history.losses) == [0.0, 1.0, 2.0]

    fig = go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])])
    out = tmp_path / "anim.html"
    export_animation(fig, str(out))
    assert out.exists()
