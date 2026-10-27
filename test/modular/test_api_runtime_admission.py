from types import SimpleNamespace

from mstar.api_server import entrypoint
from mstar.conductor import conductor
from mstar.model import registry


def test_spawn_target_forwards_admission_alongside_upstream_metrics(tmp_path, monkeypatch):
    path = tmp_path / "deployment.yaml"
    path.write_text("model: dummy\n")
    context = object()
    calls = []
    monkeypatch.setattr(entrypoint.signal, "signal", lambda *a: None)
    monkeypatch.setattr(entrypoint, "watch_parent", lambda *a: None)
    monkeypatch.setattr(entrypoint, "model_init_kwargs", lambda name: {})
    monkeypatch.setattr(registry, "get_model_class", lambda name: lambda **kw: object())
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run=lambda: calls.append("run"), shutdown=lambda: calls.append("shutdown"))
    monkeypatch.setattr(conductor, "Conductor", create)
    entrypoint._conductor_process_target(
        "dummy", str(path), "unused", runtime_admission=context, status_interval_s=0.25,
    )
    assert calls[0]["runtime_admission"] is context
    assert calls[0]["status_interval_s"] == 0.25
    assert calls[1:] == ["run", "shutdown"]
