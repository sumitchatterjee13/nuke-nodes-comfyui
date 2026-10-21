"""OCIO config resolution: $OCIO URIs and the built-in config candidates."""
import pytest


@pytest.fixture(scope="module")
def ocio(pack):
    from nuke_nodes import ocio_config
    if not ocio_config.OCIO_AVAILABLE:
        pytest.skip("OpenColorIO not installed")
    return ocio_config


def _installed_builtin_names(ocio):
    registry = ocio.OCIO.BuiltinConfigRegistry()
    names = []
    for entry in registry.getBuiltinConfigs():
        names.append(entry[0] if isinstance(entry, tuple) else str(entry))
    return names


def test_builtin_candidates_cover_installed_studio_configs(ocio):
    installed = [n for n in _installed_builtin_names(ocio) if n.startswith("studio-config-")]
    assert installed, "this OpenColorIO ships no built-in studio config"
    missing = [n for n in installed if n not in ocio.BUILTIN_CONFIGS]
    assert not missing, f"BUILTIN_CONFIGS does not know {missing}"


def test_newest_known_builtin_is_chosen(ocio, monkeypatch):
    monkeypatch.delenv("OCIO", raising=False)
    installed = _installed_builtin_names(ocio)
    expected = next(n for n in ocio.BUILTIN_CONFIGS if n in installed)
    config, source = ocio.resolve_ocio_config()
    assert config is not None
    assert source == f"built-in {expected}"


def test_ocio_env_accepts_builtin_uri(ocio, monkeypatch):
    uri = "ocio://studio-config-latest"
    try:
        ocio.OCIO.Config.CreateFromFile(uri)
    except Exception:
        pytest.skip("this OpenColorIO cannot resolve ocio:// URIs")
    monkeypatch.setenv("OCIO", uri)
    config, source = ocio.resolve_ocio_config()
    assert config is not None, "$OCIO=ocio://... must load the built-in config"
    assert source == f"$OCIO={uri}"
