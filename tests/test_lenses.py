import numpy as np
import pytest

from cx_tda_engine import LENS_REGISTRY, BaseLens, get_lens, lens_catalog, register_lens
from cx_tda_engine.lenses.umap_lens import umap_available


@pytest.mark.parametrize("name", ["pca", "density", "eccentricity", "feature"])
def test_lens_fit_transform(name, X_norm):
    out = get_lens(name).fit_transform(X_norm)
    assert out.shape == (len(X_norm),)
    assert not np.isnan(out).any()


def test_registry_complete_and_catalog():
    assert set(LENS_REGISTRY) >= {"pca", "umap", "density", "eccentricity", "feature"}
    names = {e["name"] for e in lens_catalog()}
    assert names == set(LENS_REGISTRY)
    assert all(e["description"] for e in lens_catalog())


def test_unknown_lens_raises():
    with pytest.raises(ValueError, match="Unknown lens"):
        get_lens("nope")


def test_feature_lens_clamps_index(X_norm):
    np.testing.assert_array_equal(get_lens("feature", feature_index=99).fit_transform(X_norm), X_norm[:, -1])


def test_eccentricity_is_seeded_when_sampling():
    X = np.random.default_rng(1).normal(size=(1500, 4))
    a = get_lens("eccentricity", sample_n=200).fit_transform(X)
    b = get_lens("eccentricity", sample_n=200).fit_transform(X)
    np.testing.assert_allclose(a, b)


def test_umap_lens_falls_back_to_pca_without_umap(X_norm):
    out = get_lens("umap").fit_transform(X_norm)
    assert out.shape == (len(X_norm),)
    if not umap_available():
        np.testing.assert_allclose(out, get_lens("pca").fit_transform(X_norm))


def test_register_custom_lens(X_norm):
    class Sum(BaseLens):
        name = "sum"
        description = "row sum"

        def fit_transform(self, X):
            return X.sum(axis=1)

    register_lens("sum", Sum)
    try:
        np.testing.assert_allclose(get_lens("sum").fit_transform(X_norm), X_norm.sum(axis=1))
    finally:
        del LENS_REGISTRY["sum"]
