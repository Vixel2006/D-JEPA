from pathlib import Path

import numpy as np
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def encode_vision(encoder, images, image_size):
    images = images.permute(0, 3, 1, 2)
    images = F.interpolate(images, size=(image_size, image_size), mode="bilinear")
    embedding, shared, unique = encoder(images.unsqueeze(1))
    return embedding, shared.squeeze(0), unique.squeeze(0)


def encode_vision_backbone(encoder, images, image_size):
    images = images.permute(0, 3, 1, 2)
    images = F.interpolate(images, size=(image_size, image_size), mode="bilinear")
    return encoder.encode_backbone(images.unsqueeze(1))


def factor_probe_metrics(train_features, train_factors, test_features, test_factors):
    """Fit small linear probes and score held-out synthetic factors."""
    scores = {}
    for representation, train_x in train_features.items():
        test_x = test_features[representation]
        for factor, train_y in train_factors.items():
            test_y = test_factors[factor]
            train_y = np.asarray(train_y)
            test_y = np.asarray(test_y)
            discrete = train_y.dtype == np.bool_ or (
                np.issubdtype(train_y.dtype, np.integer)
                and len(np.unique(train_y)) <= 20
            )

            if discrete:
                model = make_pipeline(
                    StandardScaler(), LogisticRegression(max_iter=1000)
                )
                model.fit(train_x, train_y.reshape(-1))
                score = accuracy_score(test_y.reshape(-1), model.predict(test_x))
                metric = "accuracy"
            else:
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(train_x, train_y)
                score = r2_score(
                    test_y, model.predict(test_x), multioutput="uniform_average"
                )
                metric = "r2"

            scores[f"probe/{representation}/{factor}_{metric}"] = float(score)
    return scores


def representation_health(features):
    """Report mean feature spread and effective rank for collapse checks."""
    scores = {}
    for name, values in features.items():
        values = np.asarray(values, dtype=np.float64)
        scores[f"health/{name}_mean_std"] = float(values.std(axis=0).mean())
        covariance = np.cov(values, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(np.atleast_2d(covariance)).clip(min=0)
        probabilities = eigenvalues / max(eigenvalues.sum(), 1e-12)
        positive = probabilities[probabilities > 0]
        scores[f"health/{name}_effective_rank"] = float(
            np.exp(-(positive * np.log(positive)).sum())
        )
    return scores


def joint_synergy_metrics(train_features, train_factors, test_features, test_factors):
    """Measure nonlinear synergy recovery from paired modality features."""
    scores = {}
    for representation in (
        "c_joint",
        "u_joint",
        "all_joint",
        "embedding_joint",
    ):
        if representation not in train_features:
            continue
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=0,
            ),
        )
        model.fit(
            train_features[representation],
            np.asarray(train_factors["synergy"]).reshape(-1),
        )
        prediction = model.predict(test_features[representation])
        scores[f"probe_nonlinear/{representation}/synergy_r2"] = float(
            r2_score(np.asarray(test_factors["synergy"]).reshape(-1), prediction)
        )
    return scores


def save_pid_inputs(path, features, factors):
    """Save paired representations and known targets for offline Flow-PID."""
    arrays = {name: np.asarray(value) for name, value in features.items()}
    for alias, source in {
        "x1_c": "c_vision",
        "x2_c": "c_text",
        "x1_u": "u_vision",
        "x2_u": "u_text",
    }.items():
        if source in features:
            arrays[alias] = np.asarray(features[source])
    arrays.update(
        {f"target_{name}": np.asarray(value) for name, value in factors.items()}
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
