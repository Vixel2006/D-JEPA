def get_config(overrides=None):
    """
    Returns the default configuration dictionary, updated with any overrides.
    """
    config = {
        "num_samples": 2000,
        "batch_size": 64,
        "epochs": 5,
        "lr": 1e-4,
        "beta_teacher": 0.5,
        "beta_student": 0.1,
        "seed": 42,
        "latent_dim": 128,
        "img_in_dim": 512,
        "txt_in_dim": 384,
        "d_model": 256,
        "nhead": 4,
        "output_dir": "outputs",
    }
    if overrides:
        config.update(overrides)
    return config
