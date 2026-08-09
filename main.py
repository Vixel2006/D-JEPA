import sys
import os
import argparse
import random
from typing import Dict, Any

import torch
import numpy as np
from torch.utils.data import DataLoader

# Ensure package directory is in sys.path so absolute imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config
from src.data import SyntheticPairedData, make_flickr8k_features_dataloader
from src.models import (
    ImageEncoder,
    TextEncoder,
    CLIPModel,
    SSDTeacher,
    ImageStudent,
    TextStudent,
)
from src.training import train_clip, train_ssd_teacher, train_ssd_students
from src.evaluation import (
    extract_embeddings,
    compute_modality_gap,
    compute_retrieval_metrics,
    compute_latent_variance,
    get_retrieval_examples,
    plot_tsne,
)


def seed_everything(seed: int = 42) -> None:
    """
    Seeds random number generators for NumPy, Python, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_evaluation(
    model_or_models: Any,
    dataloader: DataLoader,
    model_type: str,
    device: torch.device,
    output_dir: str,
    tsne_filename: str,
) -> Dict[str, Any]:
    """
    Runs the full evaluation suite and returns the computed metrics.
    """
    img_embs, txt_embs, shared_latents = extract_embeddings(
        model_or_models, dataloader, device=device, model_type=model_type
    )

    gap_metrics = compute_modality_gap(img_embs, txt_embs)
    ret_metrics = compute_retrieval_metrics(img_embs, txt_embs)
    latent_var = compute_latent_variance(img_embs)

    # Save t-SNE plot
    os.makedirs(output_dir, exist_ok=True)
    tsne_path = os.path.join(output_dir, tsne_filename)
    plot_tsne(img_embs, txt_embs, shared_latents, tsne_path, subset_size=500)

    examples = get_retrieval_examples(img_embs, txt_embs, num_queries=5)

    return {
        "gap": gap_metrics["overall"],
        "retrieval": ret_metrics,
        "variance": latent_var,
        "examples": examples,
    }


def print_results(model_name: str, res: Dict[str, Any]) -> None:
    """
    Formatted print for evaluation results.
    """
    print(
        f"\n================ {model_name.upper()} EVALUATION RESULTS ================"
    )
    print(f"Modality Gap (Euclidean): {res['gap']:.6f}")
    print(f"Latent Variance:          {res['variance']:.6f}")
    print("Retrieval Metrics:")
    ret = res["retrieval"]
    print(
        f"  Image-to-Text: R@1: {ret['i2t_r1']:.1f}%, R@5: {ret['i2t_r5']:.1f}%, R@10: {ret['i2t_r10']:.1f}%"
    )
    print(
        f"  Text-to-Image: R@1: {ret['t2i_r1']:.1f}%, R@5: {ret['t2i_r5']:.1f}%, R@10: {ret['t2i_r10']:.1f}%"
    )

    print("\nRetrieval Examples (Image-to-Text):")
    for idx, ex in enumerate(res["examples"]["i2t"]):
        retrieved = ", ".join(
            [
                f"{r_idx} (sim: {sim:.3f})"
                for r_idx, sim in zip(ex["retrieved_indices"], ex["similarities"])
            ]
        )
        print(f"  Query Img Index {ex['query_idx']} -> {retrieved}")


def run_test_synthetic(beta_teacher: float = 0.5, beta_student: float = 0.1) -> None:
    """
    Critical end-to-end verification test on synthetic data.
    Runs 2 training epochs for CLIP, SSD Teacher, and SSD Students,
    then evaluates and prints comparison metrics.
    """
    print("================ RUNNING SYNTHETIC VERIFICATION TEST ================")
    device = torch.device("cpu")  # Verify correctness on CPU
    print(f"Verification will run on device: {device}")

    config = get_config(
        {
            "num_samples": 2000,
            "batch_size": 64,
            "epochs": 2,
            "lr": 1e-3,
            "beta_teacher": beta_teacher,
            "beta_student": beta_student,
            "seed": 42,
        }
    )

    seed_everything(config["seed"])

    # 1. Generate Synthetic Data
    print("\nGenerating synthetic data splits...")
    train_dataset = SyntheticPairedData(
        num_samples=config["num_samples"],
        d_z=config["latent_dim"],
        img_dim=config["img_in_dim"],
        txt_dim=config["txt_in_dim"],
        seed=config["seed"],
    )

    # Use distinct seed 43 for evaluation to prevent leakage
    eval_dataset = SyntheticPairedData(
        num_samples=500,
        d_z=config["latent_dim"],
        img_dim=config["img_in_dim"],
        txt_dim=config["txt_in_dim"],
        seed=config["seed"] + 1,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # 2. Train CLIP Baseline
    print("\n--- Training CLIP Baseline ---")
    img_enc = ImageEncoder(
        input_dim=config["img_in_dim"],
        d_model=config["d_model"],
        d_z=config["latent_dim"],
    )
    txt_enc = TextEncoder(
        input_dim=config["txt_in_dim"],
        d_model=config["d_model"],
        d_z=config["latent_dim"],
    )
    clip_model = CLIPModel(img_enc, txt_enc)

    clip_model = train_clip(config, clip_model, train_loader, device=device)
    clip_res = run_evaluation(
        clip_model,
        eval_loader,
        model_type="clip",
        device=device,
        output_dir=config["output_dir"],
        tsne_filename="tsne_clip_test.png",
    )
    print_results("CLIP Baseline", clip_res)

    # 3. Train SSD Teacher
    print("\n--- Training SSD Teacher VAE ---")
    teacher = SSDTeacher(
        img_in_dim=config["img_in_dim"],
        txt_in_dim=config["txt_in_dim"],
        d_model=config["d_model"],
        d_z=config["latent_dim"],
        nhead=config["nhead"],
    )
    teacher = train_ssd_teacher(config, teacher, train_loader, device=device)

    # 4. Distill SSD Students
    print("\n--- Distilling SSD Student Encoders ---")
    img_student = ImageStudent(
        input_dim=config["img_in_dim"],
        d_model=config["d_model"],
        d_z=config["latent_dim"],
        nhead=config["nhead"],
    )
    txt_student = TextStudent(
        input_dim=config["txt_in_dim"],
        d_model=config["d_model"],
        d_z=config["latent_dim"],
        nhead=config["nhead"],
    )
    img_student, txt_student = train_ssd_students(
        config, teacher, img_student, txt_student, train_loader, device=device
    )

    ssd_res = run_evaluation(
        (img_student, txt_student),
        eval_loader,
        model_type="ssd",
        device=device,
        output_dir=config["output_dir"],
        tsne_filename="tsne_ssd_test.png",
    )
    print_results("SSD Students (Distilled)", ssd_res)

    # 5. Contrast and Summary
    print("\n================ COMPARISON SUMMARY ================")
    print(f"Metric                 | CLIP Baseline | SSD Students (Proposed)")
    print("-" * 65)
    print(
        f"Modality Gap (v)       | {clip_res['gap']:>13.6f} | {ssd_res['gap']:>23.6f}"
    )
    print(
        f"Latent Variance        | {clip_res['variance']:>13.6f} | {ssd_res['variance']:>23.6f}"
    )
    print(
        f"Image->Text Recall@1   | {clip_res['retrieval']['i2t_r1']:>12.1f}% | {ssd_res['retrieval']['i2t_r1']:>22.1f}%"
    )
    print(
        f"Text->Image Recall@1   | {clip_res['retrieval']['t2i_r1']:>12.1f}% | {ssd_res['retrieval']['t2i_r1']:>22.1f}%"
    )
    print("====================================================")
    print("Verification test passed successfully! No errors found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal Representation Learning: CLIP vs Sufficient Statistic Distillation (SSD)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments
    def add_common_training_args(subp):
        subp.add_argument(
            "--epochs", type=int, default=5, help="Number of epochs to train"
        )
        subp.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
        subp.add_argument("--batch_size", type=int, default=64, help="Batch size")
        subp.add_argument("--seed", type=int, default=42, help="Random seed")

    # train_clip parser
    parser_clip = subparsers.add_parser("train_clip", help="Train CLIP baseline model")
    add_common_training_args(parser_clip)
    parser_clip.add_argument(
        "--save_path",
        type=str,
        default="checkpoints/clip_model.pt",
        help="Path to save the model",
    )
    parser_clip.add_argument(
        "--dataset",
        type=str,
        choices=["synthetic", "flickr8k"],
        default="synthetic",
        help="Data source to train on",
    )

    # train_teacher parser
    parser_teacher = subparsers.add_parser(
        "train_teacher", help="Train SSD cross-modal teacher"
    )
    add_common_training_args(parser_teacher)
    parser_teacher.add_argument(
        "--beta_teacher",
        type=float,
        default=0.5,
        help="KL loss scaling coefficient beta for teacher",
    )
    parser_teacher.add_argument(
        "--save_path",
        type=str,
        default="checkpoints/ssd_teacher.pt",
        help="Path to save teacher checkpoint",
    )
    parser_teacher.add_argument(
        "--dataset",
        type=str,
        choices=["synthetic", "flickr8k"],
        default="synthetic",
        help="Data source to train on",
    )

    # distill parser
    parser_distill = subparsers.add_parser(
        "distill", help="Distill SSD student models from a trained teacher"
    )
    add_common_training_args(parser_distill)
    parser_distill.add_argument(
        "--beta_student",
        type=float,
        default=0.1,
        help="KL loss scaling coefficient beta for student bottleneck",
    )
    parser_distill.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained teacher checkpoint",
    )
    parser_distill.add_argument(
        "--save_path_img",
        type=str,
        default="checkpoints/ssd_student_img.pt",
        help="Path to save image student",
    )
    parser_distill.add_argument(
        "--save_path_txt",
        type=str,
        default="checkpoints/ssd_student_txt.pt",
        help="Path to save text student",
    )
    parser_distill.add_argument(
        "--dataset",
        type=str,
        choices=["synthetic", "flickr8k"],
        default="synthetic",
        help="Data source to distill on",
    )

    # eval parser
    parser_eval = subparsers.add_parser("eval", help="Evaluate trained representations")
    parser_eval.add_argument(
        "--model",
        type=str,
        choices=["clip", "ssd"],
        required=True,
        help="Representation method to evaluate",
    )
    parser_eval.add_argument(
        "--checkpoint",
        type=str,
        help="Path to CLIP checkpoint (required if model=clip)",
    )
    parser_eval.add_argument(
        "--student_checkpoint_img",
        type=str,
        help="Path to ImageStudent checkpoint (required if model=ssd)",
    )
    parser_eval.add_argument(
        "--student_checkpoint_txt",
        type=str,
        help="Path to TextStudent checkpoint (required if model=ssd)",
    )
    parser_eval.add_argument(
        "--batch_size", type=int, default=64, help="Evaluation batch size"
    )
    parser_eval.add_argument(
        "--dataset",
        type=str,
        choices=["synthetic", "flickr8k"],
        default="synthetic",
        help="Data source to evaluate on",
    )
    parser_eval.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "test"],
        default="test",
        help="Flickr8k split to evaluate (used when --dataset flickr8k)",
    )
    parser_eval.add_argument(
        "--seed", type=int, default=43, help="Seed for evaluation data generation"
    )

    # test_synthetic parser
    parser_test = subparsers.add_parser(
        "test_synthetic", help="Run self-contained end-to-end CPU verification test"
    )
    parser_test.add_argument(
        "--beta_teacher",
        type=float,
        default=0.5,
        help="KL loss scaling coefficient beta for teacher",
    )
    parser_test.add_argument(
        "--beta_student",
        type=float,
        default=0.1,
        help="KL loss scaling coefficient beta for student bottleneck",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "train_clip":
        config = get_config(
            {
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "seed": args.seed,
            }
        )
        seed_everything(config["seed"])
        if args.dataset == "flickr8k":
            config.update({"img_in_dim": 512, "txt_in_dim": 512})
            dataloader = make_flickr8k_features_dataloader(
                split="train", batch_size=config["batch_size"], shuffle=True
            )
        else:
            dataset = SyntheticPairedData(
                num_samples=config["num_samples"], seed=config["seed"]
            )
            dataloader = DataLoader(
                dataset, batch_size=config["batch_size"], shuffle=True
            )

        img_enc = ImageEncoder(
            input_dim=config["img_in_dim"],
            d_model=config["d_model"],
            d_z=config["latent_dim"],
        )
        txt_enc = TextEncoder(
            input_dim=config["txt_in_dim"],
            d_model=config["d_model"],
            d_z=config["latent_dim"],
        )
        model = CLIPModel(img_enc, txt_enc)

        model = train_clip(config, model, dataloader, device=device)

        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": config}, args.save_path)
        print(f"Saved CLIP baseline model checkpoint to: {args.save_path}")

    elif args.command == "train_teacher":
        config = get_config(
            {
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "beta_teacher": args.beta_teacher,
                "seed": args.seed,
            }
        )
        seed_everything(config["seed"])
        if args.dataset == "flickr8k":
            config.update({"img_in_dim": 512, "txt_in_dim": 512})
            dataloader = make_flickr8k_features_dataloader(
                split="train", batch_size=config["batch_size"], shuffle=True
            )
        else:
            dataset = SyntheticPairedData(
                num_samples=config["num_samples"], seed=config["seed"]
            )
            dataloader = DataLoader(
                dataset, batch_size=config["batch_size"], shuffle=True
            )

        model = SSDTeacher(
            img_in_dim=config["img_in_dim"],
            txt_in_dim=config["txt_in_dim"],
            d_model=config["d_model"],
            d_z=config["latent_dim"],
            nhead=config["nhead"],
        )

        model = train_ssd_teacher(config, model, dataloader, device=device)

        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": config}, args.save_path)
        print(f"Saved SSD Teacher model checkpoint to: {args.save_path}")

    elif args.command == "distill":
        if not os.path.exists(args.teacher_checkpoint):
            raise FileNotFoundError(
                f"Teacher checkpoint not found at: {args.teacher_checkpoint}"
            )

        teacher_ckpt = torch.load(args.teacher_checkpoint, map_location="cpu")
        config = teacher_ckpt["config"]
        config.update(
            {
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "beta_student": args.beta_student,
                "seed": args.seed,
            }
        )

        seed_everything(config["seed"])
        if args.dataset == "flickr8k":
            dataloader = make_flickr8k_features_dataloader(
                split="train", batch_size=config["batch_size"], shuffle=True
            )
        else:
            dataset = SyntheticPairedData(
                num_samples=config["num_samples"], seed=config["seed"]
            )
            dataloader = DataLoader(
                dataset, batch_size=config["batch_size"], shuffle=True
            )

        teacher = SSDTeacher(
            img_in_dim=config["img_in_dim"],
            txt_in_dim=config["txt_in_dim"],
            d_model=config["d_model"],
            d_z=config["latent_dim"],
            nhead=config["nhead"],
        )
        teacher.load_state_dict(teacher_ckpt["state_dict"])

        img_student = ImageStudent(
            input_dim=config["img_in_dim"],
            d_model=config["d_model"],
            d_z=config["latent_dim"],
            nhead=config["nhead"],
        )
        txt_student = TextStudent(
            input_dim=config["txt_in_dim"],
            d_model=config["d_model"],
            d_z=config["latent_dim"],
            nhead=config["nhead"],
        )

        img_student, txt_student = train_ssd_students(
            config, teacher, img_student, txt_student, dataloader, device=device
        )

        os.makedirs(os.path.dirname(args.save_path_img), exist_ok=True)
        torch.save(
            {"state_dict": img_student.state_dict(), "config": config},
            args.save_path_img,
        )
        torch.save(
            {"state_dict": txt_student.state_dict(), "config": config},
            args.save_path_txt,
        )
        print(
            f"Saved distilled SSD students to: {args.save_path_img} and {args.save_path_txt}"
        )

    elif args.command == "eval":
        if args.dataset == "flickr8k":
            eval_dataloader = make_flickr8k_features_dataloader(
                split=args.split, batch_size=args.batch_size, shuffle=False
            )
        else:
            eval_dataset = SyntheticPairedData(num_samples=1000, seed=args.seed)
            eval_dataloader = DataLoader(
                eval_dataset, batch_size=args.batch_size, shuffle=False
            )

        if args.model == "clip":
            if not args.checkpoint:
                raise ValueError(
                    "Must specify --checkpoint when evaluating clip model."
                )
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            config = ckpt["config"]

            img_enc = ImageEncoder(
                input_dim=config["img_in_dim"],
                d_model=config["d_model"],
                d_z=config["latent_dim"],
            )
            txt_enc = TextEncoder(
                input_dim=config["txt_in_dim"],
                d_model=config["d_model"],
                d_z=config["latent_dim"],
            )
            model = CLIPModel(img_enc, txt_enc)
            model.load_state_dict(ckpt["state_dict"])

            model_or_models = model
            tsne_name = "tsne_clip.png"
        else:  # ssd
            if not args.student_checkpoint_img or not args.student_checkpoint_txt:
                raise ValueError(
                    "Must specify both --student_checkpoint_img and --student_checkpoint_txt for ssd evaluation."
                )

            ckpt_img = torch.load(args.student_checkpoint_img, map_location="cpu")
            ckpt_txt = torch.load(args.student_checkpoint_txt, map_location="cpu")
            config = ckpt_img["config"]

            img_student = ImageStudent(
                input_dim=config["img_in_dim"],
                d_model=config["d_model"],
                d_z=config["latent_dim"],
                nhead=config["nhead"],
            )
            txt_student = TextStudent(
                input_dim=config["txt_in_dim"],
                d_model=config["d_model"],
                d_z=config["latent_dim"],
                nhead=config["nhead"],
            )

            img_student.load_state_dict(ckpt_img["state_dict"])
            txt_student.load_state_dict(ckpt_txt["state_dict"])

            model_or_models = (img_student, txt_student)
            tsne_name = "tsne_ssd.png"

        res = run_evaluation(
            model_or_models,
            eval_dataloader,
            model_type=args.model,
            device=device,
            output_dir=config["output_dir"],
            tsne_filename=tsne_name,
        )
        print_results(args.model, res)

    elif args.command == "test_synthetic":
        run_test_synthetic(args.beta_teacher, args.beta_student)


if __name__ == "__main__":
    main()
