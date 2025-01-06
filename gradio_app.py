import gradio as gr
from pathlib import Path
from scripts.inference import main
from omegaconf import OmegaConf
import argparse
from datetime import datetime

CONFIG_PATH = Path("configs/unet/second_stage.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")


def process_video(
    video_path,
    audio_path,
    guidance_scale,
    sync_loss_weight,
    perceptual_loss_weight,
    recon_loss_weight,
    trepa_loss_weight,
    inference_steps,
    mixed_noise_alpha,
    use_mixed_noise,
    seed,
):
    # Create the temp directory if it doesn't exist
    output_dir = Path("./temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to absolute Path objects and normalize them
    video_file_path = Path(video_path)
    video_path = str(video_file_path.absolute()).replace("\\", "/")
    audio_path = str(Path(audio_path).absolute()).replace("\\", "/")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Set the output path for the processed video
    output_path = str(
        output_dir / f"{video_file_path.stem}_{current_time}.mp4"
    )  # Change the filename as needed

    config = OmegaConf.load(CONFIG_PATH)

    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "sync_loss_weight": sync_loss_weight,
            "perceptual_loss_weight": perceptual_loss_weight,
            "recon_loss_weight": recon_loss_weight,
            "trepa_loss_weight": trepa_loss_weight,
            "inference_steps": inference_steps,
            "mixed_noise_alpha": mixed_noise_alpha,
            "use_mixed_noise": use_mixed_noise,
        }
    )

    # Parse the arguments
    args = create_args(video_path, audio_path, output_path, guidance_scale, seed)

    try:
        result = main(
            config=config,
            args=args,
        )
        print("Processing completed successfully.")
        return output_path  # Ensure the output path is returned
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise gr.Error(f"Error during processing: {str(e)}")


def create_args(
    video_path: str, audio_path: str, output_path: str, guidance_scale: float, seed: int
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)

    return parser.parse_args(
        [
            "--inference_ckpt_path",
            str(CHECKPOINT_PATH.absolute()).replace("\\", "/"),
            "--video_path",
            video_path,
            "--audio_path",
            audio_path,
            "--video_out_path",
            output_path,
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
        ]
    )


# Create Gradio interface
with gr.Blocks(title="LatentSync Video Processing") as demo:
    gr.Markdown(
        """
    # LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync
    Upload a video and audio file to process with LatentSync model.

    <div align="center">
        <strong>Chunyu Li1,2  Chao Zhang1  Weikai Xu1  Jinghui Xie1,†  Weiguo Feng1
        Bingyue Peng1  Weiwei Xing2,†</strong>
    </div>

    <div align="center">
        <strong>1ByteDance   2Beijing Jiaotong University</strong>
    </div>

    <div style="display:flex;justify-content:center;column-gap:4px;">
        <a href="https://github.com/bytedance/LatentSync">
            <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
        </a> 
        <a href="https://arxiv.org/pdf/2412.09262">
            <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
        </a>
        <a href="https://x.com/bdsqlsz">
            <img src="https://img.shields.io/twitter/follow/bdsqlsz">
        </a>
    </div>
    """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input Video")
            audio_input = gr.Audio(label="Input Audio", type="filepath")

            with gr.Row():
                guidance_scale = gr.Slider(
                    minimum=0.1,
                    maximum=3.0,
                    value=1.0,
                    step=0.1,
                    label="Guidance Scale",
                )
                inference_steps = gr.Slider(
                    minimum=1, maximum=50, value=20, step=1, label="Inference Steps"
                )

            with gr.Row():
                sync_loss_weight = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.05,
                    step=0.01,
                    label="Sync Loss Weight",
                )
                perceptual_loss_weight = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.1,
                    step=0.01,
                    label="Perceptual Loss Weight",
                )

            with gr.Row():
                recon_loss_weight = gr.Slider(
                    minimum=0.1,
                    maximum=5.0,
                    value=1.0,
                    step=0.1,
                    label="Reconstruction Loss Weight",
                )
                trepa_loss_weight = gr.Slider(
                    minimum=1, maximum=20, value=10, step=1, label="Trepa Loss Weight"
                )

            with gr.Row():
                mixed_noise_alpha = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Mixed Noise Alpha",
                )
                use_mixed_noise = gr.Checkbox(value=True, label="Use Mixed Noise")

            seed = gr.Number(value=1247, label="Random Seed", precision=0)
            process_btn = gr.Button("Process Video")

        with gr.Column():
            video_output = gr.Video(label="Output Video")

    process_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            audio_input,
            guidance_scale,
            sync_loss_weight,
            perceptual_loss_weight,
            recon_loss_weight,
            trepa_loss_weight,
            inference_steps,
            mixed_noise_alpha,
            use_mixed_noise,
            seed,
        ],
        outputs=video_output,
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=True)
