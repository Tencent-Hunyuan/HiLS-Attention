import argparse
from veomni.models import save_model_weights
from veomni.checkpoint import ckpt_to_state_dict
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert checkpoint to HuggingFace format")

    parser.add_argument(
        "--save_checkpoint_path",
        type=str,
        required=True,
        help="Path to the saved checkpoint directory"
    )

    args = parser.parse_args()

    hf_weights_path = os.path.join(args.save_checkpoint_path, "hf_ckpt")
    model_state_dict = ckpt_to_state_dict(
        save_checkpoint_path=args.save_checkpoint_path,
        ckpt_manager='dcp',
    )
    save_model_weights(hf_weights_path, model_state_dict)
