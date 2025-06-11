from ultralytics_utils import (
    format_trained_model,
    get_custom_parser,
    get_kwargs
)

# calling up the main function 
if __name__ == "__main__":
    parser = get_custom_parser()
    args = parser.parse_args()

    kwargs = get_kwargs(args)

    # Format and save trained model
    save_dir = kwargs.pop("save_dir")
    model_dir = kwargs.pop("model_dir")
    model_dir.mkdir(parents=True, exist_ok=True)
    format_trained_model(save_dir, model_dir)
