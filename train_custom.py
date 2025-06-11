from ultralytics_utils import (
    format_trained_model,
    get_custom_parser,
    get_kwargs,
    setup_mlflow,
    YOLO
)

# calling the main function 
if __name__ == "__main__":
    parser = get_custom_parser()
    args = parser.parse_args()

    kwargs = get_kwargs(args)

    # Setup MLFlow
    project = kwargs.pop("project")
    run = kwargs.pop("run")
    setup_mlflow(project, run)

    # Create final model directory
    model_dir = kwargs.pop("model_dir")

    # Setup YOLO model
    model = kwargs.pop("weights") or kwargs.pop("model")
    yolo = YOLO(model=model, task=kwargs.pop("task"), verbose=True)

    # Train model
    yolo.train(**kwargs)

    # Format and save trained model
    save_dir = kwargs.pop("save_dir")
    model_dir.mkdir(parents=True, exist_ok=True)
    format_trained_model(save_dir, model_dir)
