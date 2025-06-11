from ultralytics_utils import get_custom_parser, get_kwargs, YOLO # importing get_custom parser , get_kwargs and YOLO 


# calling the main function 
if __name__ == "__main__":
    parser = get_custom_parser()
    args = parser.parse_args()

    kwargs = get_kwargs(args)
    kwargs.pop("run")
    kwargs.pop("project")

    # Setup YOLO model
    model_dir = kwargs.pop("model_dir")
    model = kwargs.pop("weights") or kwargs.pop("model")
    yolo = YOLO(model=model, task=kwargs.pop("task"), verbose=True)

    num_gpus = len(args.device.split(","))
    kwargs["batch"] = args.batch * num_gpus

    yolo.val(**kwargs)
