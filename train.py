from ultralytics_utils import get_default_parser, YOLO


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()

    yolo = YOLO(model=args.model, task=args.task, verbose=True)

    if args.weights:
        yolo.load(args.weights)

    yolo.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.train_epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        batch=args.batch,
        device=args.device,
        save_dir=args.save_dir,
        conf=args.conf,
        single_cls=args.single_cls,
        plots=True,
        visualize=True,
    )
