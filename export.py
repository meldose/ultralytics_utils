from ultralytics_utils import get_export_parser, YOLO


if __name__ == "__main__":
    parser = get_export_parser()
    args = parser.parse_args()

    yolo = YOLO(model=args.weights, verbose=True)

    if args.weights:
        yolo.load(args.weights)

    yolo.export(
        format=args.format,
        half=args.half,
        int8=args.int8,
        workspace=args.workspace,
        nms=args.nms,
        simplify=args.simplify,
    )
