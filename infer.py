from ultralytics_utils import get_default_parser, YOLO


# calling up the main function 
if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()

    yolo = YOLO(model=args.weights, verbose=True)

    res = yolo(
        args.data,
        save=True,
        save_dir=args.save_dir,
        visualize=False,
        device=args.device,
    )

    for idx, r in enumerate(res):
        print(f"image {idx}: {r.path}")
        print(r)
