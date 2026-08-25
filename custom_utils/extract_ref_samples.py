import pandas as pd
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input reference file")
    parser.add_argument("--output", type=str, required=True, help="Output file to put list of smiles into")
    parser.add_argument(
        "--device_component",
        type=str,
        choices=["acc", "don"],
        required=True,
        help="Device component to work on",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=";")

    smiles = df[f"SMILES_{args.device_component}"]

    with open(args.output, "w") as f:
        data = smiles.str.cat(sep="\n")
        f.write(data)
