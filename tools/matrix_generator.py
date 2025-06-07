import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmwrite
import argparse
import os


def generate_strided_mask(n, density_percent):
    max_nnz = int(n * n * (density_percent / 100.0))
    stride = max(1, (n * n) // (2 * max_nnz))  # heuristic for stride

    rows, cols = [], []

    for i in range(n):
        rows.append(i)
        cols.append(i)

        for j in range(i + stride, n, stride):
            rows.append(i)
            cols.append(j)
            if len(rows) >= max_nnz:
                break
        if len(rows) >= max_nnz:
            break

    data = np.random.uniform(0.0, 1.0, size=len(rows)).astype(np.float32)
    return coo_matrix((data, (cols, rows)), shape=(n, n))


def main():
    parser = argparse.ArgumentParser(
        description="Generate a strided sparse matrix and save as .mtx"
    )
    parser.add_argument("--n", type=int, required=True, help="Matrix size (n x n)")
    parser.add_argument(
        "--density",
        type=float,
        required=True,
        help="Density as percentage (e.g., 5 for 5%)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Base output filename (e.g., my_mask.mtx)",
    )
    args = parser.parse_args()

    # Determine relative path to ../data/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))

    base_name = os.path.splitext(os.path.basename(args.output))[0]
    target_dir = os.path.join(data_dir, base_name)
    os.makedirs(target_dir, exist_ok=True)

    output_path = os.path.join(target_dir, f"{base_name}.mtx")

    mask = generate_strided_mask(args.n, args.density)
    mmwrite(output_path, mask)

    print(f"✅ Saved matrix to: {output_path}")
    print(f"   Shape: {args.n}x{args.n}")
    print(f"   Target density: ~{args.density}%")


if __name__ == "__main__":
    main()
