import h5py
import argparse

# Philip Mocz (2025)
# List headers of an HDF5 file.


def main(filename="fdm_1mpc_256_m2.5e-22_z127_ic.hdf5"):
    with h5py.File(filename, "r") as f:

        def print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f"  Attr: {key} = {val}")

        f.visititems(print_attrs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List headers of an HDF5 file.")
    parser.add_argument(
        "filename",
        nargs="?",
        default="fdm_1mpc_256_m2.5e-22_z127_ic.hdf5",
        help="Path to the HDF5 file",
    )
    args = parser.parse_args()
    main(args.filename)
