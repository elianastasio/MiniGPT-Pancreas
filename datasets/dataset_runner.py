import subprocess
import os
import argparse

def run_script(script_dir, script_name, extra_args=None):
    cwd = os.getcwd()
    os.chdir(script_dir)  # Change to the directory of the script
    cmd = ["python", script_name]
    if extra_args:
        cmd.extend(extra_args)
    print(f"Running: {' '.join(cmd)} in {script_dir}")
    subprocess.run(cmd, check=True)
    os.chdir(cwd)  # Return to original directory

def main(balanced=False):
    balanced_arg = ["--balanced"] if balanced else None

    # === MSD_PANCREAS ===
    msd_dir = "MSD/scripts"
    run_script(msd_dir, "MSD_generate_slices_info.py")
    run_script(msd_dir, "MSD_generate_jsons.py", balanced_arg)
    run_script(msd_dir, "MSD_save_slices.py", balanced_arg)
    run_script(msd_dir, "MSD_tumor_generate_jsons.py", balanced_arg)
    run_script(msd_dir, "MSD_tumor_save_slices.py", balanced_arg)

    # === NIH_PANCREAS ===
    nih_dir = "NIH/scripts"
    run_script(nih_dir, "NIH_generate_slices_info.py")
    run_script(nih_dir, "NIH_generate_jsons.py", balanced_arg)
    run_script(nih_dir, "NIH_save_slices.py", balanced_arg)

    # === Tumor Classification ===
    tc_dir = "TC/scripts"
    run_script(tc_dir, "TC_generate_jsons.py", balanced_arg)
    run_script(tc_dir, "TC_save_slices.py", balanced_arg)

    # === AbdomenCT-1K ===
    abd_dir = "ABD/scripts"
    run_script(abd_dir, "ABD_generate_slices_info.py")
    run_script(abd_dir, "ABD_generate_jsons.py")
    run_script(abd_dir, "ABD_save_slices.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all dataset preparation scripts with optional balanced flag.")
    parser.add_argument("--balanced", action="store_true", help="Add non-pancreas or non-tumor slices for balanced training.")
    args = parser.parse_args()
    main(balanced=args.balanced)
