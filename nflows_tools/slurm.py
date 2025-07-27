#!/usr/bin/env python3

import argparse
from pathlib import Path

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --output={log_dir}%a.out
#SBATCH --error={log_dir}%a.err
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --array=1-{repeats}
#SBATCH --exclusive
#SBATCH --time={time_limit}

# --- Validation block ---
# Get physical cores on the node (physical cores = sockets * cores per socket)
PHYSICAL_CORES=$(nproc)

# Get what SLURM allocated
ALLOCATED_CORES=${{SLURM_CPUS_PER_TASK:-0}}

# Compare
if [[ "$ALLOCATED_CORES" -ne "$PHYSICAL_CORES" ]]; then
  echo "ERROR: SLURM allocated $ALLOCATED_CORES CPUs, but $PHYSICAL_CORES physical cores are available."
  echo "Make sure to submit the job with --cpus-per-task=$PHYSICAL_CORES"
  exit 1
fi

echo "Validation OK: SLURM_CPUS_PER_TASK=$ALLOCATED_CORES matches physical cores=$PHYSICAL_CORES"

# --- Run your program here ---

export NFLOWS_CONFIG_FILE="{config_file}"
export NFLOWS_OUTPUT_FILE_NAME="$(echo "$NFLOWS_CONFIG_FILE" | sed 's|/config/|/output/|' | sed "s|config.json|${{SLURM_ARRAY_TASK_ID}}.yaml|")"

START_TIME=$(date +%s.%N)
{execute_command}
EXECUTABLE_STATUS=$?
END_TIME=$(date +%s.%N)
ELAPSED_TIME_SEC=$(echo "$END_TIME - $START_TIME" | bc)

{validate_command} "$NFLOWS_OUTPUT_FILE_NAME"
VALIDATE_STATUS=$?

if [ $EXECUTABLE_STATUS -eq 0 ] && [ $VALIDATE_STATUS -eq 0 ]; then
    printf "[SUCCESS] $NFLOWS_CONFIG_FILE (Time: %.3f s)\\n" "$ELAPSED_TIME_SEC"
else
    printf "[FAILED] $NFLOWS_CONFIG_FILE (Execute: $EXECUTABLE_STATUS, Validate: $VALIDATE_STATUS, Time: %.3f s)\\n" "$ELAPSED_TIME_SEC"
fi
"""

def main():
    parser = argparse.ArgumentParser(description="Generate a SLURM submit file from a template.")
    parser.add_argument("--job", help="The name of the SLURM job.")
    parser.add_argument("--config_file", help="Directory containing config.json")
    parser.add_argument("--log_dir", default="logs", help="Directory to store SLURM logs")
    parser.add_argument("--execute_command", help="Name of the executable to run")
    parser.add_argument("--validate_command", help="Validation command to run after execution")
    parser.add_argument("--output", default="submit.sh", help="Output path for the SLURM submit file")
    parser.add_argument("--time_limit", default="01:00:00", help="Time limit for the job (default: 1 hour)")
    parser.add_argument("--repeats", type=int, default=1, help="Number of job array repeats")

    args = parser.parse_args()

    content = TEMPLATE.format(
        job=args.job,
        config_file=args.config_file,
        execute_command=args.execute_command,
        validate_command=args.validate_command,
        log_dir=args.log_dir,
        repeats=args.repeats
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(f"[INFO] SLURM submission script written to {output_path}")

if __name__ == "__main__":
    main()
